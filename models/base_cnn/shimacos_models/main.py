import multiprocessing as mp
import os
import pickle
from glob import glob
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from config import get_config, prepair_dir, set_seed
from src.factories import get_dataset, get_loss, get_model, get_optimizer, get_scheduler
from src.sync_batchnorm import convert_model


class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.hparams = config
        self.base_config = config.base
        self.data_config = config.data
        self.model_config = config.model
        self.train_config = config.train
        self.test_config = config.test
        self.save_config = config.save
        self.cpu_count = mp.cpu_count() // len(self.base_config.gpu_id)
        self.target_cols = [
            "any",
            "epidural",
            "subdural",
            "subarachnoid",
            "intraventricular",
            "intraparenchymal",
        ]
        # load from factories
        self.model = get_model(
            model_name=config.model.model_name, model_config=config.model
        )
        if len(self.base_config.gpu_id) > 1:
            self.model = convert_model(self.model)
        self.optimizer = get_optimizer(
            opt_name=self.base_config.opt_name,
            params=self.model.parameters(),
            lr=self.train_config.learning_rate,
        )

        self.train_dataset = get_dataset(
            dataset_name=self.data_config.dataset_name,
            data_config=self.data_config,
            mode="train",
        )
        self.valid_dataset = get_dataset(
            dataset_name=self.data_config.dataset_name,
            data_config=self.data_config,
            mode="valid",
        )
        self.test_dataset = get_dataset(
            dataset_name=self.data_config.dataset_name,
            data_config=self.data_config,
            mode="test",
        )
        self.num_train_optimization_steps = int(
            self.train_config.epoch
            * len(self.train_dataset)
            / (self.train_config.batch_size)
            / self.train_config.accumulation_steps
            / len(self.base_config.gpu_id)
        )
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.num_train_optimization_steps, eta_min=1e-5
        )
        self.scheduler = get_scheduler(
            scheduler_name="warmup_scheduler",
            optimizer=self.optimizer,
            multiplier=10,
            total_epoch=int(self.num_train_optimization_steps * 0.1),
            after_scheduler=scheduler_cosine,
        )
        self.loss = get_loss(loss_name=self.base_config.loss_name)
        # path setting
        self.save_path = os.path.join(
            self.save_config.save_path, f"fold{self.data_config.fold}"
        )
        self.gcs_path = os.path.join(
            self.save_config.save_path.split("/")[-1], f"fold{self.data_config.fold}"
        )
        if self.test_config.is_validation:
            self.prefix = "valid"
        else:
            self.prefix = "test"
        self.initialize_variables()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, label = batch
        logit = self.forward(x)
        loss = self.loss(logit.float(), label.float())
        sigmoid = torch.sigmoid(logit)
        pred_class = (sigmoid >= 0.5).int()
        label = (label >= 0.5).int()
        acc = accuracy_score(
            pred_class.detach().cpu().numpy().reshape(-1),
            label.detach().cpu().numpy().reshape(-1),
        )
        metrics = {
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "train_acc": acc,
        }
        metrics = {}
        metrics["loss"] = loss
        metrics["progress_bar"] = {
            "train_acc": acc,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        metrics["log"] = {
            "train_loss": loss.item(),
            "train_acc": acc,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        return metrics

    def validation_step(self, batch, batch_nb):
        x, label = batch
        if self.test_config.is_tta:
            x = torch.cat([x, x.flip(1), x.flip(2)], dim=0)
        logit = self.forward(x)
        if self.test_config.is_tta:
            logit_mean = logit.view((3, -1) + logit.shape[1:]).mean(0)
            loss = self.loss(logit_mean.float(), label.float())
        else:
            loss = self.loss(logit.float(), label.float())
        sigmoid = torch.sigmoid(logit)
        feature = self.model.feature
        if self.test_config.is_tta:
            sigmoid = sigmoid.view((3, -1) + sigmoid.shape[1:]).mean(0)
            feature = feature.view((3, -1) + feature.shape[1:]).mean(0)
        pred_class = (sigmoid >= 0.5).int()
        label = (label >= 0.5).int()
        acc = accuracy_score(
            pred_class.detach().cpu().numpy().reshape(-1),
            label.detach().cpu().numpy().reshape(-1),
        )
        metrics = {
            "val_loss": loss.item(),
            "val_acc": acc,
            "class_preds": sigmoid.detach().cpu().numpy(),
            "feature": feature.detach().cpu().numpy(),
        }

        return metrics

    def test_step(self, batch, batch_nb):
        if self.test_config.is_tta:
            batch = torch.cat([batch, batch.flip(1), batch.flip(2)], dim=0)
        logit = self.forward(batch)
        sigmoid = torch.sigmoid(logit)
        sigmoid = sigmoid.view((3, -1) + sigmoid.shape[1:]).mean(0)
        feature = self.model.feature
        feature = feature.view((3, -1) + feature.shape[1:]).mean(0)
        metrics = {}
        metrics.update(
            {
                "class_preds": sigmoid.detach().cpu().numpy(),
                "feature": feature.detach().cpu().numpy(),
            }
        )

        return metrics

    def validation_end(self, outputs):
        avg_loss = np.nanmean(np.array([x["val_loss"] for x in outputs]))
        avg_acc = np.array([x["val_acc"] for x in outputs]).mean()
        if len(self.base_config.gpu_id) > 1:
            rank = mp.current_process()._identity[0]
        else:
            rank = 1
        res = {}
        res["step"] = int(self.global_step)
        res["epoch"] = int(self.current_epoch)
        if avg_loss <= self.best_loss:
            self.best_loss = avg_loss
            res["best_loss"] = float(self.best_loss)
            class_preds = np.concatenate([x["class_preds"] for x in outputs], axis=0)
            feature = np.concatenate([x["feature"] for x in outputs], axis=0)
            np.save(
                os.path.join(self.save_path, f"result/valid_class_{rank}.npy"),
                class_preds,
            )
            np.save(
                os.path.join(self.save_path, f"result/valid_feature_{rank}.npy"),
                feature,
            )
            with open(os.path.join(self.save_path, "logs/best_score.yaml"), "w") as f:
                yaml.dump(res, f, default_flow_style=False)
        metrics = {}
        metrics["progress_bar"] = {
            "avg_val_loss": avg_loss,
            "avg_val_acc": avg_acc,
            "best_loss": self.best_loss,
        }
        metrics["log"] = {
            "avg_val_loss": avg_loss,
            "avg_val_acc": avg_acc,
            "best_loss": self.best_loss,
        }
        return metrics

    def test_end(self, outputs):
        if len(self.base_config.gpu_id) > 1:
            rank = mp.current_process()._identity[0]
        else:
            rank = 1
        class_preds = np.concatenate([x["class_preds"] for x in outputs], axis=0)
        feature = np.concatenate([x["feature"] for x in outputs], axis=0)
        np.save(
            os.path.join(self.save_path, f"result/{self.prefix}_class_{rank}.npy"),
            class_preds,
        )
        np.save(
            os.path.join(self.save_path, f"result/{self.prefix}_feature_{rank}.npy"),
            feature,
        )
        paths = sorted(
            glob(os.path.join(self.save_path, f"result/{self.prefix}_class_*.npy"))
        )
        if len(paths) == len(self.base_config.gpu_id):
            if self.test_config.is_validation:
                df = self.valid_dataset.df
            else:
                df = self.test_dataset.df
            # aggregate prediction
            preds_class = [np.load(path) for path in paths]
            preds_class = np.vstack(zip(*preds_class))
            if preds_class.shape[0] != len(df):
                preds_class = preds_class[: len(df)]
            if self.model_config["classes"] == 6:
                for i, col in enumerate(self.target_cols):
                    df[f"{col}_pred"] = preds_class[:, i]
            elif self.model_config["classes"] == 5:
                for i, col in enumerate(self.target_cols[1:]):
                    df[f"{col}_pred"] = preds_class[:, i]
                df[f"any_pred"] = 1 - (1 - preds_class[:, 0]) * (
                    1 - preds_class[:, 1]
                ) * (1 - preds_class[:, 2]) * (1 - preds_class[:, 3]) * (
                    1 - preds_class[:, 4]
                )
            with open(
                os.path.join(self.save_path, f"result/{self.prefix}_class_pred.pkl"),
                "wb",
            ) as f:
                pickle.dump(df, f)
            # aggregate cnn feature
            paths = sorted(
                glob(
                    os.path.join(self.save_path, f"result/{self.prefix}_feature_*.npy")
                )
            )
            feature = [np.load(path) for path in paths]
            feature = np.vstack(zip(*feature))
            if feature.shape[0] != len(df):
                feature = feature[: len(df)]
            np.save(
                os.path.join(self.save_path, f"result/{self.prefix}_feature_all.npy"),
                feature,
            )

        return {}

    def on_epoch_end(self):
        if len(self.base_config.gpu_id) > 1:
            rank = mp.current_process()._identity[0]
        else:
            rank = 1
        if rank == 1:
            paths = sorted(
                glob(os.path.join(self.save_path, f"result/valid_class_*.npy"))
            )
            if len(paths) == len(self.base_config.gpu_id):
                # save valid feature
                df = self.valid_dataset.df
                # aggregate prediction
                preds_class = [np.load(path) for path in paths]
                preds_class = np.vstack(zip(*preds_class))
                if preds_class.shape[0] != len(df):
                    preds_class = preds_class[: len(df)]
                if self.model_config["classes"] == 6:
                    for i, col in enumerate(self.target_cols):
                        df[f"{col}_pred"] = preds_class[:, i]
                elif self.model_config["classes"] == 5:
                    for i, col in enumerate(self.target_cols[1:]):
                        df[f"{col}_pred"] = preds_class[:, i]
                    df[f"any_pred"] = 1 - (1 - preds_class[:, 0]) * (
                        1 - preds_class[:, 1]
                    ) * (1 - preds_class[:, 2]) * (1 - preds_class[:, 3]) * (
                        1 - preds_class[:, 4]
                    )
                with open(
                    os.path.join(self.save_path, f"result/valid_class_pred.pkl"), "wb"
                ) as f:
                    pickle.dump(df, f)

                # aggregate cnn feature
                paths = sorted(
                    glob(os.path.join(self.save_path, f"result/valid_feature_*.npy"))
                )
                feature = [np.load(path) for path in paths]
                feature = np.vstack(zip(*feature))
                if feature.shape[0] != len(df):
                    feature = feature[: len(df)]
                np.save(
                    os.path.join(self.save_path, f"result/valid_feature_all.npy"),
                    feature,
                )

    def configure_optimizers(self):
        return [self.optimizer]

    @pl.data_loader
    def train_dataloader(self):
        if len(self.base_config.gpu_id) > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, shuffle=True
            )
        else:
            sampler = torch.utils.data.sampler.RandomSampler(self.train_dataset)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_config.batch_size,
            pin_memory=True,
            num_workers=4,
            sampler=sampler,
            drop_last=True,
        )
        return train_loader

    @pl.data_loader
    def val_dataloader(self):
        if len(self.base_config.gpu_id) > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.valid_dataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(self.valid_dataset)
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.test_config.batch_size,
            num_workers=4,
            pin_memory=True,
            sampler=sampler,
        )

        return valid_loader

    @pl.data_loader
    def test_dataloader(self):
        if not self.data_config.is_train:
            if self.test_config.is_validation:
                dataset = self.valid_dataset
            else:
                dataset = self.test_dataset
            if len(self.base_config.gpu_id) > 1:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset, shuffle=False
                )
            else:
                sampler = torch.utils.data.sampler.SequentialSampler(dataset)
            test_loader = DataLoader(
                dataset,
                batch_size=self.test_config.batch_size,
                num_workers=self.cpu_count,
                pin_memory=True,
                sampler=sampler,
            )
            return test_loader
        else:
            pass

    def optimizer_step(
        self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None
    ):
        optimizer.step()
        optimizer.zero_grad()
        self.scheduler.step()

    def initialize_variables(self):
        self.step = 0
        self.best_loss = 100
        if self.train_config.warm_start:
            with open(os.path.join(self.save_path, "logs/best_score.yaml"), "r") as f:
                res = yaml.safe_load(f)
            # self.best_dice = 0
            if "best_loss" in res.keys():
                self.best_loss = res["best_loss"]
            self.step = res["step"]


def main():
    # Setup
    config = get_config()
    prepair_dir(config)
    set_seed(config.train.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(config.base.gpu_id)

    # Preparing for trainer
    base_path = os.path.join(config.save.save_path, f"fold{config.data.fold}")
    model_path = os.path.join(base_path, "model")
    model_name = config.base.yaml
    monitor_metric = "avg_val_loss"
    checkpoint_callback = ModelCheckpoint(
        filepath=model_path,
        save_best_only=True,
        verbose=True,
        monitor=monitor_metric,
        mode="min",
        prefix=model_name,
    )
    logger = TestTubeLogger(
        save_dir=os.path.join(base_path, "logs"),
        name=model_name,
        debug=False,
        create_git_tag=False,
    )
    backend = "ddp" if len(config.base.gpu_id) > 1 else None

    model = Model(config)
    if config.data.is_train:
        trainer = Trainer(
            logger=logger,
            early_stop_callback=False,
            max_nb_epochs=config.train.epoch,
            checkpoint_callback=checkpoint_callback,
            accumulate_grad_batches=config.train.accumulation_steps,
            use_amp=True,
            amp_level="O1",
            gpus=[int(id_) for id_ in config.base.gpu_id],
            distributed_backend=backend,
            show_progress_bar=True,
            train_percent_check=1.0,
            check_val_every_n_epoch=1,
            val_check_interval=1.0,
            val_percent_check=1.0,
            test_percent_check=0.0,
            nb_sanity_val_steps=0,
            nb_gpu_nodes=1,
            print_nan_grads=False,
            track_grad_norm=-1,
            gradient_clip_val=1,
            row_log_interval=1000,
            log_save_interval=10,
        )

        trainer.fit(model)
    else:
        trainer = Trainer(
            logger=False,
            early_stop_callback=False,
            max_nb_epochs=0,
            checkpoint_callback=checkpoint_callback,
            use_amp=True,
            amp_level="O1",
            gpus=[int(id_) for id_ in config.base.gpu_id],
            distributed_backend=backend,
            show_progress_bar=True,
            train_percent_check=0,
            check_val_every_n_epoch=0,
            val_check_interval=0.0,
            val_percent_check=0.0,
            test_percent_check=1.0,
            nb_sanity_val_steps=0,
            nb_gpu_nodes=1,
            print_nan_grads=False,
            track_grad_norm=-1,
        )
        trainer.test(model)


if __name__ == "__main__":
    main()
