import os
import yaml
import random
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import log_loss
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler


class StackingCNNModel(nn.Module):
    def __init__(self, feature_dim):
        super(StackingCNNModel, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(
                in_channels=3,
                out_channels=4,
                kernel_size=(3, 1),
                padding=(0, 0),
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=(3, 1),
                padding=(0, 0),
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 1),
                padding=(0, 0),
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.dense_layer = nn.Sequential(
            nn.BatchNorm1d(16 * feature_dim),
            nn.Dropout(0.3),
            nn.Linear(16 * feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 6),
        )

    def forward(self, x):
        out = self.cnn_layer(x)
        out = out.view(out.shape[0], -1)
        out = self.dense_layer(out)

        return out


class StackingCNNDataset(Dataset):
    def __init__(self, X, y=None, study_ids=None, mode="train"):
        self.X = X.squeeze()
        if y is not None:
            self.y = y.squeeze()
        self.study_ids = study_ids
        self.mode = mode

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if idx == 0:
            idxs = [idx, idx + 1, idx + 2]
        elif idx == (self.X.shape[0] - 1):
            idxs = [idx - 2, idx - 1, idx]
        else:
            study_ids = self.study_ids[[idx - 1, idx, idx + 1]]
            if len(np.unique(study_ids)) != 1:
                if study_ids[0] == study_ids[1]:
                    idxs = [idx - 2, idx - 1, idx]
                elif study_ids[1] == study_ids[2]:
                    idxs = [idx, idx + 1, idx + 2]
            else:
                idxs = [idx - 1, idx, idx + 1]
        if self.mode == "train":
            return torch.FloatTensor(self.X[idxs]), torch.FloatTensor(self.y[idx])
        else:

            return torch.FloatTensor(self.X[idxs])


def set_seed(seed):
    os.environ.PYTHONHASHSEED = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def make_prediction_feature(df, target_cols):
    groupby = df.groupby("StudyInstanceUID")[
        [col for col in df.columns if col.endswith("_pred")]
    ]
    a1 = groupby.rolling(3, min_periods=1, center=True).mean().values
    a2 = groupby.rolling(5, min_periods=1, center=True).mean().values
    a3 = groupby.rolling(1, min_periods=1, center=True).mean().values
    new_feats1 = pd.DataFrame(a1, columns=[c + "_3roll" for c in target_cols])
    new_feats2 = pd.DataFrame(a2, columns=[c + "_5roll" for c in target_cols])
    new_feats3 = pd.DataFrame(a1 - a3, columns=[c + "_3rolldiff" for c in target_cols])
    new_feats4 = pd.DataFrame(a2 - a3, columns=[c + "_5rolldiff" for c in target_cols])
    new_feats1.index = df.index
    new_feats2.index = df.index
    new_feats3.index = df.index
    new_feats4.index = df.index
    df = pd.concat([df, new_feats1, new_feats2, new_feats3, new_feats4], axis=1)
    return df


def make_feature(ref_df, model_name, mode="train"):
    target_cols = [
        "any",
        "epidural",
        "subdural",
        "subarachnoid",
        "intraventricular",
        "intraparenchymal",
    ]
    if model_name in [
        "densenet_adj_prediction",
        "seresnext50_label_smoothing",
        "seresnext50_label_smoothing_without_any",
    ]:
        dfs = []
        for n_fold in range(5):
            if mode == "train":
                dfs.append(
                    pd.read_pickle(
                        f"./intermediate_output/{model_name}/fold{n_fold}/result/valid_class_pred.pkl"
                    )
                )
            else:
                df = pd.read_pickle(
                    f"./intermediate_output/{model_name}/fold{n_fold}/result/test_class_pred.pkl"
                )
                if "StudyInstanceUID" not in df.columns:
                    df = df.merge(ref_df[["ID", "StudyInstanceUID"]], on="ID")
                dfs.append(df)
        df_all = pd.concat(dfs)
        df_all = df_all.sort_values(["StudyInstanceUID", "PositionOrd"])
        df_all = make_prediction_feature(df_all, target_cols)
    else:
        dfs = []
        n_tta = 5
        for n_fold in range(5):
            if mode == "train":
                df = pd.read_pickle(
                    f"./intermediate_output/{model_name}/fold{n_fold}_valid.pkl"
                )
            else:
                df = pd.read_pickle(
                    f"./intermediate_output/{model_name}/fold{n_fold}_test.pkl"
                )
            tmp = np.zeros([len(df[0]["ids"]), 6])
            for i in range(n_tta):
                tmp += df[i]["outputs"] / n_tta
            tmp = pd.DataFrame(tmp)
            tmp.columns = [tar_col + "_pred" for tar_col in target_cols]
            tmp["ID"] = df[0]["ids"]
            tmp["fold"] = n_fold
            tmp2 = pd.DataFrame(df[0]["targets"], columns=target_cols)
            tmp = pd.concat([tmp, tmp2], axis=1)
            dfs.append(tmp)
        df_all = pd.concat(dfs)
        df_all = df_all.merge(
            ref_df[["ID", "StudyInstanceUID", "PositionOrd"]], on="ID"
        )
        df_all = df_all.sort_values(["StudyInstanceUID", "PositionOrd"])
        df_all = make_prediction_feature(df_all, target_cols)

    return df_all


def fill_infinity(model_df):
    index = np.where(model_df.values == np.inf)
    if len(index[0]) > 0:
        model_df.iloc[index] = model_df.iloc[index[1]].max(1)
    return model_df


def make_dataset(fold, model_dfs, feature_cols, target_cols, label):
    X_train = []
    X_valid = []
    std = StandardScaler()
    for i, model_df in enumerate(model_dfs):
        X_train.append(
            std.fit_transform(model_df.query(f"fold!={fold}")[feature_cols].values)
        )
        X_valid.append(
            std.fit_transform(model_df.query(f"fold=={fold}")[feature_cols].values)
        )
    y_train = label.query(f"fold!={fold}")[target_cols].values
    y_valid = label.query(f"fold=={fold}")[target_cols].values
    X_train = np.array(X_train).transpose(1, 0, 2)[:, np.newaxis, ...]
    X_valid = np.array(X_valid).transpose(1, 0, 2)[:, np.newaxis, ...]
    return X_train, X_valid, y_train, y_valid


def make_test_dataset(model_dfs, feature_cols):
    X_test = []
    std = StandardScaler()
    for i, model_df in enumerate(model_dfs):
        X_test.append(std.fit_transform(model_df[feature_cols].values))
    X_test = np.array(X_test).transpose(1, 0, 2)[:, np.newaxis, ...]
    return X_test


def load_data():
    train_df = pd.read_csv(
        "./intermediate_output/preprocessed_data/train_fold_with_label_smooth.csv"
    )
    train_df = train_df.sort_values(by=["StudyInstanceUID", "PositionOrd"]).reset_index(
        drop=True
    )

    test_df = np.load(
        "./intermediate_output/preprocessed_data/test.pkl", allow_pickle=True
    )
    test_df = test_df.merge(
        np.load(
            "./intermediate_output/preprocessed_data/test_raw.pkl", allow_pickle=True
        )[["ID", "StudyInstanceUID"]],
        on="ID",
        how="left",
    )
    test_df = test_df.sort_values(by=["StudyInstanceUID", "PositionOrd"]).reset_index(
        drop=True
    )
    return train_df, test_df


def main():
    feature_cols = [
        "any_pred",
        "epidural_pred",
        "subdural_pred",
        "subarachnoid_pred",
        "intraventricular_pred",
        "intraparenchymal_pred",
        "PositionOrd",
        "any_3roll",
        "epidural_3roll",
        "subdural_3roll",
        "subarachnoid_3roll",
        "intraventricular_3roll",
        "intraparenchymal_3roll",
        "any_5roll",
        "epidural_5roll",
        "subdural_5roll",
        "subarachnoid_5roll",
        "intraventricular_5roll",
        "intraparenchymal_5roll",
        "any_3rolldiff",
        "epidural_3rolldiff",
        "subdural_3rolldiff",
        "subarachnoid_3rolldiff",
        "intraventricular_3rolldiff",
        "intraparenchymal_3rolldiff",
        "any_5rolldiff",
        "epidural_5rolldiff",
        "subdural_5rolldiff",
        "subarachnoid_5rolldiff",
        "intraventricular_5rolldiff",
        "intraparenchymal_5rolldiff",
    ]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_cols = [
        "any",
        "epidural",
        "subdural",
        "subarachnoid",
        "intraventricular",
        "intraparenchymal",
    ]
    train_df, test_df = load_data()
    label = train_df[target_cols + ["fold"]]
    eps = 1e-4
    with open("./cnn_stacking_2.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_names = config["model_names"]
    # Make Feature
    model_dfs_train = [
        make_feature(train_df, model_name, mode="train")
        for model_name in tqdm(model_names, desc="[Train]Feature Extraction")
    ]
    model_dfs_test = [
        make_feature(test_df, model_name, mode="test")
        for model_name in tqdm(model_names, desc="[Test]Feature Extraction")
    ]
    # Fill infinity
    model_dfs_train = [fill_infinity(model_df) for model_df in model_dfs_train]
    model_dfs_test = [fill_infinity(model_df) for model_df in model_dfs_test]

    stacking_name = "cnn_stacking_2"
    os.makedirs(f"./intermediate_output/{stacking_name}", exist_ok=True)

    criterion = nn.BCEWithLogitsLoss(
        weight=torch.FloatTensor([2, 1, 1, 1, 1, 1]).to(device)
    )
    scores = []
    for fold in range(5):
        set_seed(777)
        # Prepare for training
        X_train, X_valid, y_train, y_valid = make_dataset(
            fold, model_dfs_train, feature_cols, target_cols, label
        )
        valid_df = model_dfs_train[0].query(f"fold=={fold}").reset_index(drop=True)
        feature_dim = X_train.shape[-1]
        study_ids_train = (
            model_dfs_train[0]
            .query(f"fold!={fold}")
            .sort_values(by=["StudyInstanceUID", "PositionOrd"])
            .reset_index(drop=True)["StudyInstanceUID"]
            .values
        )
        study_ids_valid = (
            model_dfs_train[0]
            .query(f"fold=={fold}")
            .sort_values(by=["StudyInstanceUID", "PositionOrd"])
            .reset_index(drop=True)["StudyInstanceUID"]
            .values
        )
        train_loader = DataLoader(
            StackingCNNDataset(X_train, y_train, study_ids_train, mode="train"),
            shuffle=True,
            batch_size=config["batch_size"],
            num_workers=4,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            StackingCNNDataset(X_valid, y_valid, study_ids_valid, mode="train"),
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        model = StackingCNNModel(feature_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), config["learning_rate"])
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config["epoch"], eta_min=1e-4
        )

        best_loss = np.inf
        preds = []
        for epoch in range(config["epoch"]):
            losses = []
            preds = []
            targets = []
            for x, y in train_loader:
                pred = model(x.to(device))
                loss = criterion(pred, y.to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())
            losses_valid = []
            with torch.no_grad():
                model.eval()
                for x, y in valid_loader:
                    pred = model(x.to(device))
                    loss = criterion(pred, y.to(device))
                    losses_valid.append(loss.item())
                    preds.append(torch.sigmoid(pred).detach().cpu().numpy())
                    targets.append(y.detach().cpu().numpy())
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            metrics = []
            preds_clip = np.zeros_like(preds)
            for i in range(6):
                preds_clip[:, i] = np.clip(preds[:, i], eps, 1 - eps)
            for i in range(6):
                metrics.append(log_loss(np.floor(targets[:, i]), preds_clip[:, i]))
            metrics = np.average(metrics, weights=[2, 1, 1, 1, 1, 1])
            if metrics <= best_loss:
                best_loss = metrics
                for i in range(6):
                    valid_df.loc[:, f"{target_cols[i]}_pred"] = preds[:, i]

                torch.save(
                    model.state_dict(),
                    f"./intermediate_output/{stacking_name}/stacking_model_{fold}.pth",
                )
                with open(
                    f"./intermediate_output/{stacking_name}/stacking_valid_pred_fold{fold}.pkl",
                    "wb",
                ) as f:
                    pickle.dump(valid_df, f)
                print(
                    f"Epoch[{epoch}]/fold{fold}",
                    "Train:",
                    np.mean(losses),
                    "Valid:",
                    np.mean(losses_valid),
                    "Score:",
                    metrics,
                )

            model.train()
            scheduler_cosine.step()
        scores.append(best_loss)
    print("5-fold average score: ", np.mean(scores))

    X_test = make_test_dataset(model_dfs_test, feature_cols)
    # Split per fold
    X_tests = np.split(X_test, 5)
    for fold in range(5):
        study_ids_test = test_df["StudyInstanceUID"].values
        test_loader = DataLoader(
            StackingCNNDataset(
                X_tests[fold], y=None, study_ids=study_ids_test, mode="test"
            ),
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        checkpoint = torch.load(
            f"./intermediate_output/{stacking_name}/stacking_model_{fold}.pth"
        )
        model.load_state_dict(checkpoint)
        model.eval()
        preds = []
        for x in test_loader:
            pred = model(x.to(device))
            preds.append(torch.sigmoid(pred).detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        for i in range(6):
            test_df.loc[:, f"{target_cols[i]}_pred"] = preds[:, i]
        with open(
            f"./intermediate_output/{stacking_name}/stacking_test_pred_fold{fold}.pkl",
            "wb",
        ) as f:
            pickle.dump(test_df, f)


if __name__ == "__main__":
    main()
