import argparse
import json
import os
import re
import random
import shutil

import numpy as np
import torch
import yaml


class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith("__") and k.endswith("__")) and k not in (
                "update",
                "pop",
            ):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x) if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)


def set_seed(seed):
    os.environ.PYTHONHASHSEED = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def prepair_dir(config):
    """
    Logの保存先を作成
    """
    log_path = os.path.join(config.save.save_path, f"fold{config.data.fold}")
    if (
        os.path.exists(log_path)
        and config.train.warm_start is False
        and config.data.is_train
    ):
        shutil.rmtree(log_path)
    os.makedirs(log_path, exist_ok=True)
    # For save result
    os.makedirs(os.path.join(log_path, "result"), exist_ok=True)
    # For tensorboard logging
    os.makedirs(os.path.join(log_path, "logs"), exist_ok=True)
    # For save model
    os.makedirs(os.path.join(log_path, "model"), exist_ok=True)
    # Save parameter
    with open(os.path.join(log_path, "parameter.json"), "w") as f:
        json.dump(config, f, indent=4)
    with open(os.path.join(log_path, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def check_config(config):
    if not config.data.is_train:
        assert config.train.warm_start is True
    if config.model.classes == 1:
        assert config.data.dataset_name == "any_dataset"


def get_config():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-y", "--yaml", type=str, default="./yamls/config.yaml", help="Path to yaml"
    )
    parser.add_argument("-f", "--fold", type=str, help="Select fold")
    parser.add_argument("--loss_name", type=str, help="Loss name")
    parser.add_argument("--epoch", type=int, help="Epoch")
    parser.add_argument("--warm_start", action="store_true", help="Warm start flag")
    parser.add_argument("-t", "--train", action="store_true", help="Train flag")
    parser.add_argument("-v", "--validate", action="store_true", help="Train flag")
    parser.add_argument("--test", action="store_true", help="Train flag")

    args = parser.parse_args()
    with open(args.yaml, "r") as f:
        config = EasyDict(yaml.safe_load(f))

    if args.fold == "all":
        config.data.fold = args.fold
    else:
        config.data.fold = int(args.fold)
    if args.loss_name:
        config.base.loss_name = args.loss_name
    if args.warm_start:
        config.train.warm_start = True
    if args.epoch:
        config.train.epoch = args.epoch
    if args.train:
        config.data.is_train = True
    if args.validate:
        config.data.is_train = False
        config.train.warm_start = True
        config.test.is_validation = True
    if args.test:
        config.data.is_train = False
        config.train.warm_start = True
        config.test.is_validation = False
    config.base.yaml = re.split("[./]", args.yaml)[-2]

    check_config(config)

    return config
