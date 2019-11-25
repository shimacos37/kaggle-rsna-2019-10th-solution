"""
written by ricky
"""
import os
import pickle
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss


def make_feats(df):
    """
    make position feature
    """
    df["ImagePositionPatient_2"] = df["ImagePositionPatient"].apply(lambda x: x[2])
    df = df.merge(
        df.groupby(["StudyInstanceUID"])["ImagePositionPatient_2"]
        .agg(position_min="min", position_max="max")
        .reset_index(),
        on="StudyInstanceUID",
    )
    df["position"] = (df["ImagePositionPatient_2"] - df["position_min"]) / (
        df["position_max"] - df["position_min"]
    )
    res = df.sort_values(by=["StudyInstanceUID", "position"])
    return res


def pred_agg1_train(df):
    """
    1st level aggregation for train data
    """
    new_feats = []
    pred_cols = [column for column in df.columns if "pred" in column]
    for c in pred_cols:
        tmp = (
            df.groupby(["StudyInstanceUID"])[c]
            .agg(["min", "max", "mean", "std"])
            .reset_index()
        )
        tmp.columns = [
            "StudyInstanceUID",
            c + "_min",
            c + "_max",
            c + "_mean",
            c + "_std",
        ]
        if c != "any_pred_model_base":
            del tmp["StudyInstanceUID"]
        new_feats.append(tmp)
    new_feats = pd.concat(new_feats, axis=1)
    df = pd.merge(df, new_feats, on="StudyInstanceUID", how="left")
    for c in pred_cols:
        df[c + "_diff"] = df[c] - df[c + "_mean"]
        df[c + "_div"] = df[c] / df[c + "_mean"]
        df[c + "_scaled"] = (df[c] - df[c + "_mean"]) / df[c + "_std"]
    return df


def pred_agg1_test(df):
    """
    1st level aggregation for test data
    """
    new_feats = []
    pred_cols = [column for column in df.columns if "pred" in column]
    for c in pred_cols:
        tmp = (
            df.groupby(["StudyInstanceUID"])[c]
            .agg(["min", "max", "mean", "std"])
            .reset_index()
        )
        tmp.columns = [
            "StudyInstanceUID",
            c + "_min",
            c + "_max",
            c + "_mean",
            c + "_std",
        ]
        if "any_pred" not in c:
            del tmp["StudyInstanceUID"]
        new_feats.append(tmp)
    new_feats = pd.concat(new_feats, axis=1)
    df = pd.merge(df, new_feats, on="StudyInstanceUID", how="left")
    for c in pred_cols:
        df[c + "_diff"] = df[c] - df[c + "_mean"]
        df[c + "_div"] = df[c] / df[c + "_mean"]
        df[c + "_scaled"] = (df[c] - df[c + "_mean"]) / df[c + "_std"]
    return df


def pred_agg2(df):
    """
    2nd level aggregation
    """
    pred_cols = [column for column in df.columns if "pred" in column]
    a1 = (
        df.groupby("StudyInstanceUID")[pred_cols]
        .rolling(3, min_periods=1, center=True)
        .mean()
        .values
    )
    a2 = (
        df.groupby("StudyInstanceUID")[pred_cols]
        .rolling(5, min_periods=1, center=True)
        .mean()
        .values
    )
    a3 = (
        df.groupby("StudyInstanceUID")[pred_cols]
        .rolling(1, min_periods=1, center=True)
        .mean()
        .values
    )
    new_feats1 = pd.DataFrame(a1, columns=[c + "_3roll" for c in pred_cols])
    new_feats2 = pd.DataFrame(a2, columns=[c + "_5roll" for c in pred_cols])
    new_feats3 = pd.DataFrame(a1 - a3, columns=[c + "_3rolldiff" for c in pred_cols])
    new_feats4 = pd.DataFrame(a2 - a3, columns=[c + "_5rolldiff" for c in pred_cols])
    new_feats5 = pd.DataFrame(a1 / a3, columns=[c + "_3rolldiv" for c in pred_cols])
    new_feats6 = pd.DataFrame(a2 / a3, columns=[c + "_5rolldiv" for c in pred_cols])
    new_feats1.index = df.index
    new_feats2.index = df.index
    new_feats3.index = df.index
    new_feats4.index = df.index
    new_feats5.index = df.index
    new_feats6.index = df.index
    df = pd.concat(
        [df, new_feats1, new_feats2, new_feats3, new_feats4, new_feats5, new_feats6],
        axis=1,
    )
    return df


def make_dataset(
    path_to_train_raw="intermediate_output/preprocessed_data/train_raw.pkl",
    path_to_test_raw="intermediate_output/preprocessed_data/test_raw.pkl",
    save_data=True,
):
    # load dataset
    train = pd.read_pickle(path_to_train_raw)
    train = make_feats(train)
    test = pd.read_pickle(path_to_test_raw)
    test = make_feats(test)

    # define column names
    target_cols = [
        "any",
        "epidural",
        "subdural",
        "subarachnoid",
        "intraventricular",
        "intraparenchymal",
    ]
    # define model names
    models = [
        "model_base",
        "se_resnext_410",
        "se_resnext101_mixup",
        "senet154_customlabels",
        "2kyym_inception_resnet_v2",
        "2kyym_inceptionv4",
        "2kyym_xception",
        "sugawara_efficientnetb3",
    ]

    # train data
    # load and unite cnn predictions (train data)
    model_dfs = []
    for model in tqdm(models):
        df_all = []
        n_tta = 5
        for n_fold in range(5):
            df = pd.read_pickle(f"intermediate_output/{model}/fold{n_fold}_valid.pkl")
            tmp = np.zeros([len(df[0]["ids"]), 6])
            for i in range(n_tta):
                tmp += df[i]["outputs"] / n_tta
            tmp = pd.DataFrame(tmp)
            tmp.columns = [tar_col + "_pred" for tar_col in target_cols]
            tmp["ID"] = df[0]["ids"]
            target = df[0]["targets"]
            tmp["folds"] = n_fold
            if model == "senet154_customlabels":
                target[target < 0.5] = 0
            tmp2 = pd.DataFrame(target)
            tmp2.columns = [tar_col + "_true" for tar_col in target_cols]
            df_all.append(pd.concat([tmp, tmp2], axis=1))
        df_all = pd.concat(df_all)
        model_dfs.append(df_all)

    columns = list(model_dfs[0].columns)
    columns.remove("ID")
    for i, model_df in enumerate(model_dfs):
        rename_dict = {}
        for column in columns:
            rename_dict[column] = column + "_{}".format(i)
        if i == 0:
            df = model_df.rename(columns=rename_dict)
        else:
            df = pd.merge(
                df, model_df.rename(columns=rename_dict), on="ID", how="outer"
            )

    pred_columns = [c for c in df.columns if "pred" in c]
    df[pred_columns] = df[pred_columns].clip(1e-15, 1 - 1e-15)

    for i, row in df[df.isnull().any(axis=1)].iterrows():
        for column in columns:
            tmp = [c for c in df.columns if column in c]
            nan_columns = []
            for j, t in enumerate(row[tmp].isnull()):
                if t:
                    nan_columns.append(tmp[j])
            tmp = row[tmp].mean()
            df.loc[i, nan_columns] = tmp

    n_model = len(model_dfs)
    model_dfs = []
    for i in range(n_model):
        tmp_columns = [c for c in df.columns if "_{}".format(i) in c]
        df_all = df[tmp_columns + ["ID"]]
        rename_dict = {}
        for tmp_column in tmp_columns:
            rename_dict[tmp_column] = tmp_column.replace("_{}".format(i), "").replace(
                "_true", ""
            )
        df_all = df_all.rename(columns=rename_dict)
        model_dfs.append(df_all)

    for i, model_df in enumerate(model_dfs):
        rename_dict = {}
        for target_col in target_cols:
            rename_dict[target_col + "_pred"] = target_col + "_pred_" + models[i]
        if i == 0:
            df_all = model_df.rename(columns=rename_dict)
        else:
            c = model_df.columns.drop(target_cols + ["folds"])
            df_all = pd.merge(
                df_all, model_df[c].rename(columns=rename_dict), on="ID", how="left"
            )

    df_all = df_all.merge(train, on="ID")
    if "position" in df_all.columns:
        df_all = df_all.sort_values(["StudyInstanceUID", "position"])
    else:
        df_all = df_all.sort_values(["StudyInstanceUID"])

    # train data aggregation
    print("train data aggredating")
    df_all = pred_agg1_train(df_all)
    df_all = pred_agg2(df_all)

    # test data
    # load and unite cnn predictions (test data)
    model_df_list_test = []
    for model in tqdm(models):
        df_all_test = []
        for n_fold in range(5):
            df = pd.read_pickle(f"intermediate_output/{model}/fold{n_fold}_test.pkl")
            tmp = np.zeros([len(df[0]["ids"]), 6])
            for i in range(n_tta):
                tmp += df[i]["outputs"] / n_tta
            tmp = pd.DataFrame(tmp)
            tmp.columns = [tar_col + "_pred" for tar_col in target_cols]
            tmp["ID"] = df[n_fold]["ids"]
            tmp["folds"] = n_fold
            pred_columns = [c for c in tmp.columns if "pred" in c]
            tmp[pred_columns] = tmp[pred_columns].clip(1e-15, 1 - 1e-15)
            df_all_test.append(tmp)
        model_df_list_test.append(df_all_test)

    columns_test = list(model_df_list_test[0][0].columns)
    columns_test.remove("ID")
    averaged_df_list_test = []
    for j in range(len(model_df_list_test[0])):
        for i in range(len(model_df_list_test)):
            rename_dict = {}
            for column in columns_test:
                rename_dict[column] = column + "_{}".format(i)
            if i == 0:
                tmp_df = model_df_list_test[i][j].rename(columns=rename_dict)
            else:
                tmp_df = pd.merge(
                    tmp_df,
                    model_df_list_test[i][j].rename(columns=rename_dict),
                    on="ID",
                    how="outer",
                )
        averaged_df_list_test.append(tmp_df)

    for j, averaged_df in enumerate(averaged_df_list_test):
        for i, row in averaged_df[averaged_df.isnull().any(axis=1)].iterrows():
            for column in columns_test:
                tmp = [c for c in averaged_df.columns if column in c]
                nan_columns = []
                for k, t in enumerate(row[tmp].isnull()):
                    if t:
                        nan_columns.append(tmp[k])
                tmp = row[tmp].mean()
                averaged_df_list_test[j].loc[i, nan_columns] = tmp

    n_model = len(model_df_list_test)
    model_df_list_test = [[] for i in range(n_model)]
    for j, averaged_df in enumerate(averaged_df_list_test):
        for i in range(n_model):
            tmp_columns = [c for c in averaged_df.columns if "_{}".format(i) in c]
            df_all_test = averaged_df[tmp_columns + ["ID"]]
            rename_dict = {}
            for tmp_column in tmp_columns:
                rename_dict[tmp_column] = tmp_column.replace(
                    "_{}".format(i), ""
                ).replace("_true", "")
            df_all_test = df_all_test.rename(columns=rename_dict)
            model_df_list_test[i].append(df_all_test)

    # test data aggregation
    print("test data aggregating")
    for j, model_dfs_test in enumerate(model_df_list_test):
        for i, model_df in enumerate(model_dfs_test):
            rename_dict = {}
            for target_col in target_cols:
                rename_dict[target_col + "_pred"] = target_col + "_pred_" + models[j]
            model_df = model_df.rename(columns=rename_dict)
            model_df = model_df.merge(test, on="ID")
            if "position" in model_df.columns:
                model_df = model_df.sort_values(["StudyInstanceUID", "position"])
            else:
                model_df = model_df.sort_values(["StudyInstanceUID"])
            model_df = pred_agg1_test(model_df)
            model_df = pred_agg2(model_df)
            model_df_list_test[j][i] = model_df

    c_all = []
    for model in models:
        c_all.extend([c for c in df_all.columns if model in c])
    c_all = set(c_all)

    X_cols = df_all.columns.drop(
        [c for c in df_all.columns if "pred" not in c] + target_cols
    )

    df_all = df_all[
        list(X_cols) + target_cols + ["ID", "folds", "StudyInstanceUID", "position"]
    ]
    for i in range(len(model_df_list_test)):
        for j in range(len(model_df_list_test[0])):
            cols = model_df_list_test[i][j].columns
            cols = [
                col
                for col in cols
                if col in list(X_cols) + ["ID", "folds", "StudyInstanceUID", "position"]
            ]
            model_df_list_test[i][j] = model_df_list_test[i][j][cols]

    n_models = len(model_df_list_test)
    n_folds = len(model_df_list_test[0])
    test_cols = set([])
    for i in range(n_models):
        test_cols = test_cols | set(model_df_list_test[i][0].columns)

    test_list = []
    for j in range(n_folds):
        tmp_test_cols = test_cols
        for i in range(n_models):
            cols = model_df_list_test[i][j].columns
            use_cols = [col for col in cols if col in tmp_test_cols]
            if i == 0:
                df = model_df_list_test[i][j][use_cols]
            else:
                df = pd.merge(
                    df,
                    model_df_list_test[i][j][use_cols + ["ID"]],
                    on="ID",
                    how="outer",
                )
            tmp_test_cols = tmp_test_cols - set(use_cols)
        test_list.append(df)

    if save_data:
        # save aggregated data
        os.makedirs("intermediate_output/cnn_stacking_1", exist_ok=True)
        with open(
            "intermediate_output/cnn_stacking_1/df_all_preprcessed_8models.pkl", "wb"
        ) as f:
            pickle.dump(df_all, f, protocol=4)
        with open(
            "intermediate_output/cnn_stacking_1/test_list_preprocessed_8models.pkl",
            "wb",
        ) as f:
            pickle.dump(test_list, f, protocol=4)

    return df_all, test_list


def load_data(
    path_to_train_raw="intermediate_output/cnn_stacking_1/df_all_preprcessed_8models.pkl",
    path_to_test_raw="intermediate_output/cnn_stacking_1/test_raw.pkl",
):
    """
    load aggregated dataset
    """
    with open(path_to_train_raw, "rb") as f:
        df_all = pickle.load(f)
    with open(path_to_test_raw, "rb") as f:
        test_list = pickle.load(f)
    return df_all, test_list


class StackingContinuousDataset(Dataset):
    def __init__(self, X, batch_num_list, ids_list, y):
        self.X = X
        self.y = y
        self.ids_list = ids_list
        self.batch_num_list = batch_num_list

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if idx == 0:
            return (
                torch.FloatTensor(self.X[[idx, idx, idx + 1]]),
                self.ids_list[idx],
                torch.FloatTensor(self.y[idx]),
            )
        elif idx == len(self.batch_num_list) - 1:
            return (
                torch.FloatTensor(self.X[[idx - 1, idx, idx]]),
                self.ids_list[idx],
                torch.FloatTensor(self.y[idx]),
            )
        elif (
            self.batch_num_list[idx] != self.batch_num_list[idx - 1]
            and self.batch_num_list[idx] != self.batch_num_list[idx + 1]
        ):
            return (
                torch.FloatTensor(self.X[[idx, idx, idx]]),
                self.ids_list[idx],
                torch.FloatTensor(self.y[idx]),
            )
        elif self.batch_num_list[idx] != self.batch_num_list[idx - 1]:
            return (
                torch.FloatTensor(self.X[[idx, idx, idx + 1]]),
                self.ids_list[idx],
                torch.FloatTensor(self.y[idx]),
            )
        elif self.batch_num_list[idx] != self.batch_num_list[idx + 1]:
            return (
                torch.FloatTensor(self.X[[idx - 1, idx, idx]]),
                self.ids_list[idx],
                torch.FloatTensor(self.y[idx]),
            )
        else:
            return (
                torch.FloatTensor(self.X[[idx - 1, idx, idx + 1]]),
                self.ids_list[idx],
                torch.FloatTensor(self.y[idx]),
            )


class StackingContinuousDatasetTest(Dataset):
    def __init__(self, X, batch_num_list, ids_list):
        self.X = X
        self.ids_list = ids_list
        self.batch_num_list = batch_num_list

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if idx == 0:
            return torch.FloatTensor(self.X[[idx, idx, idx + 1]]), self.ids_list[idx]
        elif idx == len(self.batch_num_list) - 1:
            return torch.FloatTensor(self.X[[idx - 1, idx, idx]]), self.ids_list[idx]
        elif (
            self.batch_num_list[idx] != self.batch_num_list[idx - 1]
            and self.batch_num_list[idx] != self.batch_num_list[idx + 1]
        ):
            return torch.FloatTensor(self.X[[idx, idx, idx]]), self.ids_list[idx]
        elif self.batch_num_list[idx] != self.batch_num_list[idx - 1]:
            return torch.FloatTensor(self.X[[idx, idx, idx + 1]]), self.ids_list[idx]
        elif self.batch_num_list[idx] != self.batch_num_list[idx + 1]:
            return torch.FloatTensor(self.X[[idx - 1, idx, idx]]), self.ids_list[idx]
        else:
            return (
                torch.FloatTensor(self.X[[idx - 1, idx, idx + 1]]),
                self.ids_list[idx],
            )


class StackingModel(nn.Module):
    def __init__(self):
        super(StackingModel, self).__init__()
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=8,
                kernel_size=(4, 1),
                padding=(0, 0),
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 1),
                padding=(0, 0),
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 1),
                padding=(0, 0),
                bias=False,
            ),
            nn.ReLU(inplace=True),
        )
        self.dense_concat_layer = nn.Sequential(
            nn.Linear(7488, 1024), nn.ReLU(inplace=True), nn.Linear(1024, 6)
        )

    def forward(self, x):
        out = self.cnn_layer(x)
        out = out.view(x.shape[0], -1)
        out = self.dense_concat_layer(out)
        return out


def cnn_stacking(df_all, test_list):
    # define column names
    target_cols = [
        "any",
        "epidural",
        "subdural",
        "subarachnoid",
        "intraventricular",
        "intraparenchymal",
    ]
    # define model names
    models = [
        "model_base",
        "se_resnext_410",
        "se_resnext101_mixup",
        "senet154_customlabels",
        "2kyym_inception_resnet_v2",
        "2kyym_inceptionv4",
        "2kyym_xception",
        "sugawara_efficientnetb3",
    ]

    columns_all = list(
        df_all.columns.drop(["StudyInstanceUID", "ID", "folds", "position"])
    )
    columns_models = []
    for model in models:
        columns_model = [c for c in df_all.columns if model in c]
        columns_model = [c for c in columns_model if c[-3:] != "div"]
        columns_models.append(columns_model)
        columns_all = [c for c in columns_all if c not in columns_model]

    df_all = df_all.sort_values(["folds", "StudyInstanceUID", "position"]).reset_index(
        drop=True
    )
    for i in range(len(test_list)):
        test_list[i] = (
            test_list[i]
            .sort_values(["StudyInstanceUID", "position"])
            .reset_index(drop=True)
        )

    X_train_list = []
    ids_train_list = []
    X_valid_list = []
    ids_valid_list = []
    for i in range(len(test_list)):
        df_train = df_all[df_all.folds != i]
        tmp_train = np.empty((len(models), len(df_train), len(columns_models[0])))
        df_valid = df_all[df_all.folds == i]
        tmp_valid = np.empty((len(models), len(df_valid), len(columns_models[0])))
        for j in range(len(models)):
            tmp_train[j] = df_train[columns_models[j]]
            tmp_valid[j] = df_valid[columns_models[j]]
        X_train_list.append(tmp_train)
        ids_train_list.append(df_train.ID.to_list())
        X_valid_list.append(tmp_valid)
        ids_valid_list.append(df_valid.ID.to_list())

    train_batch_size_list = []
    for i in range(5):
        train_batch_size_list.append(
            df_all[df_all.folds == i]
            .groupby(["StudyInstanceUID"])
            .apply(lambda x: x.shape[0])
            .values
        )
    test_batch_size = (
        test_list[0].groupby(["StudyInstanceUID"]).apply(lambda x: x.shape[0]).values
    )

    all_train_batch_counter = []
    for i in range(5):
        train_batch_num = 0
        train_batch_counter = []
        for j, train_batch_size in enumerate(train_batch_size_list):
            if i != j:
                for train_batch_n in train_batch_size:
                    train_batch_counter.extend([train_batch_num] * train_batch_n)
                    train_batch_num += 1
        all_train_batch_counter.append(train_batch_counter)

    all_valid_batch_counter = []
    for i in range(5):
        valid_batch_num = 0
        valid_batch_counter = []
        for j in train_batch_size_list[i]:
            valid_batch_counter.extend([valid_batch_num] * j)
            valid_batch_num += 1
        all_valid_batch_counter.append(valid_batch_counter)

    test_batch_num = 0
    test_batch_counter = []
    for j in test_batch_size:
        test_batch_counter.extend([test_batch_num] * j)
        test_batch_num += 1

    y_train_list = []
    y_valid_list = []
    for i in range(5):
        df = df_all[df_all.folds != i]
        y_train_list.append(df[target_cols].values)
        df = df_all[df_all.folds == i]
        y_valid_list.append(df[target_cols].values)

    X_test = np.empty(
        (len(test_list), len(models), len(test_list[0]), len(columns_models[0]))
    )
    X_test_list = []
    ids_test_list = []
    for i in range(len(test_list)):
        tmp_test = np.empty((len(models), len(test_list[0]), len(columns_models[0])))
        for j in range(len(models)):
            tmp_test[j] = test_list[i][columns_models[j]]
        X_test_list.append(tmp_test)
        ids_test_list.append(test_list[i].ID.to_list())

    for i, X_train in enumerate(X_train_list):
        X_train_list[i] = X_train.transpose(1, 0, 2)
    for i, X_valid in enumerate(X_valid_list):
        X_valid_list[i] = X_valid.transpose(1, 0, 2)
    for i, X_test in enumerate(X_test_list):
        X_test_list[i] = X_test.transpose(1, 0, 2)

    def get_loader(fold, mode):
        if mode == "train":
            train_loader = DataLoader(
                StackingContinuousDataset(
                    X_train_list[fold],
                    all_train_batch_counter[fold],
                    ids_train_list[fold],
                    y_train_list[fold],
                ),
                shuffle=True,
                batch_size=512,
                num_workers=8,
            )
            return train_loader
        elif mode == "valid":
            valid_loader = DataLoader(
                StackingContinuousDataset(
                    X_valid_list[fold],
                    all_valid_batch_counter[fold],
                    ids_valid_list[fold],
                    y_valid_list[fold],
                ),
                batch_size=512,
                shuffle=False,
                num_workers=8,
            )
            return valid_loader
        elif mode == "test":
            test_loader = DataLoader(
                StackingContinuousDatasetTest(
                    X_test_list[fold], test_batch_counter, ids_test_list[fold]
                ),
                batch_size=512,
                shuffle=False,
                num_workers=8,
            )
            return test_loader
        else:
            assert False

    # cnn stacking main
    pred_test = []
    pred_test_ids = []
    pred_valid = []
    pred_valid_ids = []

    eps = 1e-4

    for i in range(5):
        print("fold {}/5\n".format(i + 1))
        tr = get_loader(i, "train")
        va = get_loader(i, "valid")
        te = get_loader(i, "test")

        model = StackingModel().cuda()
        optimizer = torch.optim.Adam(model.parameters(), 2e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[1, 3, 6, 9, 12], gamma=0.5
        )
        criterion = nn.BCEWithLogitsLoss(
            weight=torch.FloatTensor([2, 1, 1, 1, 1, 1]).cuda()
        )

        n_epoch = 15
        pred_valid_fold = []
        pred_valid_ids_fold = []
        for epoch in range(n_epoch):
            model.train()
            losses = []
            for x, ids, y in tr:
                pred = model(x.cuda())
                loss = criterion(pred, y.cuda())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses.append(loss.item())

            losses_valid = []
            pred_valid_metric = []
            target_valid = []
            with torch.no_grad():
                model.eval()
                for x, ids, y in va:
                    pred = model(x.cuda())
                    loss = criterion(pred, y.cuda())
                    losses_valid.append(loss.item())
                    if epoch == n_epoch - 1:
                        pred_valid_ids_fold.extend(ids)
                        pred_valid_fold.append(
                            torch.sigmoid(pred).detach().cpu().numpy()
                        )
                    pred_valid_metric.append(torch.sigmoid(pred).detach().cpu().numpy())
                    target_valid.append(y.detach().cpu().numpy())
                pred_valid_metric = np.concatenate(pred_valid_metric, axis=0)
                target_valid = np.concatenate(target_valid, axis=0)
                metrics = []
                for i in range(6):
                    pred_valid_metric[:, i] = np.clip(
                        pred_valid_metric[:, i], eps, 1 - eps
                    )
                for i in range(6):
                    metrics.append(
                        log_loss(np.floor(target_valid[:, i]), pred_valid_metric[:, i])
                    )
                metrics = np.average(metrics, weights=[2, 1, 1, 1, 1, 1])
                print(
                    f"Epoch[{epoch}]",
                    "Train:",
                    np.mean(losses),
                    "Valid:",
                    np.mean(losses_valid),
                    "Score",
                    metrics,
                )
            scheduler.step()

        with torch.no_grad():
            model.eval()
            pred_test_fold = []
            pred_test_ids_fold = []
            for x, ids in te:
                pred = model(x.cuda())
                pred_test_fold.append(torch.sigmoid(pred).detach().cpu().numpy())
                pred_test_ids_fold.extend(ids)
        pred_test.append(pred_test_fold)
        pred_test_ids.append(pred_test_ids_fold)
        pred_valid.append(pred_valid_fold)
        pred_valid_ids.append(pred_valid_ids_fold)

        # torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, "fold{}.pt".format(i))

    array_pred_valid = []
    for i in range(5):
        tmp = np.zeros((1, 6))
        for j in range(len(pred_valid[i])):
            tmp = np.concatenate((tmp, pred_valid[i][j]), axis=0)
        array_pred_valid.append(tmp[1:])

    array_pred_test = []
    for i in range(5):
        tmp = np.zeros((1, 6))
        for j in range(len(pred_test[i])):
            tmp = np.concatenate((tmp, pred_test[i][j]), axis=0)
        array_pred_test.append(tmp[1:])

    for i in range(5):
        with open(
            "intermediate_output/cnn_stacking_1/fold{}_valid.pkl".format(i), "wb"
        ) as f:
            pickle.dump({"ids": ids_valid_list[i], "outputs": array_pred_valid[i]}, f)

    sub_id_list = pd.read_csv("input/stage_2_sample_submission.csv")
    ids = []
    for i in range(len(sub_id_list)):
        if i % 6 == 0:
            ids.append(sub_id_list.iloc[i].ID.replace("_epidural", ""))

    for i in range(5):
        sub = pd.DataFrame(ids, columns=["ID"])
        test_sub = pd.DataFrame(array_pred_test[i], columns=target_cols)
        test_sub["ID"] = pred_test_ids[i]
        sub = pd.merge(sub, test_sub, on="ID", how="left")
        sub = sub.fillna(min(sub[target_cols].min()))
        with open(
            "intermediate_output/cnn_stacking_1/fold{}_test.pkl".format(i), "wb"
        ) as f:
            pickle.dump({"ids": sub.ID.tolist(), "outputs": sub[target_cols].values}, f)


def main():
    # first time
    df_all, test_list = make_dataset()

    # not first time
    # df_all, test_list = load_data()

    # cnn stacking main function
    cnn_stacking(df_all, test_list)


if __name__ == "__main__":
    main()
