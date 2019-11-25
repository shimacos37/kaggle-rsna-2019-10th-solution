# load
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
import re

tqdm.pandas(desc="my bar!")

target_cols = [
    "any",
    "epidural",
    "subdural",
    "subarachnoid",
    "intraventricular",
    "intraparenchymal",
]
tr_meta = pd.read_pickle("./intermediate_output/preprocessed_data/train_raw.pkl")
ts_meta = pd.read_pickle("./intermediate_output/preprocessed_data/test_raw.pkl")

n_tta = 5


def get_train_df(path):
    # train data
    df_all = []
    for n_fold in range(5):
        df = pd.read_pickle(f"{path}/fold{n_fold}_valid.pkl")
        # 予測値
        tmp = np.zeros([len(df[0]["ids"]), 6])
        for i in range(n_tta):
            tmp += df[i]["outputs"] / n_tta
        tmp = pd.DataFrame(tmp)
        tmp.columns = [tar_col + "_pred" for tar_col in target_cols]
        tmp["ID"] = df[0]["ids"]
        tmp["folds"] = n_fold
        # 実測値
        tmp2 = pd.DataFrame(df[0]["targets"], columns=target_cols)
        df_all.append(pd.concat([tmp, tmp2], axis=1))

    df_all = pd.concat(df_all)
    train = pd.merge(df_all, tr_meta, on="ID", how="inner")
    return train


def get_test_df_list(path):
    # test data
    df_all_ts = []
    for n_fold in range(5):
        df = pd.read_pickle(f"{path}/fold{n_fold}_test.pkl")
        tmp = np.zeros([len(df[0]["ids"]), 6])
        for i in range(n_tta):
            tmp += df[i]["outputs"] / n_tta
        tmp = pd.DataFrame(tmp)
        tmp.columns = [tar_col + "_pred" for tar_col in target_cols]
        tmp["ID"] = df[n_fold]["ids"]
        tmp["folds"] = n_fold
        tmp = pd.merge(tmp, ts_meta, on="ID", how="inner")
        df_all_ts.append(tmp)
    return df_all_ts


def ortho_df(df):
    df["ImagePositionPatient_2"] = df["ImagePositionPatient"].progress_apply(
        lambda x: x[2]
    )
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


def pred_agg1(df):
    new_feats = []
    for c in target_cols:
        tmp = (
            df.groupby(["StudyInstanceUID"])[c + "_pred"]
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
        if c != "any":
            del tmp["StudyInstanceUID"]
        new_feats.append(tmp)
    new_feats = pd.concat(new_feats, axis=1)
    df = pd.merge(df, new_feats, on="StudyInstanceUID", how="left")

    new_feats = []
    for c in target_cols:
        tmp = (
            df.groupby(["PatientID"])[c + "_pred"]
            .agg(["min", "max", "mean", "std"])
            .reset_index()
        )
        tmp.columns = [
            "PatientID",
            c + "_min_PatientID",
            c + "_max_PatientID",
            c + "_mean_PatientID",
            c + "_std_PatientID",
        ]
        if c != "any":
            del tmp["PatientID"]
        new_feats.append(tmp)
    new_feats = pd.concat(new_feats, axis=1)
    df = pd.merge(df, new_feats, on="PatientID", how="left")

    for c in target_cols:
        df[c + "_diff"] = df[c + "_pred"] - df[c + "_mean"]
        df[c + "_div"] = df[c + "_pred"] / df[c + "_mean"]
        df[c + "_scaled"] = (df[c + "_pred"] - df[c + "_mean"]) / df[c + "_std"]
    return df


def pred_agg2(df):
    a1 = (
        df.groupby("StudyInstanceUID")[
            [col for col in df.columns if col.endswith("_pred")]
        ]
        .rolling(3, min_periods=1, center=True)
        .mean()
        .values
    )
    a2 = (
        df.groupby("StudyInstanceUID")[
            [col for col in df.columns if col.endswith("_pred")]
        ]
        .rolling(5, min_periods=1, center=True)
        .mean()
        .values
    )
    new_feats1 = pd.DataFrame(a1, columns=[c + "_3roll" for c in target_cols])
    new_feats2 = pd.DataFrame(a2, columns=[c + "_5roll" for c in target_cols])
    new_feats1.index = df.index
    new_feats2.index = df.index
    a3 = (
        df.groupby("StudyInstanceUID")[
            [col for col in df.columns if col.endswith("_pred")]
        ]
        .rolling(1, min_periods=1, center=True)
        .mean()
        .values
    )
    new_feats17 = pd.DataFrame(a1 / a3, columns=[c + "_3rolldiv" for c in target_cols])
    new_feats18 = pd.DataFrame(a2 / a3, columns=[c + "_5rolldiv" for c in target_cols])
    new_feats17.index = df.index
    new_feats18.index = df.index

    a3 = (
        df.groupby("StudyInstanceUID")[
            [col for col in df.columns if col.endswith("_pred")]
        ]
        .rolling(7, min_periods=1, center=True)
        .mean()
        .values
    )
    a4 = (
        df.groupby("StudyInstanceUID")[
            [col for col in df.columns if col.endswith("_pred")]
        ]
        .rolling(9, min_periods=1, center=True)
        .mean()
        .values
    )
    new_feats3 = pd.DataFrame(a3, columns=[c + "_7roll" for c in target_cols])
    new_feats4 = pd.DataFrame(a4, columns=[c + "_9roll" for c in target_cols])
    new_feats3.index = df.index
    new_feats4.index = df.index

    a5 = (
        df.groupby("StudyInstanceUID")[
            [col for col in df.columns if col.endswith("_pred")]
        ]
        .rolling(3, min_periods=1, center=True)
        .min()
        .values
    )
    a6 = (
        df.groupby("StudyInstanceUID")[
            [col for col in df.columns if col.endswith("_pred")]
        ]
        .rolling(5, min_periods=1, center=True)
        .min()
        .values
    )
    new_feats5 = pd.DataFrame(a5, columns=[c + "_3rollmin" for c in target_cols])
    new_feats6 = pd.DataFrame(a6, columns=[c + "_5rollmin" for c in target_cols])
    new_feats5.index = df.index
    new_feats6.index = df.index
    a7 = (
        df.groupby("StudyInstanceUID")[
            [col for col in df.columns if col.endswith("_pred")]
        ]
        .rolling(7, min_periods=1, center=True)
        .min()
        .values
    )
    a8 = (
        df.groupby("StudyInstanceUID")[
            [col for col in df.columns if col.endswith("_pred")]
        ]
        .rolling(9, min_periods=1, center=True)
        .min()
        .values
    )
    new_feats7 = pd.DataFrame(a3, columns=[c + "_7rollmin" for c in target_cols])
    new_feats8 = pd.DataFrame(a4, columns=[c + "_9rollmin" for c in target_cols])
    new_feats7.index = df.index
    new_feats8.index = df.index

    a9 = (
        df.groupby("StudyInstanceUID")[
            [col for col in df.columns if col.endswith("_pred")]
        ]
        .rolling(3, min_periods=1, center=True)
        .max()
        .values
    )
    a10 = (
        df.groupby("StudyInstanceUID")[
            [col for col in df.columns if col.endswith("_pred")]
        ]
        .rolling(5, min_periods=1, center=True)
        .max()
        .values
    )
    new_feats9 = pd.DataFrame(a9, columns=[c + "_3rollmax" for c in target_cols])
    new_feats10 = pd.DataFrame(a10, columns=[c + "_5rollmax" for c in target_cols])
    new_feats9.index = df.index
    new_feats10.index = df.index
    a11 = (
        df.groupby("StudyInstanceUID")[
            [col for col in df.columns if col.endswith("_pred")]
        ]
        .rolling(7, min_periods=1, center=True)
        .max()
        .values
    )
    a12 = (
        df.groupby("StudyInstanceUID")[
            [col for col in df.columns if col.endswith("_pred")]
        ]
        .rolling(9, min_periods=1, center=True)
        .max()
        .values
    )
    new_feats11 = pd.DataFrame(a11, columns=[c + "_7rollmax" for c in target_cols])
    new_feats12 = pd.DataFrame(a12, columns=[c + "_9rollmax" for c in target_cols])
    new_feats11.index = df.index
    new_feats12.index = df.index

    a13 = (
        df.groupby("StudyInstanceUID")[
            [col for col in df.columns if col.endswith("_pred")]
        ]
        .rolling(3, min_periods=1, center=True)
        .max()
        .values
    )
    a14 = (
        df.groupby("StudyInstanceUID")[
            [col for col in df.columns if col.endswith("_pred")]
        ]
        .rolling(5, min_periods=1, center=True)
        .max()
        .values
    )
    new_feats13 = pd.DataFrame(a9, columns=[c + "_3rollstd" for c in target_cols])
    new_feats14 = pd.DataFrame(a10, columns=[c + "_5rollstd" for c in target_cols])
    new_feats13.index = df.index
    new_feats14.index = df.index
    a15 = (
        df.groupby("StudyInstanceUID")[
            [col for col in df.columns if col.endswith("_pred")]
        ]
        .rolling(7, min_periods=1, center=True)
        .std()
        .values
    )
    a16 = (
        df.groupby("StudyInstanceUID")[
            [col for col in df.columns if col.endswith("_pred")]
        ]
        .rolling(9, min_periods=1, center=True)
        .std()
        .values
    )
    new_feats15 = pd.DataFrame(a15, columns=[c + "_7rollstd" for c in target_cols])
    new_feats16 = pd.DataFrame(a16, columns=[c + "_9rollstd" for c in target_cols])
    new_feats15.index = df.index
    new_feats16.index = df.index
    df = pd.concat(
        [
            df,
            new_feats1,
            new_feats2,
            new_feats3,
            new_feats4,
            new_feats5,
            new_feats6,
            new_feats7,
            new_feats8,
            new_feats9,
            new_feats10,
            new_feats11,
            new_feats12,
            new_feats13,
            new_feats14,
            new_feats15,
            new_feats16,
            new_feats17,
            new_feats18,
        ],
        axis=1,
    )
    #     df = pd.concat([df, new_feats1, new_feats2, new_feats17, new_feats18], axis=1)
    return df


def make_feats(df, model_name="appian", df_type="train"):
    df = ortho_df(df)
    df = pred_agg1(df)
    df = pred_agg2(df)
    target_str = "{}|{}|{}|{}|{}|{}".format(
        target_cols[0],
        target_cols[1],
        target_cols[2],
        target_cols[3],
        target_cols[4],
        target_cols[5],
    )
    if df_type == "train":
        X_cols = [
            c
            for c in df.columns.drop(target_cols)
            if len(re.findall(target_str, c)) > 0
        ]
        df = df[["ID", "folds", "StudyInstanceUID"] + X_cols + target_cols]
        df.columns = (
            ["ID", "folds", "StudyInstanceUID"]
            + [model_name + "_" + c for c in X_cols]
            + target_cols
        )
    elif df_type == "test":
        X_cols = [c for c in df.columns if len(re.findall(target_str, c)) > 0]
        df = df[["ID", "folds", "StudyInstanceUID"] + X_cols]
        df.columns = ["ID", "folds", "StudyInstanceUID"] + [
            model_name + "_" + c for c in X_cols
        ]
    return df


if __name__ == "__main__":
    model_list = [
        "cnn_stacking_1_for_sugawara_stacking2",
        "cnn_stacking_2_for_sugawara_stacking2",
        "stacking1_lgbm_for_sugawara_stacking2",
        "stacking1_mlp_for_sugawara_stacking2",
    ]
    train_df_list = []
    for model in model_list:
        path = f"./intermediate_output/{model}/"
        train = get_train_df(path)
        train = make_feats(train, model, "train")
        X_cols = train.columns.drop(
            ["ID", "folds", "StudyInstanceUID"] + target_cols
        ).tolist()
        train_df_list.append(train[["ID"] + X_cols])
    train_df_list.append(train[train.columns.drop(X_cols)])
    train = train_df_list[0]
    for tmp_train in train_df_list[1:]:
        train = pd.merge(train, tmp_train, on="ID", how="outer")
    X_cols = train.columns.drop(["ID", "folds", "StudyInstanceUID"] + target_cols)

    path = f"./intermediate_output/model_base/"
    test_list = get_test_df_list(path)
    test_model_list = []
    # モデルごとのループ
    for model in model_list:
        path = f"./intermediate_output/{model}/"
        test_pred_original_list = get_test_df_list(path)
        test_list = []

        # foldごとの予測値を取り出すループ
        for test_tmp in test_pred_original_list:
            test = make_feats(test_tmp, model, "test")
            X_cols = test.columns.drop(["ID", "folds", "StudyInstanceUID"]).tolist()
            test_list.append(test[["ID"] + X_cols])
        test_model_list.append(test_list)

    test_list = []
    # fold毎に各モデルの予測値が横につながった1csvになるようにする
    for n_fold in range(5):
        tmp = []
        for n_model in range(len(model_list)):
            tmp.append(test_model_list[n_model][n_fold])
        test = tmp[0]
        for tmp_test in tmp[1:]:
            test = pd.merge(test, tmp_test, on="ID", how="outer")
        test_list.append(test)

    os.makedirs(
        "./intermediate_output/agged_feats_for_second_stacking_sugawara", exist_ok=True
    )
    with open(
        "./intermediate_output/agged_feats_for_second_stacking_sugawara/test_stackfeats.pickle",
        "wb",
    ) as f:
        pickle.dump(test_list, f)
    with open(
        "./intermediate_output/agged_feats_for_second_stacking_sugawara/train_stackfeats.pickle",
        "wb",
    ) as f:
        pickle.dump(train, f)
