# load
import pandas as pd
import pickle
import os

target_cols = [
    "any",
    "epidural",
    "subdural",
    "subarachnoid",
    "intraventricular",
    "intraparenchymal",
]
pred_cols = [c + "_pred" for c in target_cols]


def arrange_cnn1_test():
    for n_fold in range(5):
        df_true = pd.read_pickle(
            f"./intermediate_output/model_base/fold{n_fold}_test.pkl"
        )

        df = pd.read_pickle(
            f"./intermediate_output/cnn_stacking_1/fold{n_fold}_test.pkl"
        )  # [['ID'] + pred_cols].sort_values('ID').reset_index(drop=True)
        tmp_df = pd.DataFrame(df["outputs"], columns=pred_cols)
        tmp_df["ID"] = df["ids"]
        df = (
            tmp_df.query(f"ID in {df_true[0]['ids']}")
            .sort_values("ID")
            .reset_index(drop=True)
        )
        for n_tta in range(5):
            df_true[n_tta]["outputs"] = df[pred_cols].values
        os.makedirs(
            f"./intermediate_output/cnn_stacking_1_for_sugawara_stacking2",
            exist_ok=True,
        )
        with open(
            f"./intermediate_output/cnn_stacking_1_for_sugawara_stacking2/fold{n_fold}_test.pkl",
            "wb",
        ) as f:
            pickle.dump(df_true, f)


def arrange_cnn1_valid():
    for n_fold in range(5):
        df_true = pd.read_pickle(
            f"./intermediate_output/model_base/fold{n_fold}_valid.pkl"
        )

        df = pd.read_pickle(
            f"./intermediate_output/cnn_stacking_1/fold{n_fold}_valid.pkl"
        )  # [['ID'] + pred_cols].sort_values('ID').reset_index(drop=True)
        tmp_df = pd.DataFrame(df["outputs"], columns=pred_cols)
        tmp_df["ID"] = df["ids"]
        df = (
            tmp_df.query(f"ID in {df_true[0]['ids']}")
            .sort_values("ID")
            .reset_index(drop=True)
        )
        for n_tta in range(5):
            df_true[n_tta]["outputs"] = df[pred_cols].values
        os.makedirs(
            f"./intermediate_output/cnn_stacking_1_for_sugawara_stacking2",
            exist_ok=True,
        )
        with open(
            f"./intermediate_output/cnn_stacking_1_for_sugawara_stacking2/fold{n_fold}_valid.pkl",
            "wb",
        ) as f:
            pickle.dump(df_true, f)


def arrange_cnn2_test():
    file_path = "./intermediate_output/cnn_stacking_2"
    for n_fold in range(5):
        df_true = pd.read_pickle(
            f"./intermediate_output/model_base/fold{n_fold}_test.pkl"
        )
        df = (
            pd.read_pickle(f"{file_path}/stacking_test_pred_fold{n_fold}.pkl")[
                ["ID"] + pred_cols
            ]
            .sort_values("ID")
            .reset_index(drop=True)
        )
        for n_tta in range(5):
            df_true[n_tta]["outputs"] = df[pred_cols].values
        os.makedirs(f"{file_path}_for_sugawara_stacking2", exist_ok=True)
        with open(
            f"{file_path}_for_sugawara_stacking2/fold{n_fold}_test.pkl", "wb"
        ) as f:
            pickle.dump(df_true, f)


def arrange_cnn2_valid():
    file_path = "./intermediate_output/cnn_stacking_2"
    for n_fold in range(5):
        df_true = pd.read_pickle(
            f"./intermediate_output/model_base/fold{n_fold}_valid.pkl"
        )
        df = (
            pd.read_pickle(f"{file_path}/stacking_valid_pred_fold{n_fold}.pkl")[
                ["ID"] + pred_cols
            ]
            .sort_values("ID")
            .reset_index(drop=True)
        )
        for n_tta in range(5):
            df_true[n_tta]["outputs"] = df[pred_cols].values
        os.makedirs(f"{file_path}_for_sugawara_stacking2", exist_ok=True)
        with open(
            f"{file_path}_for_sugawara_stacking2/fold{n_fold}_valid.pkl", "wb"
        ) as f:
            pickle.dump(df_true, f)


def arrange_lgbm1_test():
    file_path = "./intermediate_output/stacking1_lgbm"
    for n_fold in range(5):
        df_true = pd.read_pickle(
            f"./intermediate_output/model_base/fold{n_fold}_test.pkl"
        )
        df = pd.read_pickle(
            f"{file_path}/pred_test.pkl"
        )  # [['ID'] + pred_cols].sort_values('ID').reset_index(drop=True)
        for n_tta in range(5):
            df_true[n_tta]["outputs"] = df[n_tta]["outputs"]
        os.makedirs(f"{file_path}_for_sugawara_stacking2", exist_ok=True)
        with open(
            f"{file_path}_for_sugawara_stacking2/fold{n_fold}_test.pkl", "wb"
        ) as f:
            pickle.dump(df_true, f)


def arrange_lgbm1_valid():
    file_path = "./intermediate_output/stacking1_lgbm"
    for n_fold in range(5):
        df_true = pd.read_pickle(
            f"./intermediate_output/model_base/fold{n_fold}_valid.pkl"
        )
        df = pd.read_pickle(
            f"{file_path}/pred_valid.pkl"
        )  # [['ID'] + pred_cols].sort_values('ID').reset_index(drop=True)
        tmp_df = df[n_fold][1]
        tmp_df["ID"] = df[n_fold][0].values
        df = tmp_df.sort_values("ID").reset_index(drop=True)
        for n_tta in range(5):
            df_true[n_tta]["outputs"] = df[pred_cols].values
        os.makedirs(f"{file_path}_for_sugawara_stacking2", exist_ok=True)
        with open(
            f"{file_path}_for_sugawara_stacking2/fold{n_fold}_valid.pkl", "wb"
        ) as f:
            pickle.dump(df_true, f)


def arrange_mlp1_test():
    file_path = "./intermediate_output/stacking1_mlp"
    for n_fold in range(5):
        df_true = pd.read_pickle(
            f"./intermediate_output/model_base/fold{n_fold}_test.pkl"
        )
        df = pd.read_pickle(
            f"{file_path}/pred_test.pkl"
        )  # [['ID'] + pred_cols].sort_values('ID').reset_index(drop=True)
        for n_tta in range(5):
            df_true[n_tta]["outputs"] = df[n_tta]["outputs"]
        os.makedirs(f"{file_path}_for_sugawara_stacking2", exist_ok=True)
        with open(
            f"{file_path}_for_sugawara_stacking2/fold{n_fold}_test.pkl", "wb"
        ) as f:
            pickle.dump(df_true, f)


def arrange_mlp1_valid():
    file_path = "./intermediate_output/stacking1_mlp"
    for n_fold in range(5):
        df_true = pd.read_pickle(
            f"./intermediate_output/model_base/fold{n_fold}_valid.pkl"
        )
        df = pd.read_pickle(
            f"{file_path}/pred_valid.pkl"
        )  # [['ID'] + pred_cols].sort_values('ID').reset_index(drop=True)
        tmp_df = df[n_fold][1]
        tmp_df["ID"] = df[n_fold][0].values
        df = tmp_df.sort_values("ID").reset_index(drop=True)
        for n_tta in range(5):
            df_true[n_tta]["outputs"] = df[pred_cols].values
        os.makedirs(f"{file_path}_for_sugawara_stacking2", exist_ok=True)
        with open(
            f"{file_path}_for_sugawara_stacking2/fold{n_fold}_valid.pkl", "wb"
        ) as f:
            pickle.dump(df_true, f)


if __name__ == "__main__":
    arrange_cnn1_test()
    arrange_cnn1_valid()
    arrange_cnn2_test()
    arrange_cnn2_valid()
    arrange_lgbm1_test()
    arrange_lgbm1_valid()
    arrange_mlp1_test()
    arrange_mlp1_valid()

