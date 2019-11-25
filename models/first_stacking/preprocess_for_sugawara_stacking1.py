# load
import pandas as pd
import pickle
import os


def arrange_shimakoshi_test1(shimakoshi_path):
    target_cols = [
        "any",
        "epidural",
        "subdural",
        "subarachnoid",
        "intraventricular",
        "intraparenchymal",
    ]
    pred_cols = [c + "_pred" for c in target_cols]
    for n_fold in range(5):
        df_true = pd.read_pickle(
            f"./intermediate_output/model_base/fold{n_fold}_test.pkl"
        )

        df = (
            pd.read_pickle(
                f"{shimakoshi_path}/fold{n_fold}/result/test_class_pred.pkl"
            )[["ID"] + pred_cols]
            .sort_values("ID")
            .reset_index(drop=True)
        )
        for n_tta in range(5):
            df_true[n_tta]["outputs"] = df[pred_cols].values
        os.makedirs(f"{shimakoshi_path}_for_sugawara_stacking", exist_ok=True)
        with open(
            f"{shimakoshi_path}_for_sugawara_stacking/fold{n_fold}_test.pkl", "wb"
        ) as f:
            pickle.dump(df_true, f)

def arrange_shimakoshi_valid1(shimakoshi_path):
    target_cols = [
        "any",
        "epidural",
        "subdural",
        "subarachnoid",
        "intraventricular",
        "intraparenchymal",
    ]
    pred_cols = [c + "_pred" for c in target_cols]
    for n_fold in range(5):
        df_true = pd.read_pickle(
            f"./intermediate_output/model_base/fold{n_fold}_valid.pkl"
        )

        df = (
            pd.read_pickle(
                f"{shimakoshi_path}/fold{n_fold}/result/valid_class_pred.pkl"
            )[["ID"] + pred_cols]
            .sort_values("ID")
            .reset_index(drop=True)
        )
        for n_tta in range(5):
            df_true[n_tta]["outputs"] = df[pred_cols].values
        os.makedirs(f"{shimakoshi_path}_for_sugawara_stacking", exist_ok=True)
        with open(
            f"{shimakoshi_path}_for_sugawara_stacking/fold{n_fold}_valid.pkl", "wb"
        ) as f:
            pickle.dump(df_true, f)


if __name__ == "__main__":
    arrange_shimakoshi_test1("./intermediate_output/seresnext50_label_smoothing")
    arrange_shimakoshi_test1(
        "./intermediate_output/seresnext50_label_smoothing_without_any"
    )
    arrange_shimakoshi_test1("./intermediate_output/densenet_adj_prediction")
    arrange_shimakoshi_valid1("./intermediate_output/seresnext50_label_smoothing")
    arrange_shimakoshi_valid1(
        "./intermediate_output/seresnext50_label_smoothing_without_any"
    )
    arrange_shimakoshi_valid1("./intermediate_output/densenet_adj_prediction")

