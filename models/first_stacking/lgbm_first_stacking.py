# load
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import log_loss
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import KFold
import re
import os
import gc

tqdm.pandas(desc="my bar!")


def main():
    ##############################################################################
    # load aggregated data
    tr_meta = pd.read_pickle("./intermediate_output/preprocessed_data/train_raw.pkl")
    ts_meta = pd.read_pickle("./intermediate_output/preprocessed_data/test_raw.pkl")
    target_cols = [
        "any",
        "epidural",
        "subdural",
        "subarachnoid",
        "intraventricular",
        "intraparenchymal",
    ]
    with open(
        "./intermediate_output/agged_feats_for_first_stacking_sugawara/test_stackfeats.pickle",
        "rb",
    ) as f:
        test_list = pickle.load(f)
    with open(
        "./intermediate_output/agged_feats_for_first_stacking_sugawara/train_stackfeats.pickle",
        "rb",
    ) as f:
        train = pickle.load(f)
    X_cols = test_list[0].columns.drop(["ID"])

    ##############################################################################
    # stacking train and predict
    stack_preds = []
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "learning_rate": 0.1,
        "num_leaves": 5,
        "max_bin": 256,
        "feature_fraction": 0.8,
        "verbosity": 0,
        "min_child_samples": 10,
        "min_child_weight": 150,
        "min_split_gain": 0,
        "subsample": 0.9,
    }
    pred_test = []
    fi = {}
    for c in target_cols:
        fi[c] = []
    stack_preds_valid = []
    for i in range(5):
        tr = train.query("folds != @i")
        va = train.query("folds == @i")
        preds = pd.DataFrame(
            np.zeros([len(va), 6]), columns=[c + "_pred" for c in target_cols]
        )
        for tar_col in target_cols:
            tr_D = lgb.Dataset(tr[X_cols], tr[tar_col])
            va_D = lgb.Dataset(va[X_cols], va[tar_col])
            clf = lgb.train(
                params,
                tr_D,
                10000,
                valid_sets=va_D,
                verbose_eval=100,
                early_stopping_rounds=120,
            )
            preds[tar_col + "_pred"] = clf.predict(va[X_cols])
            pred_test.append(clf.predict(test_list[i][X_cols]))
            df_fi = pd.DataFrame(
                clf.feature_importance(importance_type="gain"),
                index=X_cols,
                columns=["FI_score"],
            )
            fi[tar_col].append(df_fi)
        stack_preds.append([va["ID"], preds])

    ##############################################################################
    # arrange the predictions
    path = "./intermediate_output/model_base"
    sub = pd.read_pickle(f"{path}/fold0_test.pkl")
    test = test_list[0]
    preds_test_df = pd.DataFrame()
    for n_fold in range(5):
        for n_target in range(6):
            preds_test_df[
                target_cols[n_target] + "_" + str(n_fold) + "fold"
            ] = pred_test[n_fold * 6 + n_target]
    preds_test_df.index = test["ID"].values
    preds_test_df = preds_test_df.loc[sub[0]["ids"]]
    pred_test_sorted_list = []
    for i in range(30):
        pred_test_sorted_list.append(preds_test_df.iloc[:, i].values)
    pred_test_df_list = []
    for n_fold in range(5):
        tmp = np.zeros([len(test), 6])
        for n_tar in range(6):
            tmp[:, n_tar] = pred_test_sorted_list[6 * n_fold + n_tar]
        pred_test_df = pd.DataFrame(tmp, columns=target_cols)
        pred_test_df_list.append(pred_test_df)
    sub = pd.read_pickle(f"{path}/fold{n_fold}_test.pkl")
    sub_original = pd.read_pickle(f"{path}/fold{n_fold}_test.pkl")
    for i in range(5):
        sub[i]["outputs"] = pred_test_df_list[i][target_cols].values
    # save the prediction
    os.makedirs("./intermediate_output/stacking1_lgbm", exist_ok=True)
    path_to = "./intermediate_output/stacking1_lgbm/pred_test.pkl"
    with open(path_to, "wb") as f:
        pickle.dump(sub, f)
    path_to = "./intermediate_output/stacking1_lgbm/pred_valid.pkl"
    with open(path_to, "wb") as f:
        pickle.dump(stack_preds, f)


if __name__ == "__main__":
    main()
