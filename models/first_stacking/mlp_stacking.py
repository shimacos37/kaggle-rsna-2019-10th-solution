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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

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
    X_cols = [c for c in X_cols if not c.endswith("div")]

    ##############################################################################
    # stacking train and predict
    scaler = MinMaxScaler()
    _mean = train[X_cols].mean()
    scaler.fit(train[X_cols].fillna(_mean))
    model = Sequential()
    model.add(Dense(2048, input_dim=len(X_cols)))
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam")

    print("Training...")

    from sklearn.linear_model import Ridge, LogisticRegression

    stack_preds = []
    pred_test = []
    fi = {}
    for c in target_cols:
        fi[c] = []
    stack_preds_valid = []

    for i in range(5):
        tr = train.query("folds != @i")
        tr[X_cols] = tr[X_cols].fillna(_mean)
        va = train.query("folds == @i")
        va[X_cols] = va[X_cols].fillna(_mean)
        print("fitting")
        preds = pd.DataFrame(
            np.zeros([len(va), 6]), columns=[c + "_pred" for c in target_cols]
        )
        for tar_col in target_cols:
            es_cb = EarlyStopping(
                monitor="val_loss", patience=2, verbose=1, mode="auto"
            )
            chkpt = f"stack2_{i}fold_{tar_col}.hdf5"
            cp_cb = ModelCheckpoint(
                filepath=chkpt,
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="auto",
            )
            model.fit(
                scaler.transform(tr[X_cols].values),
                tr[tar_col].values,
                nb_epoch=100,
                batch_size=512,
                validation_data=(
                    scaler.transform(va[X_cols].values),
                    va[tar_col].values,
                ),
                verbose=2,
                callbacks=[es_cb, cp_cb],
            )
            model.load_weights(chkpt)
            preds[tar_col + "_pred"] = model.predict(
                scaler.transform(va[X_cols].values)
            )
            pred_test.append(
                model.predict(scaler.transform(test_list[i][X_cols].values))
            )
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
            ] = pred_test[n_fold * 6 + n_target].reshape(-1)
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
    os.makedirs("./intermediate_output/stacking1_mlp", exist_ok=True)
    path_to = "./intermediate_output/stacking1_mlp/pred_test.pkl"
    with open(path_to, "wb") as f:
        pickle.dump(sub, f)
    path_to = "./intermediate_output/stacking1_mlp/pred_valid.pkl"
    with open(path_to, "wb") as f:
        pickle.dump(stack_preds, f)


if __name__ == "__main__":
    main()
