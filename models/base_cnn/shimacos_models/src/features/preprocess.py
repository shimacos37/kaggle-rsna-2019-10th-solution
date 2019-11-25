import copy
import pickle
from glob import glob
import os
import pickle

import pandas as pd
import pydicom
import numpy as np
from tqdm import tqdm

tqdm.pandas()


def fix_pxrepr(dcm):
    if dcm.PixelRepresentation != 0 or dcm.RescaleIntercept < -100:
        return dcm
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x >= px_mode] = x[x >= px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000
    return dcm


def extract_brain_area_rate(id_):
    path = f"./input/stage_2_train_images/{id_}.dcm"
    dicom = pydicom.dcmread(path)
    dicom = fix_pxrepr(dicom)
    img = dicom.pixel_array
    img = img * dicom.RescaleSlope + dicom.RescaleIntercept
    rate = np.where((img >= 0) & (img <= 80))[0].shape[0] / (
        img.shape[0] * img.shape[1]
    )
    return rate


def make_target_array(labels):
    label_to_num = {
        "any": 0,
        "epidural": 1,
        "subdural": 2,
        "subarachnoid": 3,
        "intraventricular": 4,
        "intraparenchymal": 5,
    }
    target = np.zeros([labels.shape[0], 6])
    for i, label in enumerate(labels):
        if label != "":
            for l in label.split(" "):
                target[i, label_to_num[l]] = 1
    return target


def main():
    target_cols = [
        "any",
        "epidural",
        "subdural",
        "subarachnoid",
        "intraventricular",
        "intraparenchymal",
    ]
    train_df = pd.read_pickle("./intermediate_output/preprocessed_data/train_folds.pkl")
    train_df = train_df.merge(
        pd.read_pickle("./intermediate_output/preprocessed_data/train_raw.pkl")[
            ["ID", "StudyInstanceUID"]
        ],
        on="ID",
    )
    train_df["brain_region_rate"] = train_df["ID"].progress_apply(
        lambda x: extract_brain_area_rate(x)
    )
    train_df = train_df.sort_values(by=["StudyInstanceUID", "PositionOrd"])
    train_df = train_df.reset_index(drop=True)
    target = make_target_array(train_df["labels"].values)
    for i, col in enumerate(target_cols):
        train_df[col] = target[:, i]
    smooth_target = (
        train_df.groupby("StudyInstanceUID")[target_cols]
        .rolling(3, center=True, min_periods=1)
        .mean()
        .values
    )

    train_df[[f"{col}_smooth" for col in target_cols]] = pd.DataFrame(
        smooth_target, index=train_df.index
    )
    train_df.to_csv(
        "./intermediate_output/preprocessed_data/train_fold_with_label_smooth.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
