import cv2
import numpy as np
import pandas as pd
from albumentations import (
    Compose,
    Flip,
    RandomCrop,
    Resize,
    Transpose,
    RandomBrightnessContrast,
)
from torch.utils.data import Dataset

import pydicom

cv2.setNumThreads(0)


def get_dicom_value(x, cast=int):
    if type(x) in [pydicom.multival.MultiValue, tuple]:
        return cast(x[0])
    else:
        return cast(x)


def cast(value):
    if type(value) is pydicom.valuerep.MultiValue:
        return tuple(value)
    return value


def get_dicom_raw(dicom):
    return {
        attr: cast(getattr(dicom, attr))
        for attr in dir(dicom)
        if attr[0].isupper() and attr not in ["PixelData"]
    }


def rescale_image(image, slope, intercept):
    return image * slope + intercept


def apply_window(image, center, width):
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    return image


class RSNADataset(Dataset):
    def __init__(self, data_config, mode="train"):
        self.data_config = data_config
        self.image_size = data_config.image_size
        self.crop_width = data_config.crop_width
        self.mode = mode
        self.label_to_num = {
            "any": 0,
            "epidural": 1,
            "subdural": 2,
            "subarachnoid": 3,
            "intraventricular": 4,
            "intraparenchymal": 5,
        }
        self.target_cols = [
            "any",
            "epidural",
            "subdural",
            "subarachnoid",
            "intraventricular",
            "intraparenchymal",
        ]

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if mode == "train":
            self.df = pd.read_csv(
                "./intermediate_output/preprocessed_data/train_fold_with_label_smooth.csv"
            )
            if data_config.fold != "all":
                self.df = self.df.query(f"fold != {data_config.fold}")
            # Delete noise
            self.df = self.df.query("brain_region_rate >= 0.02")
        elif mode == "valid":
            self.df = pd.read_csv(
                "./intermediate_output/preprocessed_data/train_fold_with_label_smooth.csv"
            )
            if data_config.fold != "all":
                self.df = self.df.query(f"fold == {data_config.fold}")
        else:
            self.df = np.load(
                "./intermediate_output/preprocessed_data/test.pkl", allow_pickle=True
            )
        self.df = self.df.reset_index(drop=True)

    def _augmentation(self, img, p=0.3):
        aug_list = []
        height, width = img.shape[0], img.shape[1]
        height = np.random.randint(height - 10, height)
        width = np.random.randint(width - 10, width)
        if self.mode == "train":
            aug_list.extend(
                [
                    RandomCrop(height=height, width=width),
                    Resize(self.image_size, self.image_size, cv2.INTER_CUBIC),
                    Flip(p=p),
                    Transpose(p=p),
                    RandomBrightnessContrast(
                        brightness_limit=0.08, contrast_limit=0.08, p=p
                    ),
                ]
            )
        else:
            aug_list.extend([Resize(self.image_size, self.image_size, cv2.INTER_CUBIC)])
        return Compose(aug_list)

    def _fix_pxrepr(self, dicom):
        if dicom.PixelRepresentation != 0 or dicom.RescaleIntercept < -100:
            return dicom
        x = dicom.pixel_array + 1000
        px_mode = 4096
        x[x >= px_mode] = x[x >= px_mode] - px_mode
        dicom.PixelData = x.tobytes()
        dicom.RescaleIntercept = -1000
        return dicom

    def apply_window_policy(self, img, center, width, p=0.3):
        if self.data_config.is_multi_channel:
            img1 = apply_window(img, 40, 80)  # brain
            img2 = apply_window(img, 80, 200)  # subdural
            img3 = apply_window(img, 600, 2800)  # bone
            img4 = apply_window(img, 32, 8)  # stroke
            img5 = apply_window(img, 40, 370)  # soft tissues
            img6 = apply_window(img, center, width)
            img = np.array([img1, img2, img3, img4, img5, img6]).transpose(1, 2, 0)
        else:
            if self.data_config.window_aug and self.mode == "train":
                if np.random.rand() <= p:
                    noise_center = np.random.rand() * 20 - 10
                else:
                    noise_center = 0
                if np.random.rand() <= p:
                    noise_width = np.random.rand() * 20 - 10
                else:
                    noise_width = 0
                img1 = apply_window(img, 40 + noise_center, 80 + noise_width)  # brain
            else:
                img1 = apply_window(img, 40, 80)  # brain
            if self.data_config.window_policy == 1:
                img2 = apply_window(img, 80, 200)  # subdural
                img3 = apply_window(img, center, width)
                img1 = (img1 - 0) / 80
                img2 = (img2 - (-20)) / 200
                img3 = (img3 - img3.min()) / (img3.max() - img3.min())
                img = np.array([img1, img2, img3]).transpose(1, 2, 0)
            elif self.data_config.window_policy == 2:
                img2 = apply_window(img, 80, 200)  # subdural
                img3 = apply_window(img, 40, 380)  # bone
                img1 = (img1 - 0) / 80
                img2 = (img2 - (-20)) / 200
                img3 = (img3 - (-150)) / 380
                img = np.array([img1, img2, img3]).transpose(1, 2, 0)
        for i in range(img.shape[-1]):
            if self.data_config.is_std:
                img[:, :, i] = (img[:, :, i] - img[:, :, i].mean()) / img[:, :, i].std()
            else:
                img[:, :, i] = (img[:, :, i] - img[:, :, i].min()) / (
                    img[:, :, i].max() - img[:, :, i].min()
                )
                img[:, :, i] = (img[:, :, i] - self.mean[i]) / self.std[i]
        return img

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        id_, center, width = self.df.loc[idx, ["ID", "WindowCenter", "WindowWidth"]]
        if self.data_config["is_train"]:
            img_path = f"./input/stage_2_train_images/{id_}.dcm"
            dicom = pydicom.dcmread(img_path)
            dicom = self._fix_pxrepr(dicom)
            img = dicom.pixel_array
            img = rescale_image(img, dicom.RescaleSlope, dicom.RescaleIntercept)
            img = self.apply_window_policy(img, center, width)
            img = self._augmentation(img)(image=img)["image"]
            target = self.df.loc[idx, self.target_cols].values.astype(float)

            return img, target
        else:
            if self.mode == "valid":
                img_path = f"./input/stage_2_train_images/{id_}.dcm"
            else:
                img_path = f"./input/stage_2_test_images/{id_}.dcm"
            dicom = pydicom.dcmread(img_path)
            dicom = self._fix_pxrepr(dicom)
            img = dicom.pixel_array
            img = rescale_image(img, dicom.RescaleSlope, dicom.RescaleIntercept)
            img = self.apply_window_policy(img, center, width)
            img = self._augmentation(img)(image=img)["image"]
            return img


class SmoothRSNADataset(RSNADataset):
    def __init__(self, data_config, mode="train"):
        super(SmoothRSNADataset, self).__init__(data_config, mode)

    def __getitem__(self, idx):
        id_, center, width = self.df.loc[idx, ["ID", "WindowCenter", "WindowWidth"]]
        if self.data_config["is_train"]:
            img_path = f"./input/stage_2_train_images/{id_}.dcm"
            dicom = pydicom.dcmread(img_path)
            dicom = self._fix_pxrepr(dicom)
            img = dicom.pixel_array
            img = rescale_image(img, dicom.RescaleSlope, dicom.RescaleIntercept)
            img = self.apply_window_policy(img, center, width)
            img = self._augmentation(img)(image=img)["image"]
            if self.mode == "train":
                # use smooth label only when training
                target = self.df.loc[
                    idx, [f"{col}_smooth" for col in self.target_cols]
                ].values.astype(float)
            else:
                target = self.df.loc[idx, self.target_cols].values.astype(float)

            return img, target
        else:
            if self.mode == "valid":
                img_path = f"./input/stage_2_train_images/{id_}.dcm"
            else:
                img_path = f"./input/stage_2_test_images/{id_}.dcm"
            dicom = pydicom.dcmread(img_path)
            dicom = self._fix_pxrepr(dicom)
            img = dicom.pixel_array
            img = rescale_image(img, dicom.RescaleSlope, dicom.RescaleIntercept)
            img = self.apply_window_policy(img, center, width)
            img = self._augmentation(img)(image=img)["image"]
            return img


class WithoutAnySmoothRSNADataset(SmoothRSNADataset):
    def __init__(self, data_config, mode="train"):
        super(WithoutAnySmoothRSNADataset, self).__init__(data_config, mode)

    def __getitem__(self, idx):
        id_, center, width = self.df.loc[idx, ["ID", "WindowCenter", "WindowWidth"]]
        if self.data_config["is_train"]:
            img_path = f"./input/stage_2_train_images/{id_}.dcm"
            dicom = pydicom.dcmread(img_path)
            dicom = self._fix_pxrepr(dicom)
            img = dicom.pixel_array
            img = rescale_image(img, dicom.RescaleSlope, dicom.RescaleIntercept)
            img = self.apply_window_policy(img, center, width)
            img = self._augmentation(img)(image=img)["image"]
            if self.mode == "train":
                # use smooth label only when training
                target = self.df.loc[
                    idx, [f"{col}_smooth" for col in self.target_cols[1:]]
                ].values.astype(float)
            else:
                target = self.df.loc[idx, self.target_cols[1:]].values.astype(float)

            return img, target
        else:
            if self.mode == "valid":
                img_path = f"./input/stage_2_train_images/{id_}.dcm"
            else:
                img_path = f"./input/stage_2_test_images/{id_}.dcm"
            dicom = pydicom.dcmread(img_path)
            dicom = self._fix_pxrepr(dicom)
            img = dicom.pixel_array
            img = rescale_image(img, dicom.RescaleSlope, dicom.RescaleIntercept)
            img = self.apply_window_policy(img, center, width)
            img = self._augmentation(img)(image=img)["image"]
            return img


class AdjRSNADataset(RSNADataset):
    def __init__(self, data_config, mode="train"):
        super(AdjRSNADataset, self).__init__(data_config, mode)
        if self.mode == "test":
            test_df = pd.read_pickle(
                "./intermediate_output/preprocessed_data/test_raw.pkl"
            )
            self.df = self.df.merge(test_df[["ID", "StudyInstanceUID"]], on="ID")
        self.df = self.df.sort_values(
            by=["StudyInstanceUID", "PositionOrd"]
        ).reset_index(drop=True)

    def __getitem__(self, idx):
        study_ids = self.df.loc[range(idx - 1, idx + 2), "StudyInstanceUID"]
        if len(study_ids.unique()) != 1:
            if study_ids.values[0] == study_ids.values[1]:
                batch_df = self.df.loc[
                    [idx - 2, idx - 1, idx], ["ID", "WindowCenter", "WindowWidth"]
                ]
            elif study_ids.values[1] == study_ids.values[2]:
                batch_df = self.df.loc[
                    [idx, idx + 1, idx + 2], ["ID", "WindowCenter", "WindowWidth"]
                ]
        else:
            batch_df = self.df.loc[
                [idx - 1, idx, idx + 1], ["ID", "WindowCenter", "WindowWidth"]
            ]

        if self.data_config["is_train"]:
            imgs = []
            for _, row in batch_df.iterrows():
                img_path = f"./input/stage_2_train_images/{row['ID']}.dcm"
                dicom = pydicom.dcmread(img_path)
                dicom = self._fix_pxrepr(dicom)
                img = dicom.pixel_array
                img = rescale_image(img, dicom.RescaleSlope, dicom.RescaleIntercept)
                img = self.apply_window_policy(
                    img, row["WindowCenter"], row["WindowWidth"]
                )
                imgs.append(img)
            img = np.concatenate(imgs, axis=-1)
            img = self._augmentation(img)(image=img)["image"]
            if self.data_config.use_smooth_label and self.mode == "train":
                target = self.df.loc[
                    idx, [f"{col}_smooth" for col in self.target_cols]
                ].values.astype(float)
            else:
                target = self.df.loc[idx, self.target_cols].values.astype(float)

            return img, target
        else:
            imgs = []
            for _, row in batch_df.iterrows():
                if self.mode == "valid":
                    img_path = f"./input/stage_2_train_images/{row['ID']}.dcm"
                else:
                    img_path = f"./input/stage_2_test_images/{row['ID']}.dcm"
                dicom = pydicom.dcmread(img_path)
                dicom = self._fix_pxrepr(dicom)
                img = dicom.pixel_array
                img = rescale_image(img, dicom.RescaleSlope, dicom.RescaleIntercept)
                img = self.apply_window_policy(
                    img, row["WindowCenter"], row["WindowWidth"]
                )
                imgs.append(img)
            img = np.concatenate(imgs, axis=-1)
            img = self._augmentation(img)(image=img)["image"]
            return img


def get_normal_dataset(data_config, mode="train"):
    dataset = RSNADataset(data_config, mode)
    return dataset


def get_smooth_dataset(data_config, mode="train"):
    dataset = SmoothRSNADataset(data_config, mode)
    return dataset


def get_without_any_smooth_dataset(data_config, mode="train"):
    dataset = WithoutAnySmoothRSNADataset(data_config, mode)
    return dataset


def get_adj_dataset(data_config, mode="train"):
    dataset = AdjRSNADataset(data_config, mode)
    return dataset


def get_dataset(dataset_name, **params):
    print("dataset name:", dataset_name)
    f = globals().get("get_" + dataset_name)
    return f(**params)
