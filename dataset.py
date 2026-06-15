"""
数据集模块 - 加载卵巢癌CT数据
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import DATA_DIR, LABEL_FILE, FINAL_DEPTH


class OvarianCancerDataset(Dataset):
    """
    卵巢癌CT二分类数据集。
    支持 .nii 文件，不同病人层数自动处理。
    """

    def __init__(self, file_list, labels, transform=None):
        """
        Args:
            file_list: list of .nii file paths
            labels: list of int labels (0 or 1)
            transform: MONAI Compose transform (带 'image' key的字典变换)
        """
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

        assert len(self.file_list) == len(self.labels), \
            f"文件数({len(self.file_list)})和标签数({len(self.labels)})不匹配"

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]

        if self.transform is not None:
            data = self.transform({"image": file_path})
            image = data["image"]  # shape: (C, H, W, D)
        else:
            import nibabel as nib
            from monai.transforms import EnsureChannelFirst
            vol = nib.load(file_path).get_fdata().astype(np.float32)
            image = np.expand_dims(vol, 0)  # (1, H, W, D)

        image = torch.from_numpy(image).float()
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def load_data_splits(fold, n_folds, train_files, train_labels):
    """
    使用 StratifiedKFold 划分数据，确保类别平衡。

    Args:
        fold: int, 当前fold
        n_folds: int, 总fold数
        train_files: list of file paths
        train_labels: list of labels

    Returns:
        (train_sub_files, train_sub_labels),
        (val_files, val_labels),
        (test_files, test_labels)
    """
    from sklearn.model_selection import StratifiedKFold

    files = np.array(train_files)
    labels = np.array(train_labels)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(skf.split(files, labels))

    test_idx = splits[fold][1]
    train_val_idx = splits[fold][0]

    test_files = files[test_idx].tolist()
    test_labels = labels[test_idx].tolist()

    train_val_files = files[train_val_idx].tolist()
    train_val_labels = labels[train_val_idx].tolist()

    # 从 train_val 中分出验证集
    from sklearn.model_selection import train_test_split
    train_sub_files, val_files, train_sub_labels, val_labels = train_test_split(
        train_val_files,
        train_val_labels,
        test_size=0.176,  # 0.176 * 0.8 ≈ 0.14 → 接近15%验证集
        stratify=train_val_labels,
        random_state=42,
    )

    return (
        (train_sub_files, train_sub_labels),
        (val_files, val_labels),
        (test_files, test_labels),
    )


def load_patient_data():
    """
    加载所有患者数据和标签。

    Returns:
        files: list of .nii file paths
        labels: list of int (0 or 1)
    """
    df = pd.read_csv(LABEL_FILE)
    df["id"] = df["id"].astype(str)

    files = sorted([
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".nii")
    ])

    labels = []
    valid_files = []
    for f in files:
        pid = os.path.basename(f).replace(".nii", "")
        if pid in df["id"].values:
            original_label = df[df["id"] == pid]["label"].values[0]
            # 1 -> 0, 2 -> 1
            converted_label = int(original_label) - 1
            labels.append(converted_label)
            valid_files.append(f)

    print(f"Loaded {len(valid_files)} valid samples")
    print(f"  Class 0: {labels.count(0)}, Class 1: {labels.count(1)}")

    return valid_files, labels
