"""
数据预处理与增强模块 - 基于MONAI
"""

import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    NormalizeIntensityd,
    EnsureTyped,
)

from config import MIN_HU, MAX_HU, FINAL_SPATIAL_SIZE, NUM_WORKERS


def get_train_transforms():
    """训练数据增强"""
    return Compose([
        LoadImaged(keys=["image"], image_only=True, ensure_channel_first=True),
        # CT窗宽窗位裁剪
        ScaleIntensityRanged(
            keys=["image"],
            a_min=MIN_HU,
            a_max=MAX_HU,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # 去空气区域，保持尺寸可被16整除
        CropForegroundd(
            keys=["image"],
            source_key="image",
            margin=10,
            k_divisible=16,
        ),
        # 统一缩放到 (H=128, W=128, D=32)
        Resized(
            keys=["image"],
            spatial_size=(128, 128, 32),
            mode="trilinear",
        ),
        # MONAI数据增强
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=[0]),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=[1]),
        RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 1)),
        RandAffined(
            keys=["image"],
            prob=0.5,
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1),
            mode="bilinear",
        ),
        RandZoomd(
            keys=["image"],
            prob=0.3,
            min_zoom=0.9,
            max_zoom=1.1,
            mode="trilinear",
        ),
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
        RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.8, 1.2)),
        RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0)),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        EnsureTyped(keys=["image"], dtype=np.float32),
    ])


def get_val_transforms():
    """验证/测试预处理 - 无随机增强"""
    return Compose([
        LoadImaged(keys=["image"], image_only=True, ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=MIN_HU,
            a_max=MAX_HU,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(
            keys=["image"],
            source_key="image",
            margin=10,
            k_divisible=16,
        ),
        Resized(
            keys=["image"],
            spatial_size=(128, 128, 32),
            mode="trilinear",
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        EnsureTyped(keys=["image"], dtype=np.float32),
    ])
