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
    CenterSpatialCropd,
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
    SpatialPadd,
    EnsureTyped,
)

from config import (
    MIN_HU,
    MAX_HU,
    FINAL_SPATIAL_SIZE,
    FINAL_DEPTH,
    NUM_WORKERS,
)


def get_train_transforms():
    """训练数据增强 - 包含MONAI全套增强"""
    return Compose([
        LoadImaged(keys=["image"], image_only=True, ensure_channel_first=True),
        # CT窗口裁剪
        ScaleIntensityRanged(
            keys=["image"],
            a_min=MIN_HU,
            a_max=MAX_HU,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # 去除空气区域
        CropForegroundd(
            keys=["image"],
            source_key="image",
            margin=10,
            k_divisible=FINAL_SPATIAL_SIZE[0],
        ),
        # 空间归一化到指定尺寸
        Resized(
            keys=["image"],
            spatial_size=FINAL_SPATIAL_SIZE,
            mode="bilinear",
            anti_aliasing=True,
        ),
        # Z轴padding或裁剪到FINAL_DEPTH
        SpatialPadd(
            keys=["image"],
            spatial_size=(FINAL_SPATIAL_SIZE[0], FINAL_SPATIAL_SIZE[1], -1),
            mode="constant",
            constant_values=0.0,
        ),
        # MONAI数据增强
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=[0]),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=[1]),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=[2]),
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
            k_divisible=FINAL_SPATIAL_SIZE[0],
        ),
        Resized(
            keys=["image"],
            spatial_size=FINAL_SPATIAL_SIZE,
            mode="bilinear",
            anti_aliasing=True,
        ),
        SpatialPadd(
            keys=["image"],
            spatial_size=(FINAL_SPATIAL_SIZE[0], FINAL_SPATIAL_SIZE[1], -1),
            mode="constant",
            constant_values=0.0,
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        EnsureTyped(keys=["image"], dtype=np.float32),
    ])


def get_slice_sampler(volume, target_depth):
    """
    智能切片采样器。
    不是固定抽取N层，而是根据volume的depth动态采样。
    - 如果depth >= target_depth: 等间隔采样
    - 如果depth < target_depth: 居中padding

    Args:
        volume: numpy array, shape (C, H, W, D)
        target_depth: int, 目标深度

    Returns:
        sampled_volume: numpy array, shape (C, H, W, target_depth)
    """
    depth = volume.shape[-1]

    if depth >= target_depth:
        # 等间隔采样，覆盖整个depth范围
        indices = np.linspace(0, depth - 1, target_depth, dtype=int)
        return volume[..., indices]
    else:
        # Padding到target_depth（MONAI SpatialPadd已处理，这里做fallback）
        pad_total = target_depth - depth
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        return np.pad(
            volume,
            pad_width=((0, 0), (0, 0), (0, 0), (pad_before, pad_after)),
            mode="constant",
            constant_values=0.0,
        )
