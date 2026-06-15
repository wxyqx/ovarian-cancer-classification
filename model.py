"""
模型模块 - 3D ResNet + SE + GeM Pooling + GroupNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from config import (
    IN_CHANNELS,
    NUM_CLASSES,
    USE_SE,
    DROPOUT_RATE,
    USE_GEM_POOL,
    GEM_P,
    USE_GROUPNORM,
    GROUPNORM_GROUPS,
)


# ==========================================
# GroupNorm 转换工具
# ==========================================
def replace_bn_with_gn(module, num_groups=8):
    """
    递归将模型中所有 BatchNorm 替换为 GroupNorm。
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm3d):
            num_channels = child.num_features
            setattr(module, name, nn.GroupNorm(
                num_groups=min(num_groups, num_channels),
                num_channels=num_channels,
            ))
        elif isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            setattr(module, name, nn.GroupNorm(
                num_groups=min(num_groups, num_channels),
                num_channels=num_channels,
            ))
        elif isinstance(child, nn.BatchNorm1d):
            num_channels = child.num_features
            setattr(module, name, nn.GroupNorm(
                num_groups=min(num_groups, num_channels),
                num_channels=num_channels,
            ))
        else:
            replace_bn_with_gn(child, num_groups)


# ==========================================
# Squeeze-and-Excitation Block
# ==========================================
class SEBlock3D(nn.Module):
    """3D Squeeze-and-Excitation模块"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv3d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv3d(channels // reduction, channels, 1)

    def forward(self, x):
        scale = F.adaptive_avg_pool3d(x, 1)
        scale = F.gelu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


# ==========================================
# GeM Pooling
# ==========================================
class GeMPool3D(nn.Module):
    """
    Generalized Mean Pooling (3D版本)。
    可学习的p参数自动在avg pooling(p=1)和max pooling(p→∞)之间切换。
    """

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, C, D, H, W)
        return F.avg_pool3d(
            x.clamp(min=self.eps).pow(self.p),
            kernel_size=(x.size(2), x.size(3), x.size(4)),
        ).pow(1.0 / self.p)


# ==========================================
# 3D ResNet Backbone (来自MONAI或自定义)
# ==========================================
class BasicBlock3D(nn.Module):
    """3D Basic Block for ResNet"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(GROUPNORM_GROUPS, planes), planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(GROUPNORM_GROUPS, planes), planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(GROUPNORM_GROUPS, planes), planes),
            )

        self.se = SEBlock3D(planes) if USE_SE else nn.Identity()

    def forward(self, x):
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.gelu(out)
        out = self.se(out)
        return out


class ResNet3D(nn.Module):
    """
    自定义3D ResNet。
    支持 ResNet18/34/50 配置。
    使用 GroupNorm 替代 BatchNorm。
    集成 SE Block 和 GeM Pooling。
    """

    def __init__(self, block, num_blocks, in_channels=1, num_classes=2):
        super().__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=(2, 2, 2), padding=3, bias=False)
        self.gn1 = nn.GroupNorm(min(GROUPNORM_GROUPS, 64), 64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        if USE_GEM_POOL:
            self.pool = GeMPool3D(p=GEM_P)
        else:
            self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.fc = nn.Linear(512 * block.expansion, NUM_CLASSES)

        self._init_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = F.gelu(self.gn1(self.conv1(x)))
        x = F.max_pool3d(x, kernel_size=3, stride=2, padding=1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def build_resnet18():
    """构建 ResNet18-3D"""
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)


def build_resnet34():
    """构建 ResNet34-3D"""
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3], in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)


def build_model(model_name="resnet18"):
    """构建指定名称的模型"""
    builders = {
        "resnet18": build_resnet18,
        "resnet34": build_resnet34,
    }
    if model_name not in builders:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(builders.keys())}")

    model = builders[model_name]()

    if USE_GROUPNORM:
        # 额外确保所有BN都被替换（包括自定义的BasicBlock3D）
        replace_bn_with_gn(model, GROUPNORM_GROUPS)

    return model
