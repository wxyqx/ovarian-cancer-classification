"""
损失函数模块 - Focal Loss + Label Smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FOCAL_ALPHA, FOCAL_GAMMA, LABEL_SMOOTHING


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    当 γ=0 时退化为 CrossEntropy。
    当 γ>0 时减少易分类样本的权重，专注于难分类样本。
    """

    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)

        # 计算alpha权重
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            alpha_t = alpha_t.to(inputs.device)
        else:
            alpha_t = 1.0

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class FocalLossWithLabelSmoothing(nn.Module):
    """
    Focal Loss + Label Smoothing 组合。
    结合了两者优点：
    - Label Smoothing 防止过拟合
    - Focal Loss 处理样本难易程度
    """

    def __init__(
        self,
        alpha=FOCAL_ALPHA,
        gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
        num_classes=2,
        reduction="mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Label Smoothing
        log_probs = F.log_softmax(inputs, dim=-1)
        targets_onehot = torch.zeros_like(log_probs)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1.0)
        targets_smooth = targets_onehot * (1 - self.label_smoothing) + \
                         self.label_smoothing / self.num_classes

        # Focal mechanism on smoothed labels
        probs = torch.exp(log_probs)
        pt = (targets_smooth * probs).sum(dim=-1)

        # Alpha weighting
        if self.alpha is not None:
            alpha_t = self.alpha * targets_smooth[:, 1] + (1 - self.alpha) * targets_smooth[:, 0]
        else:
            alpha_t = 1.0

        focal_weight = alpha_t * (1 - pt) ** self.gamma

        # Cross entropy with label smoothing
        ce_loss = -(targets_smooth * log_probs).sum(dim=-1)
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss():
    """创建损失函数"""
    return FocalLossWithLabelSmoothing(
        alpha=FOCAL_ALPHA,
        gamma=FOCAL_GAMMA,
        label_smoothing=LABEL_SMOOTHING,
        num_classes=2,
    )
