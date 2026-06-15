"""
训练器模块 - AMP, EMA, Warmup, EarlyStopping
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from config import (
    DEVICE,
    USE_AMP,
    EMA_DECAY,
    WARMUP_EPOCHS,
    GRAD_CLIP,
)
from losses import build_loss


class EMA:
    """Exponential Moving Average - 模型参数平滑"""

    def __init__(self, model, decay=EMA_DECAY):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


class Trainer:
    """
    完整训练器。
    - AMP 混合精度训练
    - EMA 参数平滑
    - Warmup + CosineAnnealingLR
    - EarlyStopping (基于AUC)
    - Gradient Clipping
    """

    def __init__(self, model, save_dir):
        self.model = model.to(DEVICE)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.criterion = build_loss()

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0,  # 从0开始，Warmup时会逐步增加
            weight_decay=0.02,
        )

        self.scaler = GradScaler() if USE_AMP else None
        self.ema = EMA(model, EMA_DECAY) if EMA_DECAY > 0 else None
        if self.ema is not None:
            self.ema.register()

        self.best_auc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.max_patience = 25

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_auc": [],
            "lr": [],
        }

    def _warmup_lr(self, epoch, warmup_epochs, base_lr):
        """Warmup学习率调度"""
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        return base_lr

    def train_epoch(self, loader, epoch, total_epochs, base_lr):
        """训练一个epoch"""
        self.model.train()

        # Warmup LR
        warmup_lr = self._warmup_lr(epoch, WARMUP_EPOCHS, base_lr)
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = warmup_lr

        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_idx, (X, y) in enumerate(loader):
            X, y = X.to(DEVICE), y.to(DEVICE)

            self.optimizer.zero_grad(set_to_none=True)

            if USE_AMP:
                with autocast():
                    outputs = self.model(X)
                    loss = self.criterion(outputs, y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                self.optimizer.step()

            if self.ema is not None:
                self.ema.update()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(all_labels, all_preds)

        return total_loss / len(loader), accuracy

    @torch.no_grad()
    def validate(self, loader):
        """验证"""
        if self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            outputs = self.model(X)
            loss = self.criterion(outputs, y)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        if self.ema is not None:
            self.ema.restore()

        from sklearn.metrics import accuracy_score, roc_auc_score
        accuracy = accuracy_score(all_labels, all_preds)
        all_probs = np.array(all_probs)

        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1]) if len(np.unique(all_labels)) > 1 else 0.5
        except Exception:
            auc = 0.5

        return total_loss / len(loader), accuracy, auc, all_preds, all_labels, all_probs

    def fit(self, train_loader, val_loader, epochs, base_lr):
        """完整训练流程"""
        import torch.optim.lr_scheduler as lr_scheduler
        scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs - WARMUP_EPOCHS,
            eta_min=1e-6,
        )

        for epoch in range(epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch, epochs, base_lr)

            # 验证
            val_loss, val_acc, val_auc, _, _, _ = self.validate(val_loader)

            # Warmup后使用CosineAnnealing
            if epoch >= WARMUP_EPOCHS:
                scheduler.step()

            # 记录当前LR
            current_lr = self.optimizer.param_groups[0]["lr"]

            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_auc"].append(val_auc)
            self.history["lr"].append(current_lr)

            # 日志
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch+1:3d}/{epochs} | "
                    f"LR={current_lr:.1e} | "
                    f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                    f"Val: Loss={val_loss:.4f} Acc={val_acc:.4f} AUC={val_auc:.4f}"
                )

            # EarlyStopping - 基于AUC (不是Accuracy)
            if val_auc > self.best_auc + 0.001:
                self.best_auc = val_auc
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_best_model(val_auc)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.max_patience:
                    print(f"  EarlyStopping at epoch {epoch+1}, best AUC={self.best_auc:.4f}")
                    break

        return self.history

    def _save_best_model(self, auc):
        """保存最佳模型（基于AUC）"""
        model_state = self.model.state_dict()
        if self.ema is not None:
            self.ema.apply_shadow()
            model_state = copy.deepcopy(self.model.state_dict())
            self.ema.restore()

        torch.save(
            {
                "model_state_dict": model_state,
                "auc": auc,
                "best_epoch": self.best_epoch,
            },
            os.path.join(self.save_dir, "best_model.pth"),
        )

    def load_best_model(self):
        """加载最佳模型"""
        checkpoint = torch.load(
            os.path.join(self.save_dir, "best_model.pth"),
            map_location=DEVICE,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint["auc"], checkpoint["best_epoch"]
