"""
工具模块 - 指标计算、可视化、日志
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)


def compute_metrics(y_true, y_pred, y_prob):
    """
    计算全部分类指标。
    返回字典包含：
    Accuracy, Precision, Recall, Specificity, Sensitivity(=Recall),
    F1, AUC, ConfusionMatrix
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    try:
        auc_val = roc_auc_score(y_true, y_prob[:, 1]) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auc_val = 0.5

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": auc_val,
    }

    return metrics


def plot_training_curves(history, save_dir, fold):
    """绘制训练曲线"""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Val", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # AUC
    axes[2].plot(epochs, history["val_auc"], linewidth=2, color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC")
    axes[2].set_title("Validation AUC")
    axes[2].grid(alpha=0.3)

    plt.suptitle(f"Fold {fold} - Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fold{fold}_training_curves.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_dir, fold):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Fold {fold} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fold{fold}_confusion_matrix.png"), dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_prob, save_dir, fold):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Fold {fold} - ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fold{fold}_roc_curve.png"), dpi=150)
    plt.close()


def plot_pr_curve(y_true, y_prob, save_dir, fold):
    """绘制PR曲线"""
    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, linewidth=2, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Fold {fold} - PR Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fold{fold}_pr_curve.png"), dpi=150)
    plt.close()


def plot_cv_summary(fold_results, save_dir):
    """绘制交叉验证汇总"""
    metrics_list = ["Accuracy", "Precision", "Recall", "Specificity", "Sensitivity", "F1", "AUC"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_list):
        if i >= len(axes):
            break
        values = [r[metric] for r in fold_results]
        axes[i].bar(range(1, len(values) + 1), values, alpha=0.7, color="steelblue")
        axes[i].axhline(np.mean(values), color="red", linestyle="--", linewidth=2,
                        label=f"Mean: {np.mean(values):.4f}")
        axes[i].set_title(f"{metric}\n{np.mean(values):.4f} ± {np.std(values):.4f}")
        axes[i].set_xlabel("Fold")
        axes[i].set_ylim(0, 1.05)
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)

    # 隐藏多余的子图
    for i in range(len(metrics_list), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("5-Fold Cross-Validation Summary", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cv_summary.png"), dpi=150)
    plt.close()


def save_metrics_csv(fold_results, save_dir):
    """保存指标到CSV"""
    rows = []
    for i, result in enumerate(fold_results):
        row = {"fold": f"fold_{i+1}"}
        row.update(result)
        rows.append(row)

    # 添加均值行
    mean_row = {"fold": "mean"}
    for key in fold_results[0].keys():
        values = [r[key] for r in fold_results]
        mean_row[key] = np.mean(values)
    rows.append(mean_row)

    # 添加标准差行
    std_row = {"fold": "std"}
    for key in fold_results[0].keys():
        values = [r[key] for r in fold_results]
        std_row[key] = np.std(values)
    rows.append(std_row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)
    return df


def save_predictions(file_names, y_true, y_pred, y_prob, save_dir, fold):
    """保存预测结果"""
    df = pd.DataFrame({
        "file": file_names,
        "true_label": y_true,
        "pred_label": y_pred,
        "prob_class0": y_prob[:, 0],
        "prob_class1": y_prob[:, 1],
    })
    df.to_csv(
        os.path.join(save_dir, f"fold{fold}_predictions.csv"),
        index=False,
    )


def save_log(message, save_dir, mode="a"):
    """保存训练日志"""
    with open(os.path.join(save_dir, "training_log.txt"), mode, encoding="utf-8") as f:
        f.write(message + "\n")
