"""
主训练脚本 - 5折交叉验证
"""

import os
import json
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    N_FOLDS,
    NUM_WORKERS,
    SEED,
    DEVICE,
    MODEL_NAME,
)
from dataset import OvarianCancerDataset, load_patient_data, load_data_splits
from transforms import get_train_transforms, get_val_transforms
from model import build_model
from trainer import Trainer
from utils import (
    compute_metrics,
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_cv_summary,
    save_metrics_csv,
    save_predictions,
    save_log,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    set_seed(SEED)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Ovarian Cancer CT Classification - 3D ResNet")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # 加载数据
    files, labels = load_patient_data()
    print(f"Total samples: {len(files)}")
    print(f"  Negative (class 0): {labels.count(0)}")
    print(f"  Positive (class 1): {labels.count(1)}")

    save_log(f"Total samples: {len(files)}", OUTPUT_DIR, "w")
    save_log(f"Negative: {labels.count(0)}, Positive: {labels.count(1)}", OUTPUT_DIR)

    fold_results = []

    for fold in range(N_FOLDS):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{N_FOLDS}")
        print(f"{'='*60}")

        fold_dir = os.path.join(OUTPUT_DIR, f"fold{fold+1}")
        os.makedirs(fold_dir, exist_ok=True)

        # 使用 StratifiedKFold 划分数据
        (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = \
            load_data_splits(fold, N_FOLDS, files, labels)

        print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        print(f"  Train: Neg={train_labels.count(0)}, Pos={train_labels.count(1)}")
        print(f"  Val:   Neg={val_labels.count(0)}, Pos={val_labels.count(1)}")
        print(f"  Test:  Neg={test_labels.count(0)}, Pos={test_labels.count(1)}")

        # 数据加载器
        train_ds = OvarianCancerDataset(train_files, train_labels, transform=get_train_transforms())
        val_ds = OvarianCancerDataset(val_files, val_labels, transform=get_val_transforms())
        test_ds = OvarianCancerDataset(test_files, test_labels, transform=get_val_transforms())

        train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        # 构建模型
        model = build_model(MODEL_NAME)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")

        # 训练
        trainer = Trainer(model, fold_dir)
        history = trainer.fit(train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE)

        # 加载最佳模型
        best_auc, best_epoch = trainer.load_best_model()
        print(f"Best model: epoch {best_epoch+1}, val AUC={best_auc:.4f}")

        # 测试集评估
        _, _, _, test_preds, test_labels_arr, test_probs = trainer.validate(test_loader)

        # 计算全部分类指标
        metrics = compute_metrics(test_labels_arr, test_preds, test_probs)
        print(f"Test Results:")
        for k, v in metrics.items():
            print(f"  {k:15s}: {v:.4f}")

        fold_results.append(metrics)

        # 可视化
        plot_training_curves(history, fold_dir, fold + 1)
        plot_confusion_matrix(test_labels_arr, test_preds, fold_dir, fold + 1)
        plot_roc_curve(test_labels_arr, test_probs, fold_dir, fold + 1)
        plot_pr_curve(test_labels_arr, test_probs, fold_dir, fold + 1)

        # 保存预测
        save_predictions(
            [os.path.basename(f) for f in test_files],
            test_labels_arr,
            test_preds,
            test_probs,
            fold_dir,
            fold + 1,
        )

        # 保存日志
        save_log(f"\nFold {fold+1} Test Results:", fold_dir)
        for k, v in metrics.items():
            save_log(f"  {k}: {v:.4f}", fold_dir)

    # 交叉验证汇总
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")

    save_log("\n=== CROSS-VALIDATION SUMMARY ===", OUTPUT_DIR)

    for metric in fold_results[0].keys():
        values = [r[metric] for r in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {metric:15s}: {mean_val:.4f} ± {std_val:.4f}")
        save_log(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}", OUTPUT_DIR)

    # 可视化汇总
    plot_cv_summary(fold_results, OUTPUT_DIR)
    save_metrics_csv(fold_results, OUTPUT_DIR)

    # 保存完整结果到JSON
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump({
            "fold_results": fold_results,
            "summary": {
                k: float(np.mean([r[k] for r in fold_results]))
                for k in fold_results[0].keys()
            },
        }, f, indent=2)

    print(f"\nAll results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
