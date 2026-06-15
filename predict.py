"""
预测脚本 - 支持单文件和批量预测
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import DEVICE, OUTPUT_DIR, BATCH_SIZE, NUM_WORKERS, MODEL_NAME
from model import build_model
from transforms import get_val_transforms
from dataset import OvarianCancerDataset


@torch.no_grad()
def predict_single(file_path, model_path):
    """
    预测单个 .nii 文件。

    Args:
        file_path: str, .nii文件路径
        model_path: str, 训练好的模型权重路径

    Returns:
        dict: {"class": 0 or 1, "probability_class0": float, "probability_class1": float}
    """
    # 加载模型
    model = build_model(MODEL_NAME)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # 预处理
    transform = get_val_transforms()
    data = transform({"image": file_path})
    image = data["image"]  # (C, H, W, D)
    image = torch.from_numpy(image).float().unsqueeze(0).to(DEVICE)  # (1, C, H, W, D)

    # 推理
    output = model(image)
    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    pred_class = int(np.argmax(probs))

    result = {
        "file": os.path.basename(file_path),
        "class": pred_class,
        "probability_class0": float(probs[0]),
        "probability_class1": float(probs[1]),
    }

    return result


@torch.no_grad()
def predict_folder(folder_path, model_path, output_csv=None):
    """
    批量预测文件夹中所有 .nii 文件。

    Args:
        folder_path: str, 包含 .nii 文件的文件夹路径
        model_path: str, 训练好的模型权重路径
        output_csv: str, 可选，输出CSV文件路径

    Returns:
        pd.DataFrame: 预测结果
    """
    # 收集 .nii 文件
    nii_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    ])

    if not nii_files:
        print(f"No .nii files found in {folder_path}")
        return None

    print(f"Found {len(nii_files)} files")

    # 临时标签（预测时不使用）
    temp_labels = [0] * len(nii_files)

    # 数据集
    dataset = OvarianCancerDataset(nii_files, temp_labels, transform=get_val_transforms())
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 加载模型
    model = build_model(MODEL_NAME)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # 推理
    all_probs = []
    for X, _ in loader:
        X = X.to(DEVICE)
        output = model(X)
        probs = torch.softmax(output, dim=1).cpu().numpy()
        all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = all_probs.argmax(axis=1)

    # 构建结果DataFrame
    results = pd.DataFrame({
        "file": [os.path.basename(f) for f in nii_files],
        "pred_class": all_preds,
        "prob_class0": all_probs[:, 0],
        "prob_class1": all_probs[:, 1],
    })

    if output_csv is not None:
        results.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Ovarian Cancer CT Prediction")
    parser.add_argument(
        "input",
        type=str,
        help="Input .nii file path or folder path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(OUTPUT_DIR, "fold1", "best_model.pth"),
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (for folder prediction)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Predict entire folder",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Please train the model first with train.py")
        return

    if args.batch or os.path.isdir(args.input):
        print(f"Batch prediction on: {args.input}")
        results = predict_folder(args.input, args.model, args.output)
        if results is not None:
            print(f"\nPredictions:")
            print(results.to_string())
    else:
        print(f"Single prediction on: {args.input}")
        result = predict_single(args.input, args.model)
        print(f"\nResult:")
        print(f"  File: {result['file']}")
        print(f"  Predicted class: {result['class']}")
        print(f"  Probability class 0 (negative): {result['probability_class0']:.4f}")
        print(f"  Probability class 1 (positive): {result['probability_class1']:.4f}")


if __name__ == "__main__":
    main()
