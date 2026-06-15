# Ovarian Cancer CT Classification

基于 PyTorch + MONAI 的卵巢癌 CT 二分类深度学习系统。

## 项目概述

- **任务**: 卵巢癌 CT 二分类 (Positive / Negative)
- **数据**: 150 例 .nii 格式 CT (75 Positive / 75 Negative)
- **模型**: 3D ResNet18 + SE Block + GeM Pooling + GroupNorm
- **框架**: PyTorch + MONAI
- **环境**: Kaggle Notebook (GPU T4 × 2)

## 核心特性

- CT 标准预处理 (HU Windowing / Body Crop / 中心裁剪)
- 智能切片采样 (自适应 depth，不固定层数)
- MONAI 全套数据增强 (RandFlip / RandAffine / RandZoom / RandNoise ...)
- 3D ResNet + Squeeze-and-Excitation + GeM Pooling
- GroupNorm 替代 BatchNorm (小 batch 友好)
- Focal Loss + Label Smoothing
- AdamW + Warmup + CosineAnnealingLR
- Automatic Mixed Precision (AMP)
- EMA (Exponential Moving Average)
- 5-Fold Stratified 交叉验证
- EarlyStopping (基于 AUC)
- 完整指标 (Accuracy / Precision / Recall / Sensitivity / Specificity / F1 / AUC)
- 自动保存模型、ROC/PR曲线、混淆矩阵、训练日志

## 目录结构

```
ovarian-cancer-classification/
├── README.md              # 本文件
├── requirements.txt        # Python依赖
├── config.py               # 集中配置
├── dataset.py              # 数据集加载与数据划分
├── transforms.py           # MONAI 预处理与增强
├── model.py                # 3D ResNet + SE + GeM + GroupNorm
├── losses.py               # Focal Loss + Label Smoothing
├── trainer.py              # 训练器 (AMP / EMA / Warmup / EarlyStopping)
├── utils.py                # 指标计算 / 可视化 / 日志
├── train.py                # 主训练脚本 (5-Fold CV)
└── predict.py              # 预测脚本 (单文件 / 批量)
```

## 安装

```bash
pip install -r requirements.txt
```

### 依赖

```
torch>=2.0.0
torchvision>=0.15.0
monai>=1.3.0
nibabel>=5.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
pandas>=2.0.0
einops>=0.7.0
tqdm>=4.65.0
opencv-python>=4.8.0
seaborn>=0.12.0
```

## Kaggle 使用方法

### 1. 上传代码

将项目文件上传到 Kaggle Dataset 或直接放在 Notebook 同级目录。

### 2. 数据格式

```
/kaggle/input/datasets/aaaxxxiii/luachao/卵巢癌数据/
├── OC001.nii
├── OC002.nii
├── ...
├── OC150.nii
└── patient_label.csv
```

`patient_label.csv` 格式:
```
id,label
OC001,1
OC002,2
...
```
(label: 1 → class 0, 2 → class 1)

### 3. 训练

在 Kaggle Notebook 中运行:

```python
%run train.py
```

或:

```python
!python train.py
```

### 4. 预测

单文件预测:
```bash
python predict.py /path/to/image.nii --model outputs/fold1/best_model.pth
```

批量预测:
```bash
python predict.py /path/to/folder --batch --model outputs/fold1/best_model.pth --output predictions.csv
```

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| LEARNING_RATE | 3e-4 | 初始学习率 |
| BATCH_SIZE | 4 | 3D模型显存限制 |
| NUM_EPOCHS | 100 | 最大训练轮数 |
| WARMUP_EPOCHS | 5 | Warmup轮数 |
| FOCAL_GAMMA | 2.0 | Focal Loss gamma |
| LABEL_SMOOTHING | 0.1 | 标签平滑 |
| DROPOUT_RATE | 0.4 | 分类器Dropout |
| WEIGHT_DECAY | 0.02 | L2正则化 |
| EMA_DECAY | 0.999 | EMA动量 |
| N_FOLDS | 5 | 交叉验证折数 |

## 模型架构

```
Input: (B, 1, 32, 128, 128)
  │
  ├─ Conv3d(1→64, kernel=7, stride=2) + GroupNorm + GELU
  ├─ MaxPool3d(kernel=3, stride=2)
  │
  ├─ Layer1: BasicBlock3D × 2 (64→64) + SE
  ├─ Layer2: BasicBlock3D × 2 (64→128, stride=2) + SE
  ├─ Layer3: BasicBlock3D × 2 (128→256, stride=2) + SE
  ├─ Layer4: BasicBlock3D × 2 (256→512, stride=2) + SE
  │
  ├─ GeM Pooling (p=3.0, learnable)
  ├─ Dropout(0.4)
  └─ Linear(512 → 2)
```

## 输出

训练完成后，`/kaggle/working/outputs/` 目录包含:

```
outputs/
├── fold1/
│   ├── best_model.pth            # 最佳模型权重 (按AUC)
│   ├── fold1_training_curves.png # 训练曲线
│   ├── fold1_confusion_matrix.png # 混淆矩阵
│   ├── fold1_roc_curve.png       # ROC曲线
│   ├── fold1_pr_curve.png        # PR曲线
│   ├── fold1_predictions.csv     # 测试集预测结果
│   └── training_log.txt          # 训练日志
├── fold2/
├── fold3/
├── fold4/
├── fold5/
├── cv_summary.png                # 5折CV汇总图
├── metrics.csv                   # 全部指标CSV
├── results.json                  # JSON格式结果
└── training_log.txt              # 汇总训练日志
```

## 配置修改

所有超参数集中在 `config.py`，直接修改即可:

```python
# 修改学习率
LEARNING_RATE = 1e-4

# 修改模型
MODEL_NAME = "resnet34"

# 关闭Focal Loss改用CrossEntropy
FOCAL_GAMMA = 0
LABEL_SMOOTHING = 0
```

## License

MIT
