"""
配置模块 - 所有超参数和路径集中管理
"""

import os
from pathlib import Path

# ==========================================
# 路径配置
# ==========================================
DATA_DIR = "/kaggle/input/datasets/aaaxxxiii/luachao/卵巢癌数据"
LABEL_FILE = os.path.join(DATA_DIR, "patient_label.csv")
OUTPUT_DIR = "/kaggle/working/outputs"

# ==========================================
# 数据配置
# ==========================================
CT_WINDOW_CENTER = 40       # 腹部软组织窗位
CT_WINDOW_WIDTH = 400       # 腹部软组织窗宽
FINAL_DEPTH = 32            # 最终深度
FINAL_SPATIAL_SIZE = (128, 128, -1)  # H,W固定128，D保持原样
MIN_HU = CT_WINDOW_CENTER - CT_WINDOW_WIDTH // 2
MAX_HU = CT_WINDOW_CENTER + CT_WINDOW_WIDTH // 2

# ==========================================
# 模型配置
# ==========================================
MODEL_NAME = "resnet18"     # resnet18 / resnet34 / resnet50
IN_CHANNELS = 1             # 单通道CT
NUM_CLASSES = 2             # 二分类
USE_SE = True               # Squeeze-and-Excitation
DROPOUT_RATE = 0.4          # 分类器dropout
USE_GEM_POOL = True         # GeM Pooling
GEM_P = 3.0                 # GeM p值
USE_GROUPNORM = True        # 替换BatchNorm为GroupNorm
GROUPNORM_GROUPS = 8        # GroupNorm组数

# ==========================================
# 训练配置
# ==========================================
BATCH_SIZE = 4              # 3D模型显存占用大
NUM_EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.02
WARMUP_EPOCHS = 5
GRAD_CLIP = 1.0

# ==========================================
# 损失函数配置
# ==========================================
FOCAL_ALPHA = 0.25          # Focal Loss alpha
FOCAL_GAMMA = 2.0           # Focal Loss gamma
LABEL_SMOOTHING = 0.1       # 标签平滑

# ==========================================
# 混合精度 & EMA
# ==========================================
USE_AMP = True              # Automatic Mixed Precision
EMA_DECAY = 0.999           # EMA动量

# ==========================================
# 交叉验证
# ==========================================
N_FOLDS = 5
VAL_RATIO = 0.15            # 验证集比例

# ==========================================
# 其他
# ==========================================
SEED = 42
NUM_WORKERS = 2
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
