# Ovarian Cancer CT Classification

卵巢癌 CT 二分类 | PyTorch + MONAI | Lightweight 3D CNN (434K params)

## 快速开始

### Kaggle

```python
!pip install monai -q
%run train.py
```

### 本地

```bash
pip install monai nibabel pandas scikit-learn
python train_local.py
```

修改 `train_local.py` 第28行 `DATA_DIR` 为你的数据路径。

## 数据格式

```
数据目录/
├── OC001.nii ... OC150.nii
└── patient_label.csv  (列: id, label; label 1→class0, 2→class1)
```

## 配置

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 80×80×32 |
| 模型 | Light 3D CNN, 434K params |
| Batch | 8 |
| CT窗 | [-160, 240] HU |
| Epochs | 80 (早停) |
| 交叉验证 | 5-Fold Stratified |

## 输出

```
outputs/
├── fold1/best.pth, predictions.csv
├── fold2/...
└── fold5/...
```
