"""
Ovarian Cancer CT Classification - Lightweight 3D CNN
RTX 4060 8GB - Small model for 149 samples
"""
import os, sys, json, random, copy, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityRanged, CropForegroundd, Resized,
    RandFlipd, RandRotate90d, RandAffined, RandZoomd, RandGaussianNoised,
    RandAdjustContrastd, RandGaussianSmoothd, RandScaleIntensityd,
    RandShiftIntensityd, NormalizeIntensityd, EnsureTyped,
)

# ==========================================
# Config
# ==========================================
DATA_DIR = r"E:\download\卵巢癌数据"
OUTPUT_DIR = r"E:\download\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_HU = -160
MAX_HU = 240
FINAL_SIZE = (80, 80, 32)
BATCH_SIZE = 8
NUM_EPOCHS = 80
LR = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
N_FOLDS = 5
SEED = 42
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE} | torch {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# ==========================================
# Transforms
# ==========================================
def make_transforms(augment):
    base = [
        LoadImaged(keys=["image"], image_only=True, ensure_channel_first=True),
        ScaleIntensityRanged(keys=["image"], a_min=MIN_HU, a_max=MAX_HU, b_min=0, b_max=1, clip=True),
        CropForegroundd(keys=["image"], source_key="image", margin=10, k_divisible=16),
        Resized(keys=["image"], spatial_size=FINAL_SIZE, mode="trilinear"),
    ]
    if augment:
        base += [
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=[0]),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=[1]),
            RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 1)),
            RandAffined(keys=["image"], prob=0.5, rotate_range=(0.15,0.15,0.15), scale_range=(0.15,0.15,0.15), mode="bilinear"),
            RandZoomd(keys=["image"], prob=0.3, min_zoom=0.85, max_zoom=1.15, mode="trilinear"),
            RandGaussianNoised(keys=["image"], prob=0.3, std=0.05),
            RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.0)),
            RandScaleIntensityd(keys=["image"], factors=0.15, prob=0.3),
            RandShiftIntensityd(keys=["image"], offsets=0.15, prob=0.3),
        ]
    base += [NormalizeIntensityd(keys=["image"], nonzero=True), EnsureTyped(keys=["image"], dtype=np.float32)]
    return Compose(base)

# ==========================================
# Dataset
# ==========================================
class CTDataset(Dataset):
    def __init__(self, file_list, label_list, augment=False):
        self.file_list = file_list
        self.label_list = label_list
        self.transform = make_transforms(augment)
    def __len__(self): return len(self.file_list)
    def __getitem__(self, idx):
        data = self.transform({"image": self.file_list[idx]})
        img = data["image"]
        if hasattr(img, "as_tensor"): img = img.as_tensor().float()
        elif isinstance(img, np.ndarray): img = torch.from_numpy(img).float()
        else: img = img.float()
        return img, torch.tensor(self.label_list[idx], dtype=torch.long)

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, "patient_label.csv"))
df["id"] = df["id"].astype(str)
labels_map = dict(zip(df["id"], df["label"]))
all_nii = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".nii")])
files, labels = [], []
for fp in all_nii:
    pid = os.path.basename(fp).replace(".nii", "")
    if pid not in labels_map or os.path.getsize(fp) == 0: continue
    try: nib.load(fp).get_fdata()
    except: continue
    files.append(fp); labels.append(int(labels_map[pid]) - 1)
print(f"Samples: {len(files)} (Neg={labels.count(0)} Pos={labels.count(1)})")

# ==========================================
# Lightweight 3D CNN (~300K params)
# ==========================================
class Light3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))

n_params = sum(p.numel() for p in Light3DCNN().parameters())
print(f"Model params: {n_params:,}")

# ==========================================
# EMA
# ==========================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model; self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad: self.shadow[n] = self.decay * self.shadow[n] + (1-self.decay) * p.data
    def apply(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad: self.backup[n] = p.data.clone(); p.data = self.shadow[n]
    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad: p.data = self.backup[n]

# ==========================================
# Trainer
# ==========================================
class Trainer:
    def __init__(self, model, fold_dir):
        self.model = model.to(DEVICE)
        self.fold_dir = fold_dir
        os.makedirs(fold_dir, exist_ok=True)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
        self.scaler = GradScaler()
        self.ema = EMA(model)
        self.best_auc = 0.0
        self.patience = 0

    def run_epoch(self, loader, train):
        self.model.train() if train else self.model.eval()
        total_loss, preds, lbls, all_probs = 0, [], [], []
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                with autocast():
                    out = self.model(X)
                    loss = self.criterion(out, y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.ema.update()
            else:
                self.ema.apply()
                out = self.model(X)
                loss = self.criterion(out, y)
                self.ema.restore()
            total_loss += loss.item()
            prob = torch.softmax(out.detach(), 1)
            preds.extend(prob.argmax(1).cpu().numpy())
            lbls.extend(y.cpu().numpy())
            all_probs.append(prob.cpu().numpy())
        all_probs = np.concatenate(all_probs)
        auc = roc_auc_score(lbls, all_probs[:, 1]) if len(np.unique(lbls)) > 1 else 0.5
        return total_loss / len(loader), accuracy_score(lbls, preds), auc, preds, lbls, all_probs

    def fit(self, tr_loader, va_loader, epochs):
        for ep in range(epochs):
            tl, ta, _, _, _, _ = self.run_epoch(tr_loader, True)
            vl, va, vau, _, _, _ = self.run_epoch(va_loader, False)
            self.scheduler.step()
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"  Ep{ep+1:3d} | Tr L={tl:.4f} A={ta:.4f} | Va L={vl:.4f} A={va:.4f} AUC={vau:.4f}")
            if vau > self.best_auc + 0.001:
                self.best_auc = vau; self.patience = 0
                self.ema.apply(); torch.save(self.model.state_dict(), f"{self.fold_dir}/best.pth"); self.ema.restore()
            else:
                self.patience += 1
                if self.patience >= 20: print(f"  EarlyStop @ ep{ep+1}"); break

    def load_best(self):
        self.model.load_state_dict(torch.load(f"{self.fold_dir}/best.pth", map_location=DEVICE))

# ==========================================
# Train
# ==========================================
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print("=" * 60)
print(f"Light 3D CNN | {n_params:,} params | Size={FINAL_SIZE} | Batch={BATCH_SIZE}")
print("=" * 60)

fold_results = []

for fold in range(N_FOLDS):
    print(f"\nFold {fold+1}/{N_FOLDS}")
    fold_dir = f"{OUTPUT_DIR}/fold{fold+1}"
    os.makedirs(fold_dir, exist_ok=True)

    # Stratified splits
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(files, labels))
    te_f = [files[i] for i in splits[fold][1]]; te_l = [labels[i] for i in splits[fold][1]]
    tv_f = [files[i] for i in splits[fold][0]]; tv_l = [labels[i] for i in splits[fold][0]]

    tr_f, va_f, tr_l, va_l = train_test_split(tv_f, tv_l, test_size=0.18, stratify=tv_l, random_state=SEED)
    print(f"Train={len(tr_f)} Val={len(va_f)} Test={len(te_f)}")

    tr_dl = DataLoader(CTDataset(tr_f, tr_l, True), BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    va_dl = DataLoader(CTDataset(va_f, va_l, False), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    te_dl = DataLoader(CTDataset(te_f, te_l, False), BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    trainer = Trainer(Light3DCNN(), fold_dir)
    trainer.fit(tr_dl, va_dl, NUM_EPOCHS)
    trainer.load_best()

    _, te_acc, te_auc, te_preds, te_lbls, te_probs = trainer.run_epoch(te_dl, False)
    print(f"  >> TEST: Acc={te_acc:.4f} AUC={te_auc:.4f}")
    fold_results.append({"acc": te_acc, "auc": te_auc})

    pd.DataFrame({
        "file": [os.path.basename(f) for f in te_f],
        "true": te_lbls, "pred": te_preds,
        "prob0": te_probs[:, 0], "prob1": te_probs[:, 1],
    }).to_csv(f"{fold_dir}/predictions.csv", index=False)

print(f"\n{'='*60}")
accs = [r["acc"] for r in fold_results]
aucs = [r["auc"] for r in fold_results]
print(f"5-Fold | Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f} | AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"Results: {OUTPUT_DIR}")
