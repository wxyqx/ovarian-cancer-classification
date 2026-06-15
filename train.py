import os, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import nibabel as nib
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityRanged, CropForegroundd, Resized,
    RandFlipd, RandRotate90d, RandGaussianNoised,
    NormalizeIntensityd, EnsureTyped,
)

DATA_DIR = "/kaggle/input/datasets/aaaxxxiii/luachao/卵巢癌数据"
OUTPUT_DIR = "/kaggle/working/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_HU, MAX_HU = -160, 240
FINAL_SIZE = (96, 96, 32)
BATCH_SIZE = 4
NUM_EPOCHS = 100
LR = 1e-4
N_FOLDS = 5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

def make_transforms(aug):
    base = [
        LoadImaged(keys=["image"], image_only=True, ensure_channel_first=True),
        ScaleIntensityRanged(keys=["image"], a_min=MIN_HU, a_max=MAX_HU, b_min=0, b_max=1, clip=True),
        CropForegroundd(keys=["image"], source_key="image", margin=10, k_divisible=16),
        Resized(keys=["image"], spatial_size=FINAL_SIZE, mode="trilinear"),
    ]
    if aug:
        base += [
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=[0]),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=[1]),
            RandRotate90d(keys=["image"], prob=0.3, spatial_axes=(0, 1)),
            RandGaussianNoised(keys=["image"], prob=0.2, std=0.02),
        ]
    base += [NormalizeIntensityd(keys=["image"], nonzero=True), EnsureTyped(keys=["image"], dtype=np.float32)]
    return Compose(base)

class CTDataset(Dataset):
    def __init__(self, flist, labs, aug=False):
        self.flist, self.labs = flist, labs
        self.tf = make_transforms(aug)
    def __len__(self): return len(self.flist)
    def __getitem__(self, i):
        d = self.tf({"image": self.flist[i]})
        img = d["image"]
        if hasattr(img, "as_tensor"): img = img.as_tensor().float()
        elif isinstance(img, np.ndarray): img = torch.from_numpy(img).float()
        else: img = img.float()
        return img, torch.tensor(self.labs[i], dtype=torch.long)

df = pd.read_csv(os.path.join(DATA_DIR, "patient_label.csv"))
df["id"] = df["id"].astype(str)
lmap = dict(zip(df["id"], df["label"]))
nii_files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".nii")])
files, labels = [], []
for fp in nii_files:
    pid = os.path.basename(fp).replace(".nii", "")
    if pid not in lmap or os.path.getsize(fp) == 0: continue
    try: nib.load(fp).get_fdata()
    except: continue
    files.append(fp); labels.append(int(lmap[pid]) - 1)
print(f"Samples: {len(files)} (Neg={labels.count(0)} Pos={labels.count(1)})")

class ResBlock3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv3d(c, c, 3, padding=1); self.g1 = nn.GroupNorm(min(8,c), c)
        self.c2 = nn.Conv3d(c, c, 3, padding=1); self.g2 = nn.GroupNorm(min(8,c), c)
    def forward(self, x): return F.relu(self.g2(self.c2(F.relu(self.g1(self.c1(x)))))) + x

class Res3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.ReLU())
        self.s1 = nn.Sequential(ResBlock3D(32), ResBlock3D(32))
        self.d1 = nn.Sequential(nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.ReLU())
        self.s2 = nn.Sequential(ResBlock3D(64), ResBlock3D(64))
        self.d2 = nn.Sequential(nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.ReLU())
        self.s3 = nn.Sequential(ResBlock3D(128), ResBlock3D(128))
        self.d3 = nn.Sequential(nn.Conv3d(128, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.ReLU())
        self.s4 = nn.Sequential(ResBlock3D(128), ResBlock3D(128))
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(128, 2))
    def forward(self, x):
        x = self.stem(x)
        x = self.s1(x); x = self.d1(x)
        x = self.s2(x); x = self.d2(x)
        x = self.s3(x); x = self.d3(x)
        x = self.s4(x)
        return self.head(self.pool(x).view(x.size(0), -1))

n_params = sum(p.numel() for p in Res3DCNN().parameters())
print(f"Params: {n_params:,}")

class EMA:
    def __init__(self, m, d=0.999):
        self.m, self.d = m, d
        self.sh = {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad}
        self.bk = {}
    def upd(self):
        for n, p in self.m.named_parameters():
            if p.requires_grad: self.sh[n] = self.d * self.sh[n] + (1-self.d) * p.data
    def on(self):
        for n, p in self.m.named_parameters():
            if p.requires_grad: self.bk[n] = p.data.clone(); p.data = self.sh[n]
    def off(self):
        for n, p in self.m.named_parameters():
            if p.requires_grad: p.data = self.bk[n]

class Trainer:
    def __init__(self, mdl, fdir):
        self.mdl = mdl.to(DEVICE)
        self.fdir = fdir; os.makedirs(fdir, exist_ok=True)
        self.ce = nn.CrossEntropyLoss()
        self.opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=1e-5)
        self.sch = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=NUM_EPOCHS, eta_min=1e-6)
        self.scl = GradScaler()
        self.ema = EMA(mdl)
        self.best, self.pat = 0, 0

    def run(self, dl, train):
        self.mdl.train() if train else self.mdl.eval()
        ls, ps, ys, pr = 0, [], [], []
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if train:
                self.opt.zero_grad(set_to_none=True)
                with autocast(): out = self.mdl(x); loss = self.ce(out, y)
                self.scl.scale(loss).backward()
                self.scl.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.mdl.parameters(), 1.0)
                self.scl.step(self.opt); self.scl.update(); self.ema.upd()
            else:
                self.ema.on(); out = self.mdl(x); loss = self.ce(out, y); self.ema.off()
            ls += loss.item()
            p = torch.softmax(out.detach(), 1)
            ps.extend(p.argmax(1).cpu().numpy()); ys.extend(y.cpu().numpy()); pr.append(p.cpu().numpy())
        pr = np.concatenate(pr)
        return ls/len(dl), accuracy_score(ys, ps), roc_auc_score(ys, pr[:,1]) if len(np.unique(ys))>1 else .5, ps, ys, pr

    def fit(self, tr, va, ep):
        for e in range(ep):
            tl,ta,_,_,_,_ = self.run(tr, True)
            vl,vacc,vau,_,_,_ = self.run(va, False)
            self.sch.step()
            if (e+1)%10==0 or e==0: print(f" E{e+1:3d}| T L={tl:.3f} A={ta:.3f} | V L={vl:.3f} A={vacc:.3f} AUC={vau:.3f}")
            if vau>self.best+.001: self.best=vau; self.pat=0; self.ema.on(); torch.save(self.mdl.state_dict(),f"{self.fdir}/best.pth"); self.ema.off()
            else: self.pat+=1
            if self.pat>=25: print("  EarlyStop"); break

    def load(self): self.mdl.load_state_dict(torch.load(f"{self.fdir}/best.pth", map_location=DEVICE))

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"Res3DCNN | {n_params:,} params | {FINAL_SIZE}")
print("="*50)

res = []
for fold in range(N_FOLDS):
    print(f"\nFold {fold+1}/{N_FOLDS}")
    fd = f"{OUTPUT_DIR}/f{fold+1}"; os.makedirs(fd, exist_ok=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    sp = list(skf.split(files, labels))
    te_f = [files[i] for i in sp[fold][1]]; te_l = [labels[i] for i in sp[fold][1]]
    tv_f = [files[i] for i in sp[fold][0]]; tv_l = [labels[i] for i in sp[fold][0]]
    tr_f, va_f, tr_l, va_l = train_test_split(tv_f, tv_l, test_size=0.18, stratify=tv_l, random_state=SEED)
    print(f"Train={len(tr_f)} Val={len(va_f)} Test={len(te_f)}")
    tr_dl = DataLoader(CTDataset(tr_f, tr_l, True), BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    va_dl = DataLoader(CTDataset(va_f, va_l, False), BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    te_dl = DataLoader(CTDataset(te_f, te_l, False), BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    t = Trainer(Res3DCNN(), fd)
    t.fit(tr_dl, va_dl, NUM_EPOCHS); t.load()
    _, acc, auc, preds, lbls, probs = t.run(te_dl, False)
    print(f"  >> TEST Acc={acc:.4f} AUC={auc:.4f}")
    res.append({"acc": acc, "auc": auc})
    pd.DataFrame({"file":[os.path.basename(f) for f in te_f],"true":lbls,"pred":preds,"p0":probs[:,0],"p1":probs[:,1]}).to_csv(f"{fd}/preds.csv", index=False)

print(f"\n{'='*50}")
accs, aucs = [r["acc"] for r in res], [r["auc"] for r in res]
print(f"Acc: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
print(f"AUC: {np.mean(aucs):.4f} +- {np.std(aucs):.4f}")
print(f"Done: {OUTPUT_DIR}")
