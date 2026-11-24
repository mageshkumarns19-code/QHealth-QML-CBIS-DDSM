# %% [markdown]
# # Stage 1 — CBIS-DDSM: CNN fine-tuning & embedding export (GPU)
#
# **Hybrid quantum-classical study: classical front-end.**
#
# This notebook
# 1. loads the CBIS-DDSM lesion crops (mass + calcification ROIs) with benign / malignant labels,
# 2. keeps the **official CBIS-DDSM test set** untouched and splits the official training set
#    **by patient** into *fit* (≈80 %) and *validation* (≈20 %) → overall ≈ **64 / 16 / 20**,
# 3. fine-tunes an ImageNet-pretrained **EfficientNet-B0** (classical end-to-end baseline),
# 4. exports a 1280-d **embedding** for every ROI from (a) the fine-tuned CNN and
#    (b) the frozen ImageNet CNN (ablation), for the quantum heads in Stage 2.
#
# **Kaggle settings:** Accelerator = GPU (T4 or P100) · Internet = On (for pretrained weights) ·
# Input = `awsaf49/cbis-ddsm-breast-cancer-image-dataset` (+ Add Input → Datasets).
#
# **v2:** 448×448 input (better for fine micro-calcifications) and 3 CNN seeds (42, 43, 44); each seed
# gets its own patient-grouped fit/validation split, the official test set is shared.
#
# Runtime ≈ 1.5–2.5 h in total on a T4 (≈ 25–45 min per seed). Outputs go to `/kaggle/working/`.
# After it finishes: **Save Version → Save & Run All**, then add this notebook's output as an
# input to the Stage 2 notebook.

# %%
import glob
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# %% [markdown]
# ## 1. Configuration

# %%
QUICK = os.environ.get("QUICK") == "1"          # smoke-test mode (tiny run); leave off on Kaggle

CFG = dict(
    backbone="efficientnet_b0",   # 1280-d embedding
    img_size=224,                 # main setting; 448 for the resolution ablation
    epochs=3 if QUICK else 30,
    batch_size=8 if QUICK else 32,   # 32 at 224 px; use 24 at 448 px
    lr=1e-4,
    weight_decay=1e-4,
    patience=7,                   # early stopping on validation ROC-AUC
    val_fraction=0.2,             # of the official training patients
    seeds=[42, 43] if QUICK else [42, 43, 44],   # 3 CNN repeats
    abnormality="both",           # "both" | "mass" | "calc"
    num_workers=0 if QUICK else 2,
    pretrained=os.environ.get("NO_PRETRAINED") != "1",
)
OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, "| config:", CFG)


def find_root() -> Path:
    if os.environ.get("CBIS_ROOT"):
        return Path(os.environ["CBIS_ROOT"])
    hits = glob.glob("/kaggle/input/**/csv/dicom_info.csv", recursive=True)
    if not hits:
        raise FileNotFoundError("CBIS-DDSM not found. Add 'awsaf49/cbis-ddsm-breast-cancer-image-dataset' as input.")
    return Path(hits[0]).parent.parent


ROOT = find_root()
print("dataset root:", ROOT)
for p in sorted(ROOT.iterdir()):
    print("  ", p.name)

# %% [markdown]
# ## 2. Build the ROI table (one row per abnormality)
#
# The case-description CSVs reference original DICOM paths
# `<case>/<study UID>/<series UID>/000000.dcm`. In the Kaggle JPEG release every series is a
# folder `jpeg/<SeriesInstanceUID>/`, and `dicom_info.csv` tells which JPEG is the *cropped image*
# (vs. the ROI mask, which can live in the same series). If a series is not listed, we fall back to
# the non-binary image in the folder (masks are black/white).

# %%
def norm_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


case_files = {
    ("mass", "train"): "mass_case_description_train_set.csv",
    ("mass", "test"): "mass_case_description_test_set.csv",
    ("calc", "train"): "calc_case_description_train_set.csv",
    ("calc", "test"): "calc_case_description_test_set.csv",
}
frames = []
for (abn, split), fn in case_files.items():
    if CFG["abnormality"] not in ("both", abn):
        continue
    d = norm_cols(pd.read_csv(ROOT / "csv" / fn))
    d["abnormality"], d["official_split"] = abn, split
    frames.append(d)
cases = pd.concat(frames, ignore_index=True)
cases["label"] = (cases["pathology"].str.upper().str.strip() == "MALIGNANT").astype(int)
cases["patient_id"] = cases["patient_id"].str.strip()

dicom = pd.read_csv(ROOT / "csv" / "dicom_info.csv")
dicom["SeriesDescription"] = dicom["SeriesDescription"].astype(str).str.strip().str.lower()
dicom["jpg"] = dicom["image_path"].astype(str).str.replace("CBIS-DDSM/jpeg", str(ROOT / "jpeg"), regex=False)
crop_by_series = (dicom[dicom["SeriesDescription"] == "cropped images"]
                  .drop_duplicates("SeriesInstanceUID").set_index("SeriesInstanceUID")["jpg"].to_dict())


def is_binary_mask(path: str) -> bool:
    """ROI masks are black/white; JPEG compression adds slight grey noise, so test with a tolerance."""
    a = np.asarray(Image.open(path).convert("L").resize((128, 128), Image.NEAREST)).astype(int)
    return ((a < 20) | (a > 235)).mean() > 0.95


def resolve_crop(dcm_path: str):
    parts = [p for p in str(dcm_path).strip().replace("\\", "/").split("/") if p]
    for uid in reversed(parts[:-1]):           # series UID is normally the 2nd-to-last segment
        if uid in crop_by_series and os.path.exists(crop_by_series[uid]):
            return crop_by_series[uid]
    for uid in reversed(parts[:-1]):           # fallback: inspect the series folder
        folder = ROOT / "jpeg" / uid
        if folder.is_dir():
            imgs = sorted(folder.glob("*.jpg"))
            non_mask = [str(p) for p in imgs if not is_binary_mask(str(p))]
            if non_mask:
                return non_mask[0]
    return None


cases["image"] = cases["cropped_image_file_path"].map(resolve_crop)
missing = cases["image"].isna().sum()
print(f"ROIs: {len(cases)} | unresolved images dropped: {missing}")
cases = cases.dropna(subset=["image"]).reset_index(drop=True)

# patients that occur in both official splits are removed from training (no patient leakage)
test_patients = set(cases.loc[cases.official_split == "test", "patient_id"])
overlap = cases.official_split.eq("train") & cases.patient_id.isin(test_patients)
print("train ROIs from patients also in the official test set (removed):", int(overlap.sum()))
cases = cases[~overlap].reset_index(drop=True)

# %% [markdown]
# ## 3. Patient-grouped split: fit / validation / test

# %%
def make_split(df, seed):
    df = df.copy()
    df["split"] = np.where(df.official_split == "test", "test", "fit")
    tr = df[df.split == "fit"]
    n_splits = max(2, int(round(1 / CFG["val_fraction"])))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, val_idx = next(sgkf.split(tr, tr.label, groups=tr.patient_id))
    df.loc[tr.index[val_idx], "split"] = "val"
    assert not (set(df[df.split == "fit"].patient_id) & set(df[df.split == "val"].patient_id))
    assert not (set(df[df.split != "test"].patient_id) & set(df[df.split == "test"].patient_id))
    return df


summary = make_split(cases, CFG["seeds"][0]).groupby("split").agg(
    rois=("label", "size"), malignant=("label", "sum"), patients=("patient_id", "nunique"))
summary["share"] = (summary.rois / summary.rois.sum()).round(3)
print(summary)

# %% [markdown]
# ## 4. Data pipeline

# %%
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
S = CFG["img_size"]
train_tf = transforms.Compose([
    transforms.Resize((int(S * 1.15), int(S * 1.15))),
    transforms.RandomResizedCrop(S, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((S, S)), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])


class ROIDataset(Dataset):
    def __init__(self, df, tf):
        self.paths, self.labels, self.tf = df.image.tolist(), df.label.values.astype(np.float32), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("L").convert("RGB")   # grayscale -> 3 channels
        return self.tf(img), self.labels[i]


def loader(df, tf, shuffle):
    return DataLoader(ROIDataset(df, tf), batch_size=CFG["batch_size"], shuffle=shuffle,
                      num_workers=CFG["num_workers"], pin_memory=DEVICE.type == "cuda")

# %% [markdown]
# ## 5. Model: EfficientNet-B0 with a 1-logit head (embedding = 1280-d pooled features)

# %%
class Net(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        w = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.efficientnet_b0(weights=w)
        self.features, self.pool = m.features, m.avgpool
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(1280, 1)

    def embed(self, x):
        return torch.flatten(self.pool(self.features(x)), 1)

    def forward(self, x):
        return self.head(self.dropout(self.embed(x))).squeeze(1)


@torch.no_grad()
def run_inference(model, df, tta=True):
    """Returns (probabilities, embeddings); TTA = average with horizontal flip."""
    model.eval()
    probs, embs = [], []
    for x, _ in loader(df, eval_tf, shuffle=False):
        x = x.to(DEVICE)
        with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            e = model.embed(x)
            if tta:
                e = (e + model.embed(torch.flip(x, dims=[3]))) / 2
            logit = model.head(e).squeeze(1)
        probs.append(torch.sigmoid(logit.float()).cpu().numpy())
        embs.append(e.float().cpu().numpy())
    return np.concatenate(probs), np.concatenate(embs)


def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# %% [markdown]
# ## 6. Fine-tune (early stopping on validation ROC-AUC) and export embeddings

# %%
def train_one_seed(seed):
    seed_everything(seed)
    df = make_split(cases, seed)
    fit, val, test = (df[df.split == s].reset_index(drop=True) for s in ("fit", "val", "test"))
    model = Net(CFG["pretrained"]).to(DEVICE)
    pos_weight = torch.tensor(float((fit.label == 0).sum()) / max(float((fit.label == 1).sum()), 1.0), device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["epochs"])
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")
    best, best_state, wait, hist = -1, None, 0, []
    for epoch in range(CFG["epochs"]):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in loader(fit, train_tf, shuffle=True):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
        sched.step()
        pv, _ = run_inference(model, val, tta=False)
        auc = roc_auc_score(val.label, pv) if val.label.nunique() > 1 else float("nan")
        hist.append({"epoch": epoch + 1, "train_loss": tot / len(fit), "val_auc": auc, "sec": time.time() - t0})
        print(hist[-1])
        if auc > best + 1e-4:
            best, wait = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= CFG["patience"]:
                print("early stop"); break
    model.load_state_dict(best_state)
    torch.save(best_state, OUT / f"cnn_seed{seed}.pt")

    out = {}
    for name, part in (("fit", fit), ("val", val), ("test", test)):
        p, e = run_inference(model, part)
        out.update({f"X_{name}": e.astype(np.float32), f"y_{name}": part.label.values,
                    f"cnn_prob_{name}": p, f"patient_{name}": part.patient_id.values,
                    f"abn_{name}": part.abnormality.values})
    np.savez_compressed(OUT / f"embeddings_finetuned_seed{seed}.npz", **out)
    df.to_csv(OUT / f"split_seed{seed}.csv", index=False)
    pd.DataFrame(hist).to_csv(OUT / f"history_seed{seed}.csv", index=False)
    test_auc = roc_auc_score(out["y_test"], out["cnn_prob_test"])
    print(f"seed {seed}: best val AUC {best:.4f} | CNN test AUC {test_auc:.4f}")
    return {"seed": seed, "best_val_auc": best, "cnn_test_auc": test_auc, "epochs_run": len(hist)}, (fit, val, test)


results, parts = [], None
for s in CFG["seeds"]:
    r, parts = train_one_seed(s)
    results.append(r)
print(pd.DataFrame(results))

# %% [markdown]
# ## 7. Ablation: frozen ImageNet embeddings (no fine-tuning)

# %%
frozen = Net(CFG["pretrained"]).to(DEVICE)
out = {}
for name, part in zip(("fit", "val", "test"), parts):
    _, e = run_inference(frozen, part)
    out.update({f"X_{name}": e.astype(np.float32), f"y_{name}": part.label.values,
                f"patient_{name}": part.patient_id.values, f"abn_{name}": part.abnormality.values})
np.savez_compressed(OUT / "embeddings_imagenet.npz", **out)

json.dump({"config": CFG, "split_summary": summary.reset_index().to_dict(orient="records"),
           "cnn_results": results}, open(OUT / "stage1_summary.json", "w"), indent=2, default=str)
print("Saved:", sorted(p.name for p in OUT.iterdir()))
