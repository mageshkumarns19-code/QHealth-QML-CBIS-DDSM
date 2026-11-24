"""Re-run Stage 2 heads (CNN seed 42, head seed 0) to save per-patient predictions + training histories."""
import os, sys, pickle, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OUT_DIR", str(ROOT / "figures" / "tmp_out"))
src = open(ROOT / "src" / "stage2_qml_heads.py").read()
cut = src.index("all_rows, all_info, roc_store")
src = src[:cut]
HIST = {}
g = {"__name__": "s2"}
exec(compile(src, "stage2", "exec"), g)
orig = g["train_torch"]
def rec(module, *a, **k):
    m, h = orig(module, *a, **k)
    HIST[type(module).__name__] = h
    return m, h
g["train_torch"] = rec
D = g["load"](sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "outputs" / "stage1_224px" / "embeddings_finetuned_seed42.npz"))
preds, info = g["fit_heads"](D, 6, 0, {})
out = {"preds": {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in preds.items()}, "info": info, "hist": HIST,
       "y_val": D["y_val"].astype(int), "y_test": D["y_test"].astype(int), "patient_test": D["patient_test"],
       "abn_test": D["abn_test"]}
pickle.dump(out, open(ROOT / "figures" / "preds_seed42_head0.pkl", "wb"))
from sklearn.metrics import roc_auc_score
print({k: round(roc_auc_score(out["y_test"], v[1]), 3) for k, v in out["preds"].items()})
print({k: len(v) for k, v in HIST.items()})
