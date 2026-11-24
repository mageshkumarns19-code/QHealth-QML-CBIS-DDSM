# %% [markdown]
# # Stage 2 — Hybrid quantum heads on CBIS-DDSM embeddings: discrimination, calibration, noise
#
# **Input:** the output of Stage 1 (`embeddings_finetuned_seed*.npz`, `embeddings_imagenet.npz`).
# Add the Stage 1 notebook as an input (+ Add Input → Notebook Output).
#
# **Settings:** Accelerator = None (CPU is enough for 4–8 qubit simulation) · Internet = On (pip install).
#
# Protocol (identical for every head):
# * fit heads on the **fit** split, tune hyper-parameters, decision thresholds and calibration on
#   the **validation** split, report once on the untouched **official test** split;
# * every head sees the same PCA-compressed embedding (width = number of qubits); a logistic
#   regression on the full 1280-d embedding and the CNN's own head are reported as references;
# * repeated over several seeds (head initialisation / quantum parameter init);
# * metrics: ROC-AUC with **patient-cluster bootstrap** 95 % CI, **DeLong** test vs. reference,
#   sensitivity at fixed specificity, specificity at fixed sensitivity, Brier, ECE, calibration
#   slope/intercept, decision-curve net benefit;
# * **noise study** (v2: all head seeds, VQC *and* QSVM): trained models deployed with depolarising
#   noise + shot noise; threshold drift and recovery after **recalibration with 20 / 50 / 100 patients**;
# * results pooled over **all Stage 1 CNN seeds** (`stage2_summary_pooled.csv`).

# %%
import subprocess, sys
try:
    import pennylane  # noqa: F401
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pennylane"], check=True)

import glob
import json
import os
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import scipy.stats as st
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
torch.set_num_threads(max(1, os.cpu_count() or 1))
print("PennyLane", qml.__version__, "| torch", torch.__version__)

# %% [markdown]
# ## 1. Configuration

# %%
QUICK = os.environ.get("QUICK") == "1"     # smoke-test mode; leave off on Kaggle

CFG = dict(
    n_qubits=6,
    seeds=[0, 1] if QUICK else [0, 1, 2, 3, 4],
    qubit_ablation=[4] if QUICK else [4, 8],     # extra widths (QSVM + VQC, seed 0, fine-tuned embeddings)
    vqc=dict(layers=3, epochs=4 if QUICK else 60, lr=0.05, batch=32, patience=12),
    qnn=dict(layers=2, hidden=16, in_dim=32, epochs=4 if QUICK else 60, lr=0.02, batch=32, patience=12),
    qsvm_grid=dict(bandwidth=[0.02, 0.05, 0.1, 0.2], C=[1.0, 10.0]),
    target_sensitivity=0.95,     # screening / rule-out operating point
    target_specificity=0.90,     # matched-specificity comparison
    n_boot=200 if QUICK else 1000,
    noise_p=[0.0, 0.01, 0.05] if QUICK else [0.0, 0.005, 0.01, 0.02, 0.05, 0.1],
    noise_shots=1024,
    recal_sizes=[20, 50, 100],
)
OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
OUT.mkdir(parents=True, exist_ok=True)

files = sorted(glob.glob(os.environ.get("EMB_GLOB", "/kaggle/input/**/embeddings_*.npz"), recursive=True))
assert files, "No embeddings_*.npz found - add the Stage 1 notebook output as an input."
print("embedding files:", *[Path(f).name for f in files], sep="\n  ")


def load(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}

# %% [markdown]
# ## 2. Classical pre-processing: standardise → PCA → angle scaling
# Fitted on the *fit* split only.

# %%
class Compressor:
    def __init__(self, n, angle=True):
        self.n, self.angle = n, angle

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        self.pca = PCA(self.n, random_state=0).fit(Z)
        P = self.pca.transform(Z)
        self.lo, self.hi = np.quantile(P, 0.01, 0), np.quantile(P, 0.99, 0)
        return self

    def transform(self, X):
        P = self.pca.transform(self.sc.transform(np.asarray(X, dtype=np.float64)))
        if not self.angle:
            return P
        return np.pi * np.clip((P - self.lo) / np.where(self.hi > self.lo, self.hi - self.lo, 1), 0, 1)

# %% [markdown]
# ## 3. Quantum models
# * **QSVM** – fidelity kernel of a second-order ZZ feature map (2 reps, linear entanglement) with
#   **bandwidth** scaling to avoid kernel concentration; kernel matrix → classical SVM.
# * **VQC** – data re-uploading: [RY(x) · StronglyEntanglingLayer(θ_l)] × L, readout ⟨Z₀⟩ → affine logit.
# * **Hybrid QNN** – classical encoder (PCA-32 → 16 → n) → AngleEmbedding → L entangling layers →
#   ⟨Z_i⟩ → linear head.

# %%
def zz_map(x, wires, reps=2):
    n = len(wires)
    for _ in range(reps):
        for i in range(n):
            qml.Hadamard(wires=i)
            qml.RZ(2.0 * x[..., i], wires=i)
        for i in range(n - 1):
            qml.IsingZZ(2.0 * (np.pi - x[..., i]) * (np.pi - x[..., i + 1]), wires=[i, i + 1])


class QSVM:
    def __init__(self, n, bandwidth=0.05, C=10.0, reps=2, seed=0):
        self.n, self.bw, self.C, self.reps, self.seed = n, bandwidth, C, reps, seed
        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev)
        def state(x):
            zz_map(x, list(range(n)), reps)
            return qml.state()
        self._state = state

    def states(self, X):
        return np.concatenate([np.asarray(self._state(self.bw * X[i:i + 512])).reshape(len(X[i:i + 512]), -1)
                               for i in range(0, len(X), 512)])

    def fit(self, X, y):
        self.S = self.states(X)
        K = np.abs(self.S @ self.S.conj().T) ** 2
        self.svc = SVC(kernel="precomputed", C=self.C, probability=True, class_weight="balanced",
                       random_state=self.seed).fit(K, y)
        off = K[~np.eye(len(K), dtype=bool)]
        self.kernel_stats = {"offdiag_mean": float(off.mean()), "offdiag_std": float(off.std())}
        return self

    def predict_proba(self, X):
        return self.svc.predict_proba(np.abs(self.states(X) @ self.S.conj().T) ** 2)[:, 1]


def vqc_circuit(x, w, n, noise_p=0.0):
    for layer in range(w.shape[0]):
        for i in range(n):
            qml.RY(x[..., i], wires=i)
        qml.StronglyEntanglingLayers(w[layer:layer + 1], wires=range(n))
        if noise_p > 0:
            for i in range(n):
                qml.DepolarizingChannel(noise_p, wires=i)


class VQCModule(torch.nn.Module):
    def __init__(self, n, layers):
        super().__init__()
        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circ(inputs, weights):
            vqc_circuit(inputs, weights, n)
            return qml.expval(qml.PauliZ(0))
        self.q = qml.qnn.TorchLayer(circ, {"weights": (layers, n, 3)},
                                    init_method=lambda t: torch.nn.init.uniform_(t, 0, 2 * np.pi))
        self.scale = torch.nn.Parameter(torch.tensor(3.0))
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return -self.scale * self.q(x).reshape(-1) + self.bias


class QNNModule(torch.nn.Module):
    def __init__(self, d, n, layers, hidden):
        super().__init__()
        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circ(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n))
            return [qml.expval(qml.PauliZ(i)) for i in range(n)]
        self.enc = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.Tanh(),
                                       torch.nn.Linear(hidden, n), torch.nn.Tanh())
        self.q = qml.qnn.TorchLayer(circ, {"weights": (layers, n, 3)},
                                    init_method=lambda t: torch.nn.init.normal_(t, 0, 0.5))
        self.head = torch.nn.Linear(n, 1)

    def forward(self, x):
        q = self.q((self.enc(x) + 1) * np.pi / 2)
        return self.head(q.reshape(len(x), -1)).reshape(-1)


def train_torch(module, Xf, yf, Xv, yv, epochs, lr, batch, patience, seed):
    """Adam on all (classical + quantum) parameters; early stopping on validation loss."""
    torch.manual_seed(seed)
    module = module.double()
    Xf, yf = torch.tensor(np.asarray(Xf, dtype=np.float64)), torch.tensor(yf, dtype=torch.float64)
    Xv, yv = torch.tensor(np.asarray(Xv, dtype=np.float64)), torch.tensor(yv, dtype=torch.float64)
    pw = torch.tensor((len(yf) - yf.sum().item()) / max(yf.sum().item(), 1), dtype=torch.float64)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(epochs):
        module.train()
        perm = torch.randperm(len(Xf), generator=g)
        for s in range(0, len(Xf), batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss_fn(module(Xf[idx]), yf[idx]).backward()
            opt.step()
        module.eval()
        with torch.no_grad():
            vl = float(loss_fn(module(Xv), yv))
        hist.append(vl)
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in module.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    module.load_state_dict(best_state)
    module.eval()
    return module, hist


def torch_proba(module, X):
    with torch.no_grad():
        return torch.sigmoid(module(torch.tensor(np.asarray(X, dtype=np.float64)))).numpy()

# %% [markdown]
# ## 4. Clinical evaluation toolkit

# %%
def thr_youden(y, p):
    f, t, th = roc_curve(y, p)
    return float(np.clip(th[np.argmax(t - f)], 0, 1))


def thr_sens(y, p, target):
    f, t, th = roc_curve(y, p)
    return float(np.clip(th[np.where(t >= target)[0][0]], 0, 1))


def thr_spec(y, p, target):
    f, t, th = roc_curve(y, p)
    ok = np.where(1 - f >= target)[0]
    return float(np.clip(th[ok[-1]], 0, 1))


def sens_spec(y, p, t):
    yhat = p >= t
    return float(yhat[y == 1].mean()), float((~yhat[y == 0]).mean())


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def calib_slope_intercept(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    m = LogisticRegression(C=1e6).fit(lp.reshape(-1, 1), y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def net_benefit(y, p, pt):
    n = len(y)
    tp = np.sum((p >= pt) & (y == 1)); fp = np.sum((p >= pt) & (y == 0))
    return float(tp / n - fp / n * pt / (1 - pt))


def cluster_bootstrap_auc(y, p, groups, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rows = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[g] for g in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], p[idx]))
    return tuple(np.quantile(vals, [0.025, 0.975]).round(4))


def delong(y, p1, p2):
    """Two-sided DeLong test for correlated ROC-AUCs (Sun & Xu fast algorithm)."""
    def midrank(x):
        j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n); i = 0
        while i < n:
            k = i
            while k < n and z[k] == z[i]:
                k += 1
            t[i:k] = 0.5 * (i + k - 1) + 1; i = k
        out = np.empty(n); out[j] = t; return out
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v01, v10 = [], [], []
    for p in (p1, p2):
        tx, ty, tz = midrank(p[pos]), midrank(p[neg]), midrank(np.concatenate([p[pos], p[neg]]))
        aucs.append(tz[:m].sum() / m / n - (m + 1) / 2 / n)
        v01.append((tz[:m] - tx) / n); v10.append(1 - (tz[m:] - ty) / m)
    s = np.cov(np.vstack(v01)) / m + np.cov(np.vstack(v10)) / n
    z = (aucs[0] - aucs[1]) / np.sqrt(max(s[0, 0] + s[1, 1] - 2 * s[0, 1], 1e-12))
    return float(aucs[0] - aucs[1]), float(2 * st.norm.sf(abs(z)))


def evaluate(name, p_val, y_val, p_test, y_test, g_test):
    t_y = thr_youden(y_val, p_val)
    t_rule_out = thr_sens(y_val, p_val, CFG["target_sensitivity"])
    t_spec = thr_spec(y_val, p_val, CFG["target_specificity"])
    se_y, sp_y = sens_spec(y_test, p_test, t_y)
    se_ro, sp_ro = sens_spec(y_test, p_test, t_rule_out)
    se_s, sp_s = sens_spec(y_test, p_test, t_spec)
    f, t, _ = roc_curve(y_test, p_test)
    slope, intercept = calib_slope_intercept(y_test, p_test)
    lo, hi = cluster_bootstrap_auc(y_test, p_test, g_test, CFG["n_boot"])
    return {
        "model": name, "auc": roc_auc_score(y_test, p_test), "auc_lo": lo, "auc_hi": hi,
        "sens@youden": se_y, "spec@youden": sp_y,
        "sens@ruleout(val sens 0.95)": se_ro, "spec@ruleout(val sens 0.95)": sp_ro,
        "sens@val-spec0.90": se_s, "spec@val-spec0.90": sp_s,
        "oracle sens@spec0.90": float(t[np.where(1 - f >= CFG["target_specificity"])[0][-1]]),
        "oracle spec@sens0.95": float(1 - f[np.where(t >= CFG["target_sensitivity"])[0][0]]),
        "brier": float(np.mean((p_test - y_test) ** 2)), "ece": ece(y_test, p_test),
        "calib_slope": slope, "calib_intercept": intercept,
        **{f"NB@{pt}": net_benefit(y_test, p_test, pt) for pt in (0.1, 0.2, 0.3, 0.4, 0.5)},
        "thr_youden": t_y, "thr_ruleout": t_rule_out,
    }

# %% [markdown]
# ## 5. Train & evaluate all heads

# %%
def fit_heads(D, n, seed, store=None):
    yf, yv, yt = D["y_fit"].astype(int), D["y_val"].astype(int), D["y_test"].astype(int)
    comp = Compressor(n, angle=True).fit(D["X_fit"])
    Af, Av, At = comp.transform(D["X_fit"]), comp.transform(D["X_val"]), comp.transform(D["X_test"])
    comp_raw = Compressor(n, angle=False).fit(D["X_fit"])
    Rf, Rv, Rt = comp_raw.transform(D["X_fit"]), comp_raw.transform(D["X_val"]), comp_raw.transform(D["X_test"])
    preds, info = {}, {}

    # --- QSVM: grid over bandwidth x C, selected on validation AUC
    t0, best = time.time(), (-1, None)
    for bw in CFG["qsvm_grid"]["bandwidth"]:
        for C in CFG["qsvm_grid"]["C"]:
            m = QSVM(n, bw, C, seed=seed).fit(Af, yf)
            auc = roc_auc_score(yv, m.predict_proba(Av))
            if auc > best[0]:
                best = (auc, m)
    m = best[1]
    preds["QSVM"] = (m.predict_proba(Av), m.predict_proba(At))
    info["QSVM"] = {"bandwidth": m.bw, "C": m.C, **m.kernel_stats, "sec": time.time() - t0}

    # --- VQC
    t0 = time.time(); c = CFG["vqc"]
    vqc, hist = train_torch(VQCModule(n, c["layers"]), Af, yf, Av, yv, c["epochs"], c["lr"], c["batch"], c["patience"], seed)
    preds["VQC"] = (torch_proba(vqc, Av), torch_proba(vqc, At))
    info["VQC"] = {"epochs_run": len(hist), "sec": time.time() - t0}

    # --- hybrid QNN (classical encoder sees a wider PCA input)
    t0 = time.time(); c = CFG["qnn"]
    wide = Compressor(min(c["in_dim"], D["X_fit"].shape[1]), angle=False).fit(D["X_fit"])
    Wf, Wv, Wt = wide.transform(D["X_fit"]), wide.transform(D["X_val"]), wide.transform(D["X_test"])
    s = Wf.std(0) + 1e-9
    qnn, hist = train_torch(QNNModule(Wf.shape[1], n, c["layers"], c["hidden"]), Wf / s, yf, Wv / s, yv,
                            c["epochs"], c["lr"], c["batch"], c["patience"], seed)
    preds["QNN"] = (torch_proba(qnn, Wv / s), torch_proba(qnn, Wt / s))
    info["QNN"] = {"epochs_run": len(hist), "sec": time.time() - t0}

    # --- classical heads on the SAME n-dim input (controls), tuned on validation AUC
    def tuned(make, grid, Xf_, Xv_, Xt_):
        best = (-1, None)
        for g in grid:
            mdl = make(g).fit(Xf_, yf)
            a = roc_auc_score(yv, mdl.predict_proba(Xv_)[:, 1])
            if a > best[0]:
                best = (a, mdl)
        return best[1].predict_proba(Xv_)[:, 1], best[1].predict_proba(Xt_)[:, 1]

    preds[f"LogReg ({n}-PCA)"] = tuned(lambda C: LogisticRegression(C=C, class_weight="balanced", max_iter=5000),
                                       [0.01, 0.1, 1, 10], Rf, Rv, Rt)
    preds[f"RBF-SVM ({n}-PCA)"] = tuned(lambda C: SVC(C=C, probability=True, class_weight="balanced", random_state=seed),
                                        [0.1, 1, 10], Rf, Rv, Rt)
    preds[f"MLP ({n}-PCA)"] = tuned(lambda a: MLPClassifier((32, 16), alpha=a, max_iter=2000, early_stopping=True,
                                                            random_state=seed), [1e-4, 1e-3, 1e-2], Rf, Rv, Rt)
    # --- references on the full embedding
    sc = StandardScaler().fit(D["X_fit"])
    preds["LogReg (full 1280-d)"] = tuned(lambda C: LogisticRegression(C=C, class_weight="balanced", max_iter=5000),
                                          [0.001, 0.01, 0.1], sc.transform(D["X_fit"]), sc.transform(D["X_val"]),
                                          sc.transform(D["X_test"]))
    if "cnn_prob_val" in D:
        preds["CNN end-to-end"] = (D["cnn_prob_val"], D["cnn_prob_test"])
    if store is not None:
        store.update({"vqc": vqc, "qsvm": m, "Av": Av, "At": At, "comp": comp})
    return preds, info


all_rows, all_info, roc_store, noise_store = [], [], {}, {}
# the noise study uses every head seed of the first fine-tuned CNN (or the first file if none)
KEY_FILE = next((f for f in files if "finetuned" in Path(f).name), files[0])
for path in files:
    D = load(path)
    emb = Path(path).stem.replace("embeddings_", "")
    print(f"\n=== {emb}: fit {len(D['y_fit'])} | val {len(D['y_val'])} | test {len(D['y_test'])} ROIs ===")
    for seed in CFG["seeds"]:
        t0 = time.time()
        store = {} if (seed == CFG["seeds"][0] or path == KEY_FILE) else None
        preds, info = fit_heads(D, CFG["n_qubits"], seed, store)
        ref = preds.get("CNN end-to-end", preds["LogReg (full 1280-d)"])[1]
        for name, (pv, pt) in preds.items():
            r = evaluate(name, pv, D["y_val"].astype(int), pt, D["y_test"].astype(int), D["patient_test"])
            r["delta_auc_vs_ref"], r["delong_p_vs_ref"] = delong(D["y_test"].astype(int), pt, ref)
            r.update({"embedding": emb, "seed": seed, "family": "quantum" if name in ("QSVM", "VQC", "QNN") else "classical"})
            all_rows.append(r)
        all_info.append({"embedding": emb, "seed": seed, **{f"{k}_{kk}": vv for k, v in info.items() for kk, vv in v.items()}})
        if store is not None and seed == CFG["seeds"][0]:
            roc_store[emb] = {"preds": preds, "D": D, **store}
        if path == KEY_FILE:
            noise_store[seed] = {"D": D, **store}
        print(f"seed {seed} done in {time.time() - t0:.0f}s")

res = pd.DataFrame(all_rows)
res.to_csv(OUT / "stage2_results_all_seeds.csv", index=False)
pd.DataFrame(all_info).to_csv(OUT / "stage2_model_info.csv", index=False)

# %% [markdown]
# ## 6. Summary tables
# * `stage2_summary.csv` – per embedding file (mean ± SD over head seeds)
# * `stage2_summary_pooled.csv` – fine-tuned embeddings pooled over all CNN seeds × head seeds
#   (the headline table: SD now includes CNN training variability)

# %%
cols = ["auc", "sens@youden", "spec@youden", "sens@val-spec0.90", "spec@val-spec0.90",
        "sens@ruleout(val sens 0.95)", "spec@ruleout(val sens 0.95)", "oracle sens@spec0.90",
        "brier", "ece", "calib_slope", "NB@0.3", "delong_p_vs_ref"]
g = res.groupby(["embedding", "family", "model"])[cols]
summary = g.mean().round(3).astype(str) + " ± " + g.std().fillna(0).round(3).astype(str)
summary = summary.reset_index().sort_values(["embedding", "family", "model"])
summary.to_csv(OUT / "stage2_summary.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(summary.to_string(index=False))

res["emb_type"] = res["embedding"].str.replace(r"_seed\d+$", "", regex=True)
gp = res.groupby(["emb_type", "family", "model"])
pooled = (gp[cols].mean().round(3).astype(str) + " ± " + gp[cols].std().fillna(0).round(3).astype(str))
pooled.insert(0, "n_runs", gp.size())
pooled.insert(1, "n_cnn_seeds", gp["embedding"].nunique())
pooled = pooled.reset_index().sort_values(["emb_type", "family", "model"])
pooled.to_csv(OUT / "stage2_summary_pooled.csv", index=False)
print("\n=== pooled over CNN seeds x head seeds ===")
print(pooled.to_string(index=False))

# %% [markdown]
# ## 7. Figures: ROC, calibration, decision curves (first seed, fine-tuned embeddings)

# %%
key = Path(KEY_FILE).stem.replace("embeddings_", "")
S = roc_store[key]
yt = S["D"]["y_test"].astype(int)
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for name, (_, pt) in S["preds"].items():
    q = name in ("QSVM", "VQC", "QNN")
    f, t, _ = roc_curve(yt, pt)
    ax[0].plot(f, t, lw=2.2 if q else 1.1, ls="-" if q else "--", label=f"{name} ({roc_auc_score(yt, pt):.3f})")
    bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(pt, bins) - 1, 0, 9)
    xs = [pt[idx == b].mean() for b in range(10) if (idx == b).sum() >= 5]
    ys = [yt[idx == b].mean() for b in range(10) if (idx == b).sum() >= 5]
    ax[1].plot(xs, ys, marker="o", lw=2 if q else 1, ls="-" if q else "--", label=name)
    pts = np.linspace(0.05, 0.6, 23)
    ax[2].plot(pts, [net_benefit(yt, pt, x) for x in pts], lw=2 if q else 1, ls="-" if q else "--", label=name)
prev = yt.mean(); pts = np.linspace(0.05, 0.6, 23)
ax[2].plot(pts, prev - (1 - prev) * pts / (1 - pts), color="k", lw=0.8, label="treat all")
ax[2].axhline(0, color="grey", lw=0.8)
ax[0].plot([0, 1], [0, 1], ":", color="grey"); ax[1].plot([0, 1], [0, 1], ":", color="grey")
ax[0].set(title="ROC (test)", xlabel="1 − specificity", ylabel="sensitivity")
ax[1].set(title="Calibration (test)", xlabel="predicted P(malignant)", ylabel="observed frequency")
ax[2].set(title="Decision curve (test)", xlabel="threshold probability", ylabel="net benefit", ylim=(-0.05, prev + 0.05))
for a in ax:
    a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "fig_roc_calibration_dca.png", dpi=160); plt.show()

# %% [markdown]
# ## 8. Noise study: AUC-robust but decision-fragile?
# Both trained-on-simulator quantum models are *deployed* on a noisy device (train on the simulator,
# run inference on hardware). Depolarising noise of strength *p* per qubit per layer, 1024 shots.
#
# * **VQC** – exact density-matrix simulation (`default.mixed`), channel after every layer.
# * **QSVM** – the deployed model needs the test-vs-training kernel rows, i.e. compute–uncompute
#   circuits U(x)U†(x′). Simulating ~10⁶ density-matrix circuits is infeasible, so we use the
#   standard global-depolarising approximation K̃ = f·K + (1 − f)/2ⁿ with f = (1 − p)^(2·reps·n)
#   (the same per-qubit-per-layer budget), followed by binomial shot noise on every kernel entry.
#
# For each head seed we track (i) ROC-AUC, (ii) sensitivity/specificity at the **noiseless**
# validation-tuned rule-out threshold (threshold drift) and (iii) recovery after re-tuning the
# threshold on *k* validation patients executed on the noisy device (20 random draws per *k*).

# %%
def vqc_noisy_proba(module, X, p, shots, n, rng=None):
    w = module.q.weights.detach().numpy()
    dev = qml.device("default.mixed", wires=n, shots=shots)

    @qml.qnode(dev)
    def circ(x):
        vqc_circuit(x, w, n, noise_p=p)
        return qml.expval(qml.PauliZ(0))
    try:
        z = np.asarray(circ(X), dtype=float).reshape(-1)
    except Exception:
        z = np.array([float(circ(x)) for x in X])
    s, b = float(module.scale), float(module.bias)
    return 1 / (1 + np.exp(-(-s * z + b)))


def qsvm_noisy_proba(m, X, p, shots, n, rng):
    K = np.abs(m.states(X) @ m.S.conj().T) ** 2
    f = (1 - p) ** (2 * m.reps * m.n)                 # global-depolarising approximation
    K = f * K + (1 - f) / 2 ** m.n
    if shots:
        K = rng.binomial(shots, np.clip(K, 0, 1)) / shots
    return m.svc.predict_proba(K)[:, 1]


def noise_curve(model_name, proba_fn, model, S, seed):
    yv, yt = S["D"]["y_val"].astype(int), S["D"]["y_test"].astype(int)
    rng = np.random.default_rng(1000 + seed)
    p_val_clean = proba_fn(model, S["Av"], 0.0, None, CFG["n_qubits"], rng)
    t_clean = thr_sens(yv, p_val_clean, CFG["target_sensitivity"])
    rows = []
    for p in CFG["noise_p"]:
        pv = proba_fn(model, S["Av"], p, CFG["noise_shots"], CFG["n_qubits"], rng)
        pt = proba_fn(model, S["At"], p, CFG["noise_shots"], CFG["n_qubits"], rng)
        se, sp = sens_spec(yt, pt, t_clean)
        row = {"model": model_name, "seed": seed, "p": p, "auc": roc_auc_score(yt, pt),
               "mean_prob": float(pt.mean()), "sd_prob": float(pt.std()),
               "sens@clean_thr": se, "spec@clean_thr": sp}
        n_pos_avail, n_neg_avail = int((yv == 1).sum()), int((yv == 0).sum())
        for k in CFG["recal_sizes"]:
            if k > len(yv) or n_pos_avail < 2:
                row[f"sens@recal{k}"] = row[f"spec@recal{k}"] = np.nan      # validation set too small
                continue
            ses, sps = [], []
            for _ in range(20):   # repeated random stratified calibration subsets
                n_pos = min(n_pos_avail, max(2, int(round(k * yv.mean()))))
                pos = rng.choice(np.where(yv == 1)[0], n_pos, replace=False)
                neg = rng.choice(np.where(yv == 0)[0], min(n_neg_avail, k - n_pos), replace=False)
                idx = np.concatenate([pos, neg])
                a_, b_ = sens_spec(yt, pt, thr_sens(yv[idx], pv[idx], CFG["target_sensitivity"]))
                ses.append(a_); sps.append(b_)
            row[f"sens@recal{k}"], row[f"spec@recal{k}"] = float(np.mean(ses)), float(np.mean(sps))
        rows.append(row)
    return rows


noise_rows = []
for seed, S in noise_store.items():
    t0 = time.time()
    noise_rows += noise_curve("VQC", vqc_noisy_proba, S["vqc"], S, seed)
    noise_rows += noise_curve("QSVM", qsvm_noisy_proba, S["qsvm"], S, seed)
    print(f"noise study, head seed {seed}: {time.time() - t0:.0f}s")
noise = pd.DataFrame(noise_rows)
noise.to_csv(OUT / "stage2_noise_recalibration_all_seeds.csv", index=False)
metric_cols = [c for c in noise.columns if c not in ("model", "seed", "p")]
noise_mean = noise.groupby(["model", "p"])[metric_cols].mean()
noise_sd = noise.groupby(["model", "p"])[metric_cols].std().fillna(0)
noise_summary = (noise_mean.round(3).astype(str) + " ± " + noise_sd.round(3).astype(str)).reset_index()
noise_summary.to_csv(OUT / "stage2_noise_recalibration.csv", index=False)
print(noise_summary[["model", "p", "auc", "sens@clean_thr", "spec@clean_thr", "sens@recal20", "spec@recal20",
                     "sens@recal100", "spec@recal100"]].to_string(index=False))

fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
for r_, mdl in enumerate(["VQC", "QSVM"]):
    M, SD = noise_mean.loc[mdl], noise_sd.loc[mdl]
    x = M.index.values

    def band(a, col, style, label, color=None):
        a.plot(x, M[col], style, color=color, label=label)
        a.fill_between(x, M[col] - SD[col], M[col] + SD[col], color=a.lines[-1].get_color(), alpha=0.15)
    band(ax[r_, 0], "auc", "o-", "ROC-AUC")
    band(ax[r_, 0], "sens@clean_thr", "s--", "sensitivity @ noiseless threshold")
    band(ax[r_, 0], "spec@clean_thr", "^--", "specificity @ noiseless threshold")
    ax[r_, 0].set(title=f"{mdl}: ranking vs. decision (mean ± SD, {noise.seed.nunique()} seeds)",
                  xlabel="depolarising p", ylim=(0, 1.02)); ax[r_, 0].legend(fontsize=8)
    for i, k in enumerate(CFG["recal_sizes"]):
        if M[f"sens@recal{k}"].notna().any():
            band(ax[r_, 1], f"sens@recal{k}", "o-", f"sensitivity, recalibrated k={k}", f"C{i}")
            band(ax[r_, 1], f"spec@recal{k}", "^:", f"specificity, recalibrated k={k}", f"C{i}")
    ax[r_, 1].plot(x, M["sens@clean_thr"], "k-", lw=0.8, label="sensitivity, no recalibration")
    ax[r_, 1].plot(x, M["spec@clean_thr"], "k--", lw=0.8, label="specificity, no recalibration")
    ax[r_, 1].axhline(CFG["target_sensitivity"], color="grey", ls=":")
    ax[r_, 1].set(title=f"{mdl}: recovery by on-device recalibration", xlabel="depolarising p",
                  ylim=(0, 1.02)); ax[r_, 1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "fig_noise_recalibration.png", dpi=160); plt.show()

# %% [markdown]
# ## 8b. QSVM shot scaling (no gate noise)
# A kernel model's prediction sums hundreds of estimated kernel entries (one per support vector), so
# finite-shot sampling error accumulates in the decision function. Test AUC vs shots per kernel entry,
# for every head seed of the key CNN; also the total circuit executions the test set would need.

# %%
shot_rows = []
for seed, S in noise_store.items():
    m = S["qsvm"]
    Kt = np.abs(m.states(S["At"]) @ m.S.conj().T) ** 2
    yt = S["D"]["y_test"].astype(int)
    rng = np.random.default_rng(2000 + seed)
    n_sv = int(m.svc.support_.size)
    shot_rows.append({"seed": seed, "shots": "exact", "auc": roc_auc_score(yt, m.svc.decision_function(Kt)),
                      "n_support_vectors": n_sv, "test_circuit_runs": 0})
    for shots in [1024, 8192, 65536, 10 ** 6]:
        aucs = [roc_auc_score(yt, m.svc.decision_function(rng.binomial(shots, np.clip(Kt, 0, 1)) / shots))
                for _ in range(3)]
        shot_rows.append({"seed": seed, "shots": shots, "auc": float(np.mean(aucs)), "n_support_vectors": n_sv,
                          "test_circuit_runs": float(len(yt)) * n_sv * shots})
shots_df = pd.DataFrame(shot_rows)
shots_df.to_csv(OUT / "stage2_qsvm_shot_scaling.csv", index=False)
print(shots_df.groupby("shots", sort=False)[["auc", "n_support_vectors", "test_circuit_runs"]]
      .agg(["mean", "std"]).round(4).to_string())

# %% [markdown]
# ## 9. Qubit-width ablation (QSVM & VQC, first seed, fine-tuned embeddings)

# %%
abl = []
D = roc_store[key]["D"]
for nq in CFG["qubit_ablation"] + [CFG["n_qubits"]]:
    comp = Compressor(nq).fit(D["X_fit"])
    Af, Av, At = (comp.transform(D[k]) for k in ("X_fit", "X_val", "X_test"))
    yf, yv, yt = (D[k].astype(int) for k in ("y_fit", "y_val", "y_test"))
    best = (-1, None)
    for bw in CFG["qsvm_grid"]["bandwidth"]:
        m = QSVM(nq, bw, 10.0).fit(Af, yf)
        a = roc_auc_score(yv, m.predict_proba(Av))
        best = max(best, (a, m), key=lambda t: t[0])
    c = CFG["vqc"]
    v, _ = train_torch(VQCModule(nq, c["layers"]), Af, yf, Av, yv, c["epochs"], c["lr"], c["batch"], c["patience"], 0)
    abl.append({"qubits": nq, "QSVM_auc": roc_auc_score(yt, best[1].predict_proba(At)),
                "QSVM_kernel_offdiag_std": best[1].kernel_stats["offdiag_std"],
                "VQC_auc": roc_auc_score(yt, torch_proba(v, At)),
                "PCA_variance_kept": float(comp.pca.explained_variance_ratio_.sum())})
abl = pd.DataFrame(abl).sort_values("qubits")
abl.to_csv(OUT / "stage2_qubit_ablation.csv", index=False)
print(abl.round(4).to_string(index=False))

json.dump({"config": CFG, "embedding_files": [Path(f).name for f in files]},
          open(OUT / "stage2_config.json", "w"), indent=2)
print("Saved:", sorted(p.name for p in OUT.iterdir() if p.name.startswith(("stage2", "fig_"))))
