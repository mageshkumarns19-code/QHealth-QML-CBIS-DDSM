# %% [markdown]
# # Stage 3 — VQC on IBM device noise models (and optionally on real IBM Quantum hardware)
#
# Stage 2 used an idealised depolarising model. Here the **trained VQC** is executed with
# **calibrated noise models of real IBM processors** (Qiskit Aer + `fake_provider`: gate errors,
# readout errors, T1/T2 decoherence, native gates and qubit connectivity after transpilation).
# Optionally, a stratified subset runs on a **real IBM Quantum backend**.
#
# For each condition: test ROC-AUC, sensitivity/specificity at the **noiseless** validation-tuned
# rule-out threshold, and after **recalibration** on k validation patients executed on the same backend.
#
# **Input:** Stage 1 output (224 px run), the file `embeddings_finetuned_seed42.npz`.
# **Settings:** Accelerator = None · Internet = On.
# **Optional real hardware:** create a free IBM Quantum account (quantum.cloud.ibm.com), copy your API key,
# and add it in Kaggle under *Add-ons → Secrets* with the label `IBM_QUANTUM_TOKEN` (and your instance CRN as
# `IBM_QUANTUM_INSTANCE`); then set `RUN_IBM = True`.
#
# Runtime ≈ 15–30 min on Kaggle CPU (device-noise part); the real-hardware cell uses ≈ 1–3 min of QPU time.

# %%
import subprocess, sys
for pkg, mod in [("pennylane", "pennylane"), ("qiskit", "qiskit"), ("qiskit-aer", "qiskit_aer"),
                 ("qiskit-ibm-runtime", "qiskit_ibm_runtime")]:
    try:
        __import__(mod)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

import glob, json, os, time, warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2
from qiskit_ibm_runtime import fake_provider
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
print("PennyLane", qml.__version__)

# %% [markdown]
# ## 1. Configuration

# %%
QUICK = os.environ.get("QUICK") == "1"
CFG = dict(
    n_qubits=6, layers=3,
    seeds=[0, 1] if QUICK else [0, 1, 2, 3, 4],
    epochs=4 if QUICK else 60, lr=0.05, batch=32, patience=12,
    devices=["FakeTorino", "FakeFez"],           # IBM Heron r1 (133 q) and Heron r2 (156 q) noise models
    shots=1024,
    target_sensitivity=0.95,
    recal_sizes=[20, 50, 100],
    max_eval=60 if QUICK else None,               # cap patients per split in quick mode
    ibm_test_n=150, ibm_val_n=100,                # real-hardware subset sizes
)
RUN_IBM = os.environ.get("RUN_IBM") == "1"       # set True on Kaggle after adding the IBM_QUANTUM_TOKEN secret
IBM_LOCAL_TEST = os.environ.get("IBM_LOCAL_TEST") == "1"   # exercise the hardware code path on a fake backend
OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working")); OUT.mkdir(parents=True, exist_ok=True)

path = (glob.glob(os.environ.get("EMB_GLOB", "/kaggle/input/**/embeddings_finetuned_seed42.npz"), recursive=True) or [None])[0]
assert path, "embeddings_finetuned_seed42.npz not found - attach the Stage 1 (224 px) output."
summ = glob.glob(str(Path(path).parent / "stage1_summary.json"))
if summ:
    print("Stage 1 img_size:", json.load(open(summ[0]))["config"]["img_size"])
z = np.load(path, allow_pickle=True)
D = {k: z[k] for k in z.files}
if CFG["max_eval"]:
    for s in ("val", "test"):
        idx = np.random.default_rng(0).permutation(len(D[f"y_{s}"]))[:CFG["max_eval"]]
        for k in list(D):
            if k.endswith("_" + s):
                D[k] = D[k][idx]
print({k: v.shape for k, v in D.items() if k.startswith("X_")})

# %% [markdown]
# ## 2. Re-train the VQC exactly as in Stage 2 (same compression, circuit, optimiser, seeds)

# %%
class Compressor:
    def __init__(self, n): self.n = n
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.sc = StandardScaler().fit(X)
        self.pca = PCA(self.n, random_state=0).fit(self.sc.transform(X))
        P = self.pca.transform(self.sc.transform(X))
        self.lo, self.hi = np.quantile(P, 0.01, 0), np.quantile(P, 0.99, 0)
        return self
    def transform(self, X):
        P = self.pca.transform(self.sc.transform(np.asarray(X, dtype=np.float64)))
        return np.pi * np.clip((P - self.lo) / np.where(self.hi > self.lo, self.hi - self.lo, 1), 0, 1)


def vqc_circuit(x, w, n):
    for layer in range(w.shape[0]):
        for i in range(n):
            qml.RY(x[..., i], wires=i)
        qml.StronglyEntanglingLayers(w[layer:layer + 1], wires=range(n))


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


def train_vqc(Xf, yf, Xv, yv, seed):
    torch.manual_seed(seed)
    m = VQCModule(CFG["n_qubits"], CFG["layers"]).double()
    Xf, yf = torch.tensor(Xf), torch.tensor(yf, dtype=torch.float64)
    Xv, yv = torch.tensor(Xv), torch.tensor(yv, dtype=torch.float64)
    pw = torch.tensor((len(yf) - yf.sum().item()) / max(yf.sum().item(), 1), dtype=torch.float64)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.Adam(m.parameters(), lr=CFG["lr"])
    g = torch.Generator().manual_seed(seed)
    best, best_state, wait = np.inf, None, 0
    for _ in range(CFG["epochs"]):
        m.train()
        perm = torch.randperm(len(Xf), generator=g)
        for s in range(0, len(Xf), CFG["batch"]):
            idx = perm[s:s + CFG["batch"]]
            opt.zero_grad(); loss_fn(m(Xf[idx]), yf[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(loss_fn(m(Xv), yv))
        if vl < best - 1e-4:
            best, wait, best_state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            wait += 1
            if wait >= CFG["patience"]:
                break
    m.load_state_dict(best_state); m.eval()
    return m


comp = Compressor(CFG["n_qubits"]).fit(D["X_fit"])
Af, Av, At = (comp.transform(D[k]) for k in ("X_fit", "X_val", "X_test"))
yf, yv, yt = (D[k].astype(int) for k in ("y_fit", "y_val", "y_test"))

# %% [markdown]
# ## 3. The same circuit in Qiskit (verified identical to PennyLane) and its executors

# %%
def qiskit_vqc(weights, n):
    """RY(x) re-uploading + one StronglyEntanglingLayer per block (Rot = RZ·RY·RZ, CNOT ring range 1)."""
    x = ParameterVector("x", n)
    qc = QuantumCircuit(n, 1)
    for layer in range(weights.shape[0]):
        for i in range(n):
            qc.ry(x[i], i)
        for i in range(n):
            phi, th, om = (float(v) for v in weights[layer, i])
            qc.rz(phi, i); qc.ry(th, i); qc.rz(om, i)
        for i in range(n):
            qc.cx(i, (i + 1) % n)
    qc.measure(0, 0)
    return qc, x


def z_from_counts(counts, shots):
    return (counts.get("0", 0) - counts.get("1", 0)) / shots


def run_aer(qc_t, xparams, X, sim, shots, seed):
    """One transpiled circuit, one parameter set per patient (batched: ~30 ms per patient)."""
    col = {p.name: i for i, p in enumerate(xparams)}
    binds = [{p: [float(v) for v in X[:, col[p.name]]] for p in qc_t.parameters}]
    res = sim.run(qc_t, parameter_binds=binds, shots=shots, seed_simulator=seed).result()
    return np.array([z_from_counts(res.get_counts(j), shots) for j in range(len(X))])


def to_prob(z, m):
    return 1 / (1 + np.exp(-(-float(m.scale) * z + float(m.bias))))


def thr_sens(y, p, target):
    f, t, th = roc_curve(y, p)
    return float(np.clip(th[np.where(t >= target)[0][0]], 0, 1))


def sens_spec(y, p, t):
    yhat = p >= t
    return float(yhat[y == 1].mean()), float((~yhat[y == 0]).mean())


def evaluate_condition(name, seed, pv, pt, t_clean, yv_, yt_, rng):
    se, sp = sens_spec(yt_, pt, t_clean)
    row = {"condition": name, "seed": seed, "auc": roc_auc_score(yt_, pt), "mean_prob": float(pt.mean()),
           "sd_prob": float(pt.std()), "sens@clean_thr": se, "spec@clean_thr": sp}
    n_pos, n_neg = int((yv_ == 1).sum()), int((yv_ == 0).sum())
    for k in CFG["recal_sizes"]:
        if k > len(yv_) or n_pos < 2:
            row[f"sens@recal{k}"] = row[f"spec@recal{k}"] = np.nan
            continue
        ses, sps = [], []
        for _ in range(20):
            kp = min(n_pos, max(2, int(round(k * yv_.mean()))))
            idx = np.concatenate([rng.choice(np.where(yv_ == 1)[0], kp, replace=False),
                                  rng.choice(np.where(yv_ == 0)[0], min(n_neg, k - kp), replace=False)])
            a, b = sens_spec(yt_, pt, thr_sens(yv_[idx], pv[idx], CFG["target_sensitivity"]))
            ses.append(a); sps.append(b)
        row[f"sens@recal{k}"], row[f"spec@recal{k}"] = float(np.mean(ses)), float(np.mean(sps))
    return row

# %% [markdown]
# ## 4. Run: exact · ideal shots · IBM device noise models (every head seed)

# %%
backends = {name: getattr(fake_provider, name)() for name in CFG["devices"]}
sims = {"ideal (1024 shots)": AerSimulator()}
sims.update({f"{name} noise model": AerSimulator.from_backend(b) for name, b in backends.items()})

rows, circuit_info, models = [], [], {}
for seed in CFG["seeds"]:
    t0 = time.time()
    m = train_vqc(Af, yf, Av, yv, seed)
    models[seed] = m
    print(f"  seed {seed}: VQC trained in {time.time() - t0:.0f}s", flush=True)
    with torch.no_grad():
        pv0 = torch.sigmoid(m(torch.tensor(Av))).numpy(); pt0 = torch.sigmoid(m(torch.tensor(At))).numpy()
    t_clean = thr_sens(yv, pv0, CFG["target_sensitivity"])
    rng = np.random.default_rng(100 + seed)
    rows.append(evaluate_condition("exact (simulator)", seed, pv0, pt0, t_clean, yv, yt, rng))
    w = m.q.weights.detach().numpy()
    qc, xp = qiskit_vqc(w, CFG["n_qubits"])
    for name, sim in sims.items():
        target = backends.get(name.replace(" noise model", ""))
        qc_t = transpile(qc, backend=target, optimization_level=1, seed_transpiler=seed) if target else qc
        if target and seed == CFG["seeds"][0]:
            ops = qc_t.count_ops()
            circuit_info.append({"device": name, "depth": qc_t.depth(),
                                 "two_qubit_gates": int(sum(v for k, v in ops.items() if k in ("cx", "cz", "ecr"))),
                                 "ops": dict(ops)})
        t1 = time.time()
        zv = run_aer(qc_t, xp, Av, sim, CFG["shots"], seed); zt = run_aer(qc_t, xp, At, sim, CFG["shots"], seed + 7)
        print(f"  seed {seed}: {name} executed in {time.time() - t1:.0f}s", flush=True)
        rows.append(evaluate_condition(name, seed, to_prob(zv, m), to_prob(zt, m), t_clean, yv, yt, rng))
    print(f"seed {seed}: {time.time() - t0:.0f}s |", {r['condition']: round(r['auc'], 3) for r in rows if r['seed'] == seed})

res = pd.DataFrame(rows)
res.to_csv(OUT / "stage3_device_noise_all_seeds.csv", index=False)
mcols = [c for c in res.columns if c not in ("condition", "seed")]
order = list(dict.fromkeys(res.condition))
mean, sd = res.groupby("condition")[mcols].mean().loc[order], res.groupby("condition")[mcols].std().fillna(0).loc[order]
summary = (mean.round(3).astype(str) + " ± " + sd.round(3).astype(str)).reset_index()
summary.to_csv(OUT / "stage3_device_noise_summary.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(summary[["condition", "auc", "mean_prob", "sens@clean_thr", "spec@clean_thr", "sens@recal20", "spec@recal20",
               "sens@recal100", "spec@recal100"]].to_string(index=False))
print(pd.DataFrame(circuit_info)[["device", "depth", "two_qubit_gates"]].to_string(index=False))
json.dump({"config": CFG, "circuits": circuit_info}, open(OUT / "stage3_config.json", "w"), indent=2, default=str)

# %% [markdown]
# ## 5. Figure: ranking vs decision on IBM device noise models

# %%
x = np.arange(len(order)); wbar = 0.2
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
for j, (col, lab) in enumerate([("auc", "ROC-AUC"), ("sens@clean_thr", "sensitivity, noiseless thr"),
                                ("spec@clean_thr", "specificity, noiseless thr")]):
    ax[0].bar(x + (j - 1) * wbar, mean[col], wbar, yerr=sd[col], label=lab, capsize=3)
ax[0].set_xticks(x); ax[0].set_xticklabels(order, rotation=15, fontsize=8); ax[0].set_ylim(0, 1.05); ax[0].legend(fontsize=8)
for j, (col, lab) in enumerate([("sens@recal20", "sensitivity, recalibrated k=20"), ("spec@recal20", "specificity, recalibrated k=20")]):
    ax[1].bar(x + (j - 0.5) * wbar, mean[col], wbar, yerr=sd[col], label=lab, capsize=3)
ax[1].axhline(CFG["target_sensitivity"], color="grey", ls=":")
ax[1].set_xticks(x); ax[1].set_xticklabels(order, rotation=15, fontsize=8); ax[1].set_ylim(0, 1.05); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "fig_device_noise.png", dpi=160); plt.show()

# %% [markdown]
# ## 6. Optional: real IBM Quantum hardware (stratified subset, head seed 0)
# Uses one SamplerV2 job: a single parameterised circuit with one parameter set per patient.
# Requires the `IBM_QUANTUM_TOKEN` Kaggle secret and `RUN_IBM = True`. QPU time ≈ 1–3 min.

# %%
def stratified_subset(y, n, seed):
    r = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    kp = int(round(n * y.mean()))
    return np.sort(np.concatenate([r.choice(pos, min(kp, len(pos)), replace=False),
                                   r.choice(neg, min(n - kp, len(neg)), replace=False)]))


if RUN_IBM or IBM_LOCAL_TEST:
    seed = CFG["seeds"][0]; m = models[seed]
    vi, ti = stratified_subset(yv, CFG["ibm_val_n"], 1), stratified_subset(yt, CFG["ibm_test_n"], 2)
    if RUN_IBM:
        from qiskit_ibm_runtime import QiskitRuntimeService
        def secret(name):
            try:
                from kaggle_secrets import UserSecretsClient
                return UserSecretsClient().get_secret(name)
            except Exception:
                return os.environ.get(name)
        token, crn = secret("IBM_QUANTUM_TOKEN"), secret("IBM_QUANTUM_INSTANCE")   # API key; instance CRN (optional)
        assert token, "Add the IBM_QUANTUM_TOKEN secret (Add-ons -> Secrets) and attach it to this notebook."
        service = (QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=crn) if crn
                   else QiskitRuntimeService(channel="ibm_quantum_platform", token=token))
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=CFG["n_qubits"])
    else:
        backend = fake_provider.FakeTorino()          # local dry run of the exact same code path
    print("hardware backend:", backend.name)
    qc, xp = qiskit_vqc(m.q.weights.detach().numpy(), CFG["n_qubits"])
    qc_t = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=0)
    Xall = np.vstack([Av[vi], At[ti]])
    # parameters of the transpiled circuit are ordered x[0]..x[5], matching the columns of Xall
    assert [p.name for p in qc_t.parameters] == [f"x[{i}]" for i in range(CFG["n_qubits"])]
    sampler = SamplerV2(mode=backend)
    job = sampler.run([(qc_t, Xall)], shots=CFG["shots"])
    print("job id:", job.job_id())
    bits = job.result()[0].data.c.array            # uint8, shape (n_patients, shots, 1); bit 0 = measured qubit
    ones = (bits[..., -1] & 1).astype(np.int64).sum(axis=1)   # int64: avoid unsigned wrap-around
    zhw = (CFG["shots"] - 2 * ones) / CFG["shots"]
    pv_hw, pt_hw = to_prob(zhw[:len(vi)], m), to_prob(zhw[len(vi):], m)
    with torch.no_grad():
        pv0 = torch.sigmoid(m(torch.tensor(Av))).numpy(); pt0 = torch.sigmoid(m(torch.tensor(At))).numpy()
    t_clean = thr_sens(yv, pv0, CFG["target_sensitivity"])
    rng = np.random.default_rng(7)
    hw = [evaluate_condition(f"exact, same subset", seed, pv0[vi], pt0[ti], t_clean, yv[vi], yt[ti], rng),
          evaluate_condition(f"{backend.name} ({'hardware' if RUN_IBM else 'local fake'})", seed, pv_hw, pt_hw,
                             t_clean, yv[vi], yt[ti], rng)]
    hw = pd.DataFrame(hw)
    hw.to_csv(OUT / "stage3_ibm_hardware.csv", index=False)
    print(hw[["condition", "auc", "mean_prob", "sens@clean_thr", "spec@clean_thr", "sens@recal20", "spec@recal20"]]
          .round(3).to_string(index=False))
else:
    print("Real-hardware run skipped (set RUN_IBM = True after adding the IBM_QUANTUM_TOKEN secret).")

print("Saved:", sorted(p.name for p in OUT.iterdir() if p.name.startswith(("stage3", "fig_device"))))
