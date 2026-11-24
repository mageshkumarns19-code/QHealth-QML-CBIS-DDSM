"""Publication figures (12) and tables (4) for the CBIS-DDSM hybrid QML study.

Inputs: Stage 1 (224 & 448 px), Stage 2 (224 & 448 px), Stage 3 outputs + per-patient predictions (make_preds.py).
Outputs: figures/FigN.{png,tif,pdf}, tables/TableN.{csv,tex} and tables/Tables.docx
Journal style: Liberation Sans (Arial metrics), 7-8 pt, no titles inside figures, panel letters (a), (b), ...
"""
import json, pickle, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
                             confusion_matrix)

ROOT = Path(__file__).resolve().parents[1]
S = ROOT   # result folders sit at the repository root
S1, S1_448 = S / "stage1_224px", S / "stage1_448px"
S2, S2_448 = S / "stage2_224px", S / "stage2_448px"
S3 = S / "stage3"
OUT = ROOT / "figures" / "out"; (OUT / "figures").mkdir(parents=True, exist_ok=True); (OUT / "tables").mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.3, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
    "legend.frameon": False, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
})
W2, W1 = 6.85, 3.31          # double / single column width (inches): 174 mm / 84 mm
INK, MUTED = "#222222", "#6b6b6b"

# fixed model -> colour mapping (validated categorical palette, 8 slots); quantum solid, classical dashed
MODELS = ["QSVM", "VQC", "QNN", "CNN end-to-end", "LogReg (full 1280-d)", "LogReg (6-PCA)", "RBF-SVM (6-PCA)", "MLP (6-PCA)"]
LABEL = {"QSVM": "QSVM", "VQC": "VQC", "QNN": "Hybrid QNN", "CNN end-to-end": "CNN (end-to-end)",
         "LogReg (full 1280-d)": "LogReg (1280-D)", "LogReg (6-PCA)": "LogReg (6-D)",
         "RBF-SVM (6-PCA)": "RBF-SVM (6-D)", "MLP (6-PCA)": "MLP (6-D)"}
COLOR = {"QSVM": "#2a78d6", "VQC": "#eb6834", "QNN": "#1baf7a", "CNN end-to-end": "#6250d6",
         "LogReg (full 1280-d)": "#e34948", "LogReg (6-PCA)": "#eda100", "RBF-SVM (6-PCA)": "#e87ba4",
         "MLP (6-PCA)": "#008300"}
QUANTUM = {"QSVM", "VQC", "QNN"}
style = lambda m: dict(color=COLOR[m], ls="-" if m in QUANTUM else (0, (4, 2)), lw=1.6 if m in QUANTUM else 1.1)


def save(fig, name):
    for ext, kw in (("png", {"dpi": 300}), ("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}), ("pdf", {})):
        fig.savefig(OUT / "figures" / f"{name}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)


def panel(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="left")


P = pickle.load(open(ROOT / "figures" / "preds_seed42_head0.pkl", "rb"))
yv, yt = P["y_val"], P["y_test"]
R = pd.read_csv(S2 / "stage2_results_all_seeds.csv")
R["emb_type"] = R.embedding.str.replace(r"_seed\d+$", "", regex=True)
F = R[R.emb_type == "finetuned"]


def thr_youden(y, p):
    f, t, th = roc_curve(y, p); return float(np.clip(th[np.argmax(t - f)], 0, 1))

# ------------------------------------------------------------------ Fig 1  architecture
def fig1():
    fig, ax = plt.subplots(figsize=(W2, 3.4)); ax.set_xlim(0, 100); ax.set_ylim(-3.5, 46); ax.axis("off")

    def box(x, y, w, h, title, sub, fc="white", ec=INK, lw=0.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center", fontsize=7.2, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=6, color=MUTED, linespacing=1.15)

    def arrow(x0, y0, x1, y1, ls="-", txt=None, tx=0, ty=0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=MUTED, ls=ls))
        if txt: ax.text((x0 + x1) / 2 + tx, (y0 + y1) / 2 + ty, txt, fontsize=5.8, color=MUTED, ha="center")

    ax.text(1, 44, "Classical front end", fontsize=6.8, color=MUTED, fontweight="bold")
    ax.text(15, 21.0, "Quantum head and clinical decision layer", fontsize=6.8, color=MUTED, fontweight="bold")
    y1, h = 30, 10
    box(1, y1, 20, h, "CBIS-DDSM ROI", "3,483 lesion crops\npatient-grouped split")
    box(26, y1, 20, h, "EfficientNet-B0", "fine-tuned, 224 px\n3 seeds")
    box(51, y1, 20, h, "Embedding", "1,280 features\n(global avg. pool)")
    box(76, y1, 23, h, "Compression", "standardise \u2192 PCA (6)\nangles in [0, \u03c0]")
    for a, b in ((21.5, 25.5), (46.5, 50.5), (71.5, 75.5)): arrow(a, y1 + h / 2, b, y1 + h / 2)
    y2 = 8
    box(1, y2, 24, h, "Quantum head (6 qubits)", "QSVM (ZZ fidelity kernel)\nVQC (re-uploading) \u00b7 QNN", fc="#eaf2fc", ec="#2a78d6", lw=1.2)
    box(30, y2, 18, h, "P(malignant)", "probability score")
    box(53, y2, 20, h, "Threshold", "tuned on validation\n(Youden / 95% sens.)")
    box(78, y2, 21, h, "Risk decision", "recall vs routine\nfollow-up")
    ax.plot([87.5, 87.5, 13], [y1 - 0.4, 25, 25], color=MUTED, lw=0.8, solid_capstyle="butt")
    arrow(13, 25.4, 13, y2 + h + 0.4)
    for a, b in ((25.5, 29.5), (48.5, 52.5), (73.5, 77.5)): arrow(a, y2 + h / 2, b, y2 + h / 2)
    box(1, -3, 24, 8, "Noisy device", "depolarising \u00b7 IBM noise models\nibm_marrakesh (156 q)", fc="#fdf3e0", ec="#eda100", lw=1.2)
    box(50, -3, 26, 8, "On-device recalibration", "re-tune threshold on 20 patients")
    arrow(13, y2 - 0.3, 13, 5.4, ls="--")
    arrow(25.5, 1.0, 49.5, 1.0, txt="shifts scores (Proposition 1)", ty=0.9)
    arrow(63, 5.4, 63, y2 - 0.3)
    ax.text(64.2, 6.3, "restores operating point", fontsize=5.8, color=MUTED, ha="left", va="center")
    save(fig, "Fig1_architecture")

# ------------------------------------------------------------------ Fig 2  quantum circuits
def fig2():
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    n = 6; x = ParameterVector("x", n)
    zz = QuantumCircuit(n)
    for i in range(n): zz.h(i); zz.rz(2 * x[i], i)
    for i in list(range(0, n - 1, 2)) + list(range(1, n - 1, 2)): zz.rzz(2 * (np.pi - x[i]) * (np.pi - x[i + 1]), i, i + 1)
    v = QuantumCircuit(n, 1)
    ph, th, om = ParameterVector("\u03c6", n), ParameterVector("\u03b8", n), ParameterVector("\u03c9", n)
    for i in range(n): v.ry(x[i], i)
    v.barrier()
    for i in range(n): v.rz(ph[i], i); v.ry(th[i], i); v.rz(om[i], i)
    v.barrier()
    for i in range(n): v.cx(i, (i + 1) % n)
    v.barrier()
    v.measure(0, 0)
    imgs = []
    for qc, nm in ((zz, "zz"), (v, "vqc")):
        f = qc.draw("mpl", fold=-1, style={"fontsize": 9, "subfontsize": 7, "backgroundcolor": "#ffffff",
                                            "linecolor": "#222222", "textcolor": "#222222", "gatefacecolor": "#eaf2fc",
                                            "gatetextcolor": "#222222"})
        p = OUT / f"_tmp_{nm}.png"; f.savefig(p, dpi=400, bbox_inches="tight"); plt.close(f); imgs.append(plt.imread(p)); p.unlink()
    fig, axs = plt.subplots(2, 1, figsize=(W2, 5.2), gridspec_kw={"height_ratios": [1, 1.05]})
    for ax, im, l, cap in zip(axs, imgs, "ab", [r"ZZ feature map $U(\mathbf{x})$ (one of $r=2$ repetitions); kernel $k(\mathbf{x},\mathbf{x}')=|\langle 0|U^{\dagger}(\beta\mathbf{x}')\,U(\beta\mathbf{x})|0\rangle|^{2}$",
                                                  r"VQC block: $R_Y(x_i)$ re-upload $\rightarrow$ Rot$(\phi_i,\theta_i,\omega_i)$ $\rightarrow$ CNOT ring (repeated $L=3$ times); readout $\langle Z_0\rangle$"]):
        ax.imshow(im); ax.axis("off"); ax.text(0, 1.02, f"({l})  {cap}", transform=ax.transAxes, fontsize=7, va="bottom")
    save(fig, "Fig2_quantum_circuits")

# ------------------------------------------------------------------ Fig 3  training curves
def fig3():
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.1))
    cols = {42: "#2a78d6", 43: "#eb6834", 44: "#1baf7a"}
    for s in (42, 43, 44):
        h = pd.read_csv(S1 / f"history_seed{s}.csv")
        axs[0].plot(h.epoch, h.train_loss, color=cols[s], label=f"seed {s}")
        axs[1].plot(h.epoch, h.val_auc, color=cols[s], label=f"seed {s}")
        b = h.val_auc.idxmax(); axs[1].plot(h.epoch[b], h.val_auc[b], "o", ms=4, color=cols[s], mec="white", mew=0.8)
    axs[0].set(xlabel="Epoch", ylabel="Training loss (weighted BCE)"); axs[1].set(xlabel="Epoch", ylabel="Validation ROC-AUC")
    axs[0].legend(); hd, lb = axs[1].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    hd.append(Line2D([], [], ls="none", marker="o", ms=4, color=MUTED, mec="white")); lb.append("best epoch")
    axs[1].legend(hd, lb, loc="lower right")
    for k, nm in (("VQCModule", "VQC"), ("QNNModule", "QNN")):
        h = P["hist"][k]; axs[2].plot(np.arange(1, len(h) + 1), h, **{**style(nm), "ls": "-"}, label=LABEL[nm])
    axs[2].set(xlabel="Epoch", ylabel="Validation loss (weighted BCE)"); axs[2].legend()
    for a, l in zip(axs, "abc"): panel(a, l, x=-0.22)
    fig.tight_layout(w_pad=1.6); save(fig, "Fig3_training_curves")

# ------------------------------------------------------------------ Fig 4 ROC, Fig 5 PR
def fig4_5():
    sub = F[(F.embedding == "finetuned_seed42") & (F.seed == 0)].set_index("model")
    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(W1, 3.0))
        for m in MODELS:
            p = P["preds"][m][1]
            if kind == "roc":
                f, t, _ = roc_curve(yt, p)
                ax.plot(f, t, **style(m), label=f"{LABEL[m]}  {roc_auc_score(yt, p):.3f} [{sub.loc[m,'auc_lo']:.3f}\u2013{sub.loc[m,'auc_hi']:.3f}]")
            else:
                pr, rc, _ = precision_recall_curve(yt, p)
                ax.plot(rc, pr, **style(m), label=f"{LABEL[m]}  AP {average_precision_score(yt, p):.3f}")
        if kind == "roc":
            ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.7, ls=":"); ax.set(xlabel="1 \u2212 specificity", ylabel="Sensitivity")
            ax.legend(title="Model  AUC [95% CI]", title_fontsize=6.3, loc="lower right", handlelength=2.2)
        else:
            ax.axhline(yt.mean(), color="#bbbbbb", lw=0.7, ls=":"); ax.text(0.98, yt.mean() - 0.012, f"prevalence {yt.mean():.2f}", fontsize=6, color=MUTED, ha="right", va="top")
            ax.set(xlabel="Recall (sensitivity)", ylabel="Precision (PPV)", ylim=(0.3, 1.02)); ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2, fontsize=5.8, handlelength=2.0, columnspacing=1.0)
        ax.set_xlim(0, 1); ax.set_aspect("auto")
        save(fig, "Fig4_roc_curves" if kind == "roc" else "Fig5_precision_recall")

# ------------------------------------------------------------------ Fig 6 confusion matrices
def fig6():
    fig, axs = plt.subplots(2, 4, figsize=(W2, 3.9))
    for ax, m in zip(axs.ravel(), MODELS):
        pv, pt = P["preds"][m]; t = thr_youden(yv, pv)
        cm = confusion_matrix(yt, (pt >= t).astype(int), labels=[0, 1]); cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n{100 * cmn[i, j]:.0f}%", ha="center", va="center", fontsize=6.6,
                        color="white" if cmn[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ["Benign", "Malignant"]); ax.set_yticks([0, 1], ["Benign", "Malignant"], rotation=90, va="center")
        ax.grid(False); ax.set_title(f"{LABEL[m]}", fontsize=7, color=COLOR[m] if False else INK, pad=3)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.tick_params(length=0)
    fig.supxlabel("Predicted (threshold tuned on validation, Youden)", fontsize=7); fig.supylabel("True label", fontsize=7)
    fig.tight_layout(h_pad=2.4, w_pad=0.6); save(fig, "Fig6_confusion_matrices")

# ------------------------------------------------------------------ Fig 7 forest plot (15 runs)
def fig7():
    fig, ax = plt.subplots(figsize=(W1 * 1.25, 2.9))
    order = F.groupby("model").auc.mean().reindex(MODELS).sort_values().index.tolist()
    cnn = F[F.model == "CNN end-to-end"].auc.mean()
    ax.axvline(cnn, color=COLOR["CNN end-to-end"], lw=0.8, ls="--"); ax.text(cnn, len(order) - 0.35, " CNN mean", fontsize=6, color=MUTED, va="bottom")
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        d = F[F.model == m]
        ax.hlines(i, d.auc_lo.mean(), d.auc_hi.mean(), color=COLOR[m], lw=1.6 if m in QUANTUM else 1.1)
        ax.scatter(d.auc, i + rng.uniform(-0.18, 0.18, len(d)), s=7, color=COLOR[m], alpha=0.45, lw=0, zorder=2)
        ax.plot(d.auc.mean(), i, "D" if m in QUANTUM else "o", ms=5, color=COLOR[m], mec="white", mew=0.8, zorder=3)
        sig = (d.delong_p_vs_ref < 0.05).sum() if m != "CNN end-to-end" else None
        ax.text(0.842, i, "ref." if sig is None else f"{sig}/15", va="center", fontsize=6.3, color=MUTED)
    ax.text(0.842, len(order) - 0.35, "p<0.05", fontsize=6, color=MUTED, va="bottom")
    ax.set_yticks(range(len(order)), [LABEL[m] for m in order]); ax.set_xlim(0.68, 0.86); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Test ROC-AUC (dots: 15 runs; bar: mean 95% patient-cluster bootstrap CI)")
    save(fig, "Fig7_auc_forest_plot")

# ------------------------------------------------------------------ Fig 8 calibration, Fig 9 DCA
def fig8_9():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
    sub = F[(F.embedding == "finetuned_seed42") & (F.seed == 0)].set_index("model")
    bins = np.linspace(0, 1, 11)
    for m in MODELS:
        p = P["preds"][m][1]; idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() >= 8]; ys = [yt[idx == b].mean() for b in range(10) if (idx == b).sum() >= 8]
        axs[0].plot(xs, ys, marker="o", ms=2.8, **style(m), label=f"{LABEL[m]} (Brier {sub.loc[m,'brier']:.3f})")
    axs[0].plot([0, 1], [0, 1], color="#bbbbbb", lw=0.7, ls=":"); axs[0].set(xlabel="Predicted P(malignant)", ylabel="Observed frequency", xlim=(0, 1), ylim=(0, 1))
    axs[0].legend(fontsize=5.8, loc="upper left", handlelength=2.0)
    pts = np.linspace(0.05, 0.6, 56); prev = yt.mean()
    for m in MODELS:
        p = P["preds"][m][1]; n = len(yt)
        nb = [((p >= x) & (yt == 1)).sum() / n - ((p >= x) & (yt == 0)).sum() / n * x / (1 - x) for x in pts]
        axs[1].plot(pts, nb, **style(m), label=LABEL[m])
    axs[1].plot(pts, prev - (1 - prev) * pts / (1 - pts), color=INK, lw=0.8, label="Treat all"); axs[1].axhline(0, color="#999999", lw=0.8, label="Treat none")
    axs[1].set(xlabel="Threshold probability", ylabel="Net benefit", ylim=(-0.03, prev + 0.03)); axs[1].legend(fontsize=5.8, ncol=1, loc="upper right")
    panel(axs[0], "a", x=-0.17); panel(axs[1], "b", x=-0.19)
    fig.tight_layout(w_pad=2); save(fig, "Fig8_calibration_decision_curves")

# ------------------------------------------------------------------ Fig 9 noise (ranking vs decision, recovery)
def fig9():
    N = pd.read_csv(S2 / "stage2_noise_recalibration_all_seeds.csv")
    fig, axs = plt.subplots(2, 2, figsize=(W2, 4.9), sharex=True)
    c = {"auc": "#2a78d6", "sens@clean_thr": "#eb6834", "spec@clean_thr": "#6250d6"}
    lab = {"auc": "ROC-AUC", "sens@clean_thr": "Sensitivity (noiseless threshold)", "spec@clean_thr": "Specificity (noiseless threshold)"}
    for r, mdl in enumerate(["VQC", "QSVM"]):
        g = N[N.model == mdl].groupby("p"); M, SD = g.mean(numeric_only=True), g.std(numeric_only=True)
        for col in ("auc", "sens@clean_thr", "spec@clean_thr"):
            axs[r, 0].plot(M.index, M[col], marker="o", ms=3, color=c[col], label=lab[col])
            axs[r, 0].fill_between(M.index, M[col] - SD[col], M[col] + SD[col], color=c[col], alpha=0.15, lw=0)
        for k, cc in zip((20, 50, 100), ("#1baf7a", "#eda100", "#e87ba4")):
            axs[r, 1].plot(M.index, M[f"spec@recal{k}"], marker="^", ms=3, color=cc, label=f"Specificity, recalibrated (k = {k})")
            axs[r, 1].plot(M.index, M[f"sens@recal{k}"], marker="o", ms=3, color=cc, ls=(0, (4, 2)), label=f"Sensitivity, recalibrated (k = {k})")
        axs[r, 1].plot(M.index, M["spec@clean_thr"], color=INK, lw=0.9, ls=":", label="Specificity, no recalibration")
        axs[r, 1].axhline(0.95, color="#bbbbbb", lw=0.7)
        for a in axs[r]: a.set_ylim(0, 1.03); a.set_ylabel(f"{mdl} \u2013 value")
    for a in axs[1]: a.set_xlabel("Depolarising probability p (per qubit per layer), 1,024 shots")
    for a, l in zip(axs.ravel(), "abcd"): panel(a, l, x=-0.15)
    fig.tight_layout(h_pad=1.0, w_pad=1.6, rect=(0, 0.1, 1, 1))
    h0, l0 = axs[0, 0].get_legend_handles_labels(); h1, l1 = axs[0, 1].get_legend_handles_labels()
    fig.legend(h0, l0, loc="lower left", bbox_to_anchor=(0.06, 0.0), fontsize=5.8, ncol=1)
    fig.legend(h1, l1, loc="lower left", bbox_to_anchor=(0.53, 0.0), fontsize=5.8, ncol=2, columnspacing=1.2)
    save(fig, "Fig9_noise_ranking_vs_decision")

# ------------------------------------------------------------------ Fig 10 device noise + real hardware
def fig10():
    D = pd.read_csv(S3 / "stage3_device_noise_all_seeds.csv"); H = pd.read_csv(S3 / "stage3_ibm_hardware.csv")
    rows = []
    for cnd, lab in (("exact (simulator)", "Exact simulator"), ("FakeTorino noise model", "IBM Heron r1\nnoise model"),
                     ("FakeFez noise model", "IBM Heron r2\nnoise model")):
        d = D[D.condition == cnd]; rows.append((lab, d.mean(numeric_only=True), d.std(numeric_only=True), "704 test"))
    for _, h in H.iterrows():
        lab = "Exact, same\nsubset" if h.condition.startswith("exact") else "ibm_marrakesh\n(real hardware)"
        rows.append((lab, h, None, "150 test"))
    metrics = [("auc", "ROC-AUC", "#2a78d6"), ("spec@clean_thr", "Specificity, noiseless threshold", "#6250d6"),
               ("spec@recal20", "Specificity, recalibrated (k = 20)", "#1baf7a"), ("sens@recal20", "Sensitivity, recalibrated (k = 20)", "#eda100")]
    fig, ax = plt.subplots(figsize=(W2, 2.6)); x = np.arange(len(rows)); w = 0.19
    for j, (col, lab, cc) in enumerate(metrics):
        vals = [r[1][col] for r in rows]; err = [0 if r[2] is None else r[2][col] for r in rows]
        bars = ax.bar(x + (j - 1.5) * w, vals, w * 0.92, yerr=err, color=cc, label=lab, error_kw={"lw": 0.6, "capsize": 1.5})
        if col == "spec@clean_thr":
            for xi, v, e in zip(x + (j - 1.5) * w, vals, err): ax.text(xi, v + e + 0.02, f"{v:.2f}", ha="center", fontsize=5.6)
    ax.axvline(2.5, color="#999999", lw=0.6, ls=":")
    ax.text(1, 1.05, "All patients, 5 seeds (mean \u00b1 SD)", ha="center", fontsize=6.2, color=MUTED)
    ax.text(3.5, 1.05, "Stratified subset, seed 0", ha="center", fontsize=6.2, color=MUTED)
    ax.set_xticks(x, [r[0] for r in rows]); ax.set_ylim(0, 1.12); ax.set_ylabel("Value"); ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6)
    save(fig, "Fig10_device_noise_real_hardware")

# ------------------------------------------------------------------ Fig 11 QSVM shot scaling + qubit ablation
def fig11():
    sh = pd.read_csv(S2 / "stage2_qsvm_shot_scaling.csv"); ab = pd.read_csv(S2 / "stage2_qubit_ablation.csv")
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.2), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    ex = sh[sh.shots == "exact"].auc.mean(); s = sh[sh.shots != "exact"].copy(); s["shots"] = s.shots.astype(float)
    g = s.groupby("shots"); M, SD = g.auc.mean(), g.auc.std()
    axs[0].errorbar(M.index, M, yerr=SD, marker="o", ms=3.5, color=COLOR["QSVM"], capsize=2, lw=1.4)
    axs[0].axhline(ex, color=MUTED, ls="--", lw=0.8); axs[0].text(1100, ex + 0.006, f"exact kernel {ex:.3f}", fontsize=6, color=MUTED)
    runs = g.test_circuit_runs.mean()
    for k, (xv, yv_, r) in enumerate(zip(M.index, M, runs)):
        e = int(np.floor(np.log10(r))); t = f"{r / 10**e:.0f}$\\times$10$^{{{e}}}$ runs"
        if k < 2: axs[0].annotate(t, (xv, yv_), xytext=(6, -3), textcoords="offset points", fontsize=5.4, ha="left", va="top", color=MUTED)
        else: axs[0].annotate(t, (xv, yv_), xytext=(0, -6), textcoords="offset points", fontsize=5.4, ha="right" if k == len(runs) - 1 else "center", va="top", color=MUTED)
    axs[0].set(xscale="log", xlabel="Shots per kernel entry (no gate noise)", ylabel="QSVM test ROC-AUC", ylim=(0.56, 0.8))
    axs[1].plot(ab.qubits, ab.QSVM_auc, marker="D", ms=3.5, **style("QSVM"), label="QSVM")
    axs[1].plot(ab.qubits, ab.VQC_auc, marker="D", ms=3.5, **style("VQC"), label="VQC")
    axs[1].set(xlabel="Qubits", ylabel="Test ROC-AUC", xticks=[4, 6, 8], ylim=(0.7, 0.8)); axs[1].legend()
    axs[2].plot(ab.qubits, ab.QSVM_kernel_offdiag_std, marker="o", ms=3.5, color=COLOR["QSVM"])
    axs[2].set(xlabel="Qubits", ylabel="SD of off-diagonal kernel values", xticks=[4, 6, 8], ylim=(0, 0.3))
    for a, l in zip(axs, "abc"): panel(a, l, x=-0.24)
    fig.tight_layout(w_pad=1.4); save(fig, "Fig11_qsvm_shots_qubit_ablation")

# ------------------------------------------------------------------ Fig 12 feature space + kernel
def fig12():
    import pennylane as qml
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    z = np.load(S1 / "embeddings_finetuned_seed42.npz", allow_pickle=True)
    sc = StandardScaler().fit(z["X_fit"]); pca = PCA(6, random_state=0).fit(sc.transform(z["X_fit"]))
    Pt = pca.transform(sc.transform(z["X_test"])); y = z["y_test"].astype(int)
    Pf = pca.transform(sc.transform(z["X_fit"])); lo, hi = np.quantile(Pf, 0.01, 0), np.quantile(Pf, 0.99, 0)
    A = np.pi * np.clip((Pt - lo) / (hi - lo), 0, 1)
    dev = qml.device("default.qubit", wires=6)
    @qml.qnode(dev)
    def st(x):
        for _ in range(2):
            for i in range(6): qml.Hadamard(i); qml.RZ(2 * x[..., i], wires=i)
            for i in range(5): qml.IsingZZ(2 * (np.pi - x[..., i]) * (np.pi - x[..., i + 1]), wires=[i, i + 1])
        return qml.state()
    rng = np.random.default_rng(0); idx = np.concatenate([rng.choice(np.where(y == 0)[0], 100, False), rng.choice(np.where(y == 1)[0], 100, False)])
    Sv = np.asarray(st(0.05 * A[idx])).reshape(len(idx), -1); K = np.abs(Sv @ Sv.conj().T) ** 2
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7), gridspec_kw={"width_ratios": [1, 1.1]})
    for cls, cc, lab in ((0, "#2a78d6", "Benign"), (1, "#e34948", "Malignant")):
        m = y == cls; axs[0].scatter(Pt[m, 0], Pt[m, 1], s=5, color=cc, alpha=0.55, lw=0, label=lab)
    ev = pca.explained_variance_ratio_
    axs[0].set(xlabel=f"PC1 ({100*ev[0]:.1f}% var.) \u2192 qubit 0", ylabel=f"PC2 ({100*ev[1]:.1f}% var.) \u2192 qubit 1"); axs[0].legend(markerscale=2)
    im = axs[1].imshow(K, cmap="Blues", vmin=K.min(), vmax=1); axs[1].grid(False)
    axs[1].axhline(99.5, color="white", lw=0.8); axs[1].axvline(99.5, color="white", lw=0.8)
    axs[1].set_xticks([50, 150], ["Benign", "Malignant"]); axs[1].set_yticks([50, 150], ["Benign", "Malignant"], rotation=90, va="center")
    axs[1].tick_params(length=0); cb = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.03); cb.set_label("Quantum kernel k(x, x\u2032)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    for sp in axs[1].spines.values(): sp.set_visible(False)
    panel(axs[0], "a", x=-0.18); panel(axs[1], "b", x=-0.12)
    fig.tight_layout(w_pad=2); save(fig, "Fig12_feature_space_quantum_kernel")

# ------------------------------------------------------------------ Tables
def tables():
    import docx
    from docx.shared import Pt
    doc = docx.Document(); st = doc.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(9)

    def emit(df, name, caption):
        df.to_csv(OUT / "tables" / f"{name}.csv", index=False)
        (OUT / "tables" / f"{name}.tex").write_text(
            "% requires \\usepackage{booktabs}\n\\begin{table}[t]\n\\caption{" + caption + "}\n\\label{tab:" + name.split("_")[0].lower() + "}\n"
            + df.to_latex(index=False, escape=True).replace("\\toprule", "\\toprule").strip() + "\n\\end{table}\n")
        doc.add_paragraph().add_run(f"{name.split('_')[0].replace('Table', 'Table ')}  {caption}").bold = True
        t = doc.add_table(rows=1, cols=len(df.columns)); t.style = "Light List"
        for j, c in enumerate(df.columns): t.rows[0].cells[j].text = str(c)
        for _, r in df.iterrows():
            cells = t.add_row().cells
            for j, v in enumerate(r): cells[j].text = str(v)
        doc.add_paragraph()

    sp = pd.read_csv(S1 / "split_seed42.csv")
    rows = []
    for s in ("fit", "val", "test"):
        for ab in ("mass", "calc", "all"):
            d = sp[sp.split == s] if ab == "all" else sp[(sp.split == s) & (sp.abnormality == ab)]
            rows.append({"Split": {"fit": "Training (fit)", "val": "Validation", "test": "Test (official)"}[s],
                         "Lesion type": {"mass": "Mass", "calc": "Calcification", "all": "All"}[ab],
                         "ROIs": len(d), "Malignant": int(d.label.sum()), "Benign": int((d.label == 0).sum()),
                         "Malignant (%)": f"{100 * d.label.mean():.1f}", "Patients": d.patient_id.nunique()})
    emit(pd.DataFrame(rows), "Table2_dataset_splits",
         "CBIS-DDSM lesion-level splits (CNN seed 42). Splits are grouped by patient; the official test set is unchanged.")

    hp = pd.DataFrame([
        ("Front end", "EfficientNet-B0 (ImageNet init.)", "224 \u00d7 224 input, dropout 0.3, AdamW lr 1e-4, wd 1e-4, cosine, batch 32, \u226430 epochs, early stop on val. AUC (patience 7), AMP, 3 seeds"),
        ("Augmentation", "Train / test time", "Random resized crop (0.8\u20131.0), h/v flips, \u00b115\u00b0 rotation, brightness/contrast jitter / horizontal-flip TTA"),
        ("Compression", "Standardise \u2192 PCA \u2192 angles", "6 components (4, 8 ablation); scaled to [0, \u03c0] using fit-set 1st\u201399th percentiles"),
        ("QSVM", "Fidelity kernel, ZZ map", "r = 2, linear entanglement; bandwidth \u03b2 \u2208 {0.02, 0.05, 0.1, 0.2}, C \u2208 {1, 10} (val. AUC); Platt scaling"),
        ("VQC", "Re-uploading PQC", "6 qubits, L = 3 strongly entangling blocks, \u27e8Z\u2080\u27e9 \u2192 affine logit; 54 + 2 params; Adam lr 0.05, batch 32, \u226460 epochs, patience 12"),
        ("Hybrid QNN", "Encoder \u2192 PQC \u2192 head", "PCA-32 \u2192 16 \u2192 6 (tanh) \u2192 2 entangling layers \u2192 \u27e8Z\u1d62\u27e9 \u2192 linear; Adam lr 0.02"),
        ("Classical heads", "LogReg, RBF-SVM, MLP (6-D)", "C / \u03b1 grids tuned on val. AUC; LogReg on 1,280-D and CNN head as references"),
        ("Evaluation", "Test set (704 ROIs)", "AUC with 1,000 patient-cluster bootstrap CI; DeLong vs CNN; thresholds tuned on validation; Brier, ECE, calibration slope, net benefit"),
        ("Noise", "Deployment", "Depolarising p \u2208 {0\u20130.1}, 1,024 shots; IBM Heron noise models (FakeTorino, FakeFez); ibm_marrakesh hardware; recalibration k \u2208 {20, 50, 100}"),
        ("Software", "Versions", "PyTorch 2, torchvision, PennyLane 0.4x, Qiskit 2.x, Qiskit Aer, qiskit-ibm-runtime, scikit-learn; Kaggle T4 GPU / CPU"),
    ], columns=["Component", "Setting", "Details"])
    emit(hp, "Table3_implementation_details", "Implementation details and hyper-parameters.")

    g = F.groupby("model")
    def ms(c, d=3): return pd.Series({k: f"{g[c].mean()[k]:.{d}f} \u00b1 {g[c].std()[k]:.{d}f}" for k in g[c].mean().index})
    T3 = pd.DataFrame({"Model": [LABEL[m] for m in MODELS], "Type": ["Quantum" if m in QUANTUM else "Classical" for m in MODELS]})
    for c, nm in (("auc", "AUC"), ("sens@youden", "Sensitivity"), ("spec@youden", "Specificity"),
                  ("sens@ruleout(val sens 0.95)", "Sens. @ rule-out"), ("spec@ruleout(val sens 0.95)", "Spec. @ rule-out"),
                  ("brier", "Brier"), ("ece", "ECE"), ("calib_slope", "Calib. slope"), ("NB@0.2", "Net benefit @0.2")):
        T3[nm] = [ms(c).get(m) for m in MODELS]
    T3.insert(3, "95% CI (mean)", [f"{g.auc_lo.mean()[m]:.3f}\u2013{g.auc_hi.mean()[m]:.3f}" for m in MODELS])
    T3["DeLong p<0.05 vs CNN"] = ["ref." if m == "CNN end-to-end" else f"{(F[F.model == m].delong_p_vs_ref < 0.05).sum()}/15" for m in MODELS]
    emit(T3, "Table4_main_performance",
         "Test-set performance on CBIS-DDSM (mean \u00b1 SD over 3 CNN seeds \u00d7 5 head seeds). Thresholds tuned on validation (Youden; rule-out = 95% validation sensitivity).")

    R448 = pd.read_csv(S2_448 / "stage2_results_all_seeds.csv"); R448["emb_type"] = R448.embedding.str.replace(r"_seed\d+$", "", regex=True)
    a224, a448 = F.groupby("model").auc.mean(), R448[R448.emb_type == "finetuned"].groupby("model").auc.mean()
    img = R[R.emb_type == "imagenet"].groupby("model").auc.mean()
    ab = pd.read_csv(S2 / "stage2_qubit_ablation.csv").set_index("qubits")
    rows = []
    for m in MODELS:
        rows.append({"Model": LABEL[m], "Fine-tuned 224 px": f"{a224[m]:.3f}", "Fine-tuned 448 px": f"{a448[m]:.3f}",
                     "Frozen ImageNet": "\u2013" if m == "CNN end-to-end" else f"{img[m]:.3f}",
                     "4 qubits": f"{ab.loc[4, 'QSVM_auc']:.3f}" if m == "QSVM" else (f"{ab.loc[4, 'VQC_auc']:.3f}" if m == "VQC" else ""),
                     "8 qubits": f"{ab.loc[8, 'QSVM_auc']:.3f}" if m == "QSVM" else (f"{ab.loc[8, 'VQC_auc']:.3f}" if m == "VQC" else "")})
    emit(pd.DataFrame(rows), "Table5_ablations",
         "Ablations: mean test AUC by input resolution, front end (fine-tuned vs frozen ImageNet features) and qubit count (QSVM and VQC, CNN seed 42, head seed 0; 6 qubits is the main setting).")
    doc.save(OUT / "tables" / "Tables_2-5.docx")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["fig1", "fig2", "fig3", "fig4_5", "fig6", "fig7", "fig8_9", "fig9", "fig10", "fig11", "fig12", "tables"]
    for f in todo:
        globals()[f](); print("done", f, flush=True)
