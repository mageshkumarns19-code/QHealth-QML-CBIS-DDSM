# Hybrid Quantum Classifiers for Mammographic Breast Cancer Diagnosis under Hardware Noise

Code, notebooks and outputs for the article

> **Stable Ranking, Shifting Decisions: Hybrid Quantum Classifiers for Mammographic Breast Cancer Diagnosis under Hardware Noise**
> Mageshkumar N, Saravanan Srinivasan, ArunPriya N. *Quantum Machine Intelligence* (submitted, 2026).

The study compares three hybrid quantum classifiers (QSVM, VQC, hybrid QNN) with input-matched classical models for
benign-versus-malignant classification of CBIS-DDSM lesions, and tests what happens to their clinical operating point
under depolarising noise, IBM Heron device noise models and the real `ibm_marrakesh` processor.

## Main results

| | Test AUC (15 runs) | Notes |
|---|---|---|
| CNN (EfficientNet-B0, end-to-end) | 0.777 ± 0.006 | reference |
| QSVM (6 qubits) | 0.773 ± 0.012 | never significantly different from the CNN (0/15 DeLong tests) |
| Logistic regression (same 6 features) | 0.779 ± 0.009 | parity, not quantum advantage |
| VQC (6 qubits) | 0.746 ± 0.015 | significantly lower in 13/15 runs |

* **VQC is decision-fragile.** On `ibm_marrakesh`, AUC changed by 0.012 but specificity at the validation-tuned
  screening threshold fell from 0.25 to 0.04; re-tuning the threshold on 20 patients run on the device raised it to 0.48.
* **QSVM is estimation-fragile.** With 1,024 shots per kernel entry, AUC fell to 0.625; recovering simulator accuracy
  needed about 65,536 shots per entry (≈ 4 × 10¹⁰ circuit executions for the test set).

## Repository structure

```
QHealth-QML-CBIS-DDSM/
├── notebooks/                       Kaggle notebooks (run these to reproduce everything)
│   ├── 01_stage1_cnn_embeddings.ipynb   GPU: fine-tune EfficientNet-B0, export 1,280-D embeddings
│   ├── 02_stage2_qml_heads.ipynb        CPU: quantum + classical heads, evaluation, noise study
│   └── 03_stage3_device_noise.ipynb     CPU: IBM Heron noise models, optional real IBM hardware run
├── src/                             The same three stages as plain Python scripts
├── figures/
│   ├── make_preds.py                Saves per-patient predictions used by Figs. 3–6 and 8
│   ├── make_figures.py              Regenerates Figs. 1–12 and Tables 2–5 from the result folders
│   └── png/                         The 12 figures as submitted
├── stage1_224px/                    Patient splits and training histories (embeddings: see Releases)
├── stage2_224px/                    All Stage 2 result tables (main results)
├── stage3/                          IBM noise-model and hardware results
├── stage1_448px/, stage2_448px/     Resolution ablation (results of our runs; inputs to make_figures.py)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

**Large files are on the Releases page.** Download the four embedding files from release
[v1.0.0](../../releases/tag/v1.0.0) (`embeddings_finetuned_seed42.npz`, `…seed43.npz`, `…seed44.npz`,
`embeddings_imagenet.npz`) and put them in `stage1_224px/` before running Level 1 or Level 2 below.
Trained CNN weights (`cnn_seed*.pt`) are not stored; Stage 1 recreates them.

## Data

CBIS-DDSM is **not** redistributed in this repository. Get it from either source:

* The Cancer Imaging Archive: https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
* Kaggle JPEG release used in this study: `awsaf49/cbis-ddsm-breast-cancer-image-dataset`

The files in `stage1_224px/` contain only derived embeddings, labels and the public CBIS-DDSM patient IDs.

## How to reproduce

Pick the level you need. Each level starts from the outputs of the one before, which are already in this repository.

### Level 1 — regenerate all figures and tables (CPU, about 15–30 minutes)

First download the embedding files from the release into `stage1_224px/` (see above).

```bash
pip install -r requirements.txt
python figures/make_preds.py          # re-fits the heads for CNN seed 42, head seed 0
python figures/make_figures.py        # writes figures/out/figures and figures/out/tables
```

### Level 2 — rerun Stage 2 from the saved embeddings (CPU, about 2–4 hours)

```bash
EMB_GLOB="stage1_224px/embeddings_*.npz" OUT_DIR=stage2_rerun python src/stage2_qml_heads.py
```

Compare `stage2_rerun/stage2_summary_pooled.csv` with `stage2_224px/stage2_summary_pooled.csv`.

### Level 3 — full pipeline on Kaggle (free GPU)

1. **Stage 1.** Create a Kaggle notebook from `notebooks/01_stage1_cnn_embeddings.ipynb`. Add the dataset
   `awsaf49/cbis-ddsm-breast-cancer-image-dataset`. Set Accelerator = GPU T4 and Internet = On. Choose
   *Save Version → Save & Run All*. Takes about 1–2 h for 3 seeds at 224 px.
2. **Stage 2.** Create a notebook from `02_stage2_qml_heads.ipynb`. Under *Add Input → Notebook Output*, add your
   committed Stage 1 notebook. Accelerator = None, Internet = On. *Save & Run All*.
3. **Stage 3.** Same as Stage 2 with `03_stage3_device_noise.ipynb`.

For the 448-pixel ablation, set `img_size=448` and `batch_size=24` in the Stage 1 configuration cell.

### Optional — real IBM Quantum hardware

1. Create a free account at https://quantum.cloud.ibm.com and generate an API key.
2. In Kaggle: *Add-ons → Secrets*. Add `IBM_QUANTUM_TOKEN` (your API key) and `IBM_QUANTUM_INSTANCE` (your instance
   CRN). Attach both to the Stage 3 notebook.
3. Set `RUN_IBM = True` in the configuration cell and run. It uses about 1–3 minutes of QPU time.

> **Never paste your API key into a notebook cell or commit it to GitHub.** The code reads it only from Kaggle Secrets
> or an environment variable.

## Figures and tables

| Item | Produced by | Data |
|---|---|---|
| Fig. 1 architecture, Fig. 2 circuits | `make_figures.py` | — |
| Figs. 3–6, 8 | `make_figures.py` after `make_preds.py` | `stage1_224px` |
| Fig. 7 AUC over 15 runs, Table 4 | `make_figures.py` | `stage2_224px/stage2_results_all_seeds.csv` |
| Fig. 9 noise | `make_figures.py` | `stage2_224px/stage2_noise_recalibration_all_seeds.csv` |
| Fig. 10 IBM noise models and hardware | `make_figures.py` | `stage3/` |
| Fig. 11 shots and qubits, Table 5 | `make_figures.py` | `stage2_224px/`, `stage2_448px/` |
| Fig. 12 feature space and kernel | `make_figures.py` | `stage1_224px/` |
| Table 2 splits, Table 3 settings | `make_figures.py` | `stage1_224px/split_seed42.csv` |

Table 1 (comparison with prior studies) appears only in the manuscript.

## Settings used

EfficientNet-B0, 224 × 224, AdamW (1e-4), batch 32, ≤ 30 epochs, 3 CNN seeds (42, 43, 44) · PCA to 6 components → angles
in [0, π] · QSVM: ZZ fidelity kernel, r = 2, bandwidth 0.05, C = 10 · VQC: 6 qubits, 3 re-uploading layers · 5 head
seeds · patient-grouped 64/16/20 split with the official CBIS-DDSM test set · depolarising p ∈ {0 … 0.1}, 1,024
shots · recalibration on k ∈ {20, 50, 100} patients.

## Citation

Please cite the article (see `CITATION.cff`). The CBIS-DDSM data should be cited as Lee RS et al. (2017),
*Sci Data* 4:170177, https://doi.org/10.1038/sdata.2017.177.

## License

Code: MIT (see `LICENSE`). CBIS-DDSM is subject to the TCIA data usage policy.

## Contact

Mageshkumar N, SRM Institute of Science and Technology, Ramapuram, Chennai, India — mageshkumarns19@gmail.com
