# AortaST

**AortaST** predicts spatially resolved gene expression in human aorta from H&E images alone.
It pairs a frozen pathology foundation model (CTransPath) with a small multi-layer perceptron,
and is trained on 112-µm regions imaged at 224 px (0.5 µm/px, native resolution).

---

## Key results

Trained on 10 patients and evaluated on 3 held-out patients:

| Metric | Value |
|---|---|
| Region size | 112 µm (7 × 7 bins of 16 µm) |
| Input | 224 × 224 px H&E tile, 0.5 µm/px |
| Encoder | CTransPath, frozen, 768-d |
| Panel | 100 spatially variable genes |
| **Mean AUROC** | **0.730** (median 0.750) |
| Genes with AUROC ≥ 0.75 | 50 / 100 |
| **Mean Pearson r** | **0.522** (median 0.574) |
| 30-gene marker subset | AUROC 0.784, Pearson r 0.663 |

A 30-gene marker panel spanning contractile, matrix, immune, endothelial and stress programmes
is listed in [`data/marker_genes_30.csv`](data/marker_genes_30.csv). These genes form two
mutually anticorrelated modules (media/contractile/matrix vs immune/adventitial), which is
documented in [`data/marker_corr_matrix.csv`](data/marker_corr_matrix.csv).

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CTransPath weights and architecture are **third-party** and are not redistributed here.
Download them from <https://github.com/Xiyue-Wang/TransPath> and place `ctrans/` and
`ctranspath.pth` in a directory of your choice; `src/extract_features.py` takes it as
`--ctranspath-dir`.

A CUDA-capable GPU is required (the reference run used a single A100).

---

## Data layout

AortaST expects two parallel directories, one file per patient:

```
data/
├── tiles/                     # 224-px PNG tiles: <patient>_<row>_<col>.png
├── labels112/                 # <patient>_labels.npz  -> tile_ids, genes, Y
└── features112/               # <patient>.npz         -> X (n x 768), tile_ids
    └── he/<patient>.png       # H&E crop from the high-resolution scan (figures only)
```

`labels112` must contain `tile_ids` (the same `<patient>_<row>_<col>` strings as the tiles),
`genes` (gene symbols) and `Y` (regions × genes, `log1p(CP10K)` normalised counts). Regions with
fewer than 24 bins or fewer than 200 counts should be dropped before export.

`features112` is produced by `src/extract_features.py`; each `.npz` stores `X` and `tile_ids`.

> **Patient data are not included.** Raw spatial transcriptomics counts and whole-slide H&E images
> are subject to institutional data-use agreements. Only the gene panels and correlation matrix
> are distributed. See [`data/README.md`](data/README.md).

---

## Usage

### 1. Extract features

```bash
python -m src.extract_features \
    --tiles data/tiles --labels data/labels112 \
    --samples con1 con3 con5 A2 A5 A6 A7 C1 C2 C9 C11 C6 C5 \
    --ctranspath-dir /path/to/ctranspath \
    --out data/features112
```

### 2. Train and evaluate

```bash
python -m src.train --config configs/default.json
```

This runs the full pipeline:

1. genome-wide spatial-variability filter (Moran's I ≥ 0.10, BH q < 0.05 in ≥9/10 patients)
2. panel ranking by development out-of-fold AUROC, top 100 genes
3. 10-seed MLP ensemble with spatial smoothing (radius 1)
4. held-out evaluation → `results/summary.json` and `results/per_gene.csv`

### 3. Reproduce the figures

```bash
python figures/fig1_letter.py        # workflow, AUC distribution, categories, co-linearity, maps, anatomy
python figures/fig_spatial.py        # per-patient H&E with observed vs predicted maps
```

---

## Model

```
H&E 224-px tile ──► frozen CTransPath (768-d)
                              │
                              ▼
             MLP 768 → 128 → G   (ReLU, dropout 0.2)
                              │
                   10-seed ensemble
                              │
                   spatial smoothing (r = 1)
                              │
                              ▼
                 per-region expression, 100 genes
```

Training: MSE loss with patient-balanced sampling, AdamW (lr 5 × 10⁻⁴, weight decay 1 × 10⁻⁴),
batch size 128, 200 epochs. Per-gene AUROC is computed within each patient at that patient's
median expression and averaged across patients.

---

## Reproducibility notes

- Gene selection uses **development patients only**; test patients are never used for panel
  selection, early stopping or hyperparameter choice.
- The spatially-variable filter deliberately avoids selecting genes on model performance. Ranking
  by predictability alone inflates the apparent AUROC by roughly 0.03, which is why the SVG-first
  design is used here.
- Predicted values are conditional means and therefore have lower variance than measurements
  (predicted standard deviation is ~45% of observed).

---

## Citation

**Author:** Genmao Cao — Zhongshan Hospital, Fudan University, Shanghai, China

```bibtex
@software{cao2026aortast,
  title     = {AortaST: H\&E-only prediction of spatially resolved gene expression in human aorta},
  author    = {Cao, Genmao},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/leegaa734-afk/AortaST},
  note      = {Zhongshan Hospital, Fudan University}
}
```

## Licence

MIT for the code in this repository. CTransPath is distributed under its own licence and is not
included. See [`LICENSE`](LICENSE) and [`data/README.md`](data/README.md).
