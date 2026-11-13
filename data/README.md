# Data

Only derived, non-identifiable tables are distributed:

| File | Content |
|---|---|
| `gene_panel_100.csv` | the 100-gene AortaST panel with per-gene held-out AUROC and Pearson r |
| `marker_genes_30.csv` | the 30-gene marker subset, with functional category, AUROC and Pearson r |
| `marker_corr_matrix.csv` | within-patient correlation matrix of the 30 marker genes (used for the module clustering) |

## What is not included

- **Spatial transcriptomics counts and region labels.** These are patient-derived and covered by
  institutional data-use agreements.
- **Whole-slide H&E images and tile crops.** Same restriction.
- **CTransPath weights** (`ctranspath.pth`). Third-party model, see
  <https://github.com/Xiyue-Wang/TransPath>.

Processed region-level matrices can be obtained from the corresponding author upon reasonable
request and with institutional approval.

## Format expected by the code

`labels112/<patient>_labels.npz`

| array | shape | meaning |
|---|---|---|
| `tile_ids` | (n,) | `<patient>_<row>_<col>`, the 112-µm region identifier |
| `genes` | (G,) | gene symbols, same order for every patient |
| `Y` | (n, G) | `log1p(CP10K)` expression |

`features112/<patient>.npz`

| array | shape | meaning |
|---|---|---|
| `X` | (n, 768) | frozen CTransPath feature |
| `tile_ids` | (n,) | must match `labels112` |