# Figures

| Script | Produces | Needs |
|---|---|---|
| `fig_marker_boxplots.py` | AUC and Pearson r of the 100-gene panel versus the 30-gene marker subset, overall and by functional category (boxplots) | only `data/*.csv` — runs out of the box |
| `fig_spatial.py` | per-patient H&E with observed vs predicted maps for the marker genes | `data/he/<patient>.png` + `results/test_predictions.npz` |
| `fig1_letter.py` | six-panel summary figure (workflow, AUC waterfall, categories, co-linearity, spatial maps, anatomy benchmark) | the above plus `data/marker_corr_matrix.csv` |

Only `fig_marker_boxplots.py` is fully self-contained. The other two require the H&E crops and
the per-region predictions, which are derived from the restricted patient data (see
`../data/README.md`).

## Colour convention

Spatial maps use within-patient **percentile rank** (0–100) so that observed and predicted panels
share a comparable visual scale. Predicted values are conditional means and therefore have lower
variance than measurements; percentile colouring restores contrast without altering the numbers.