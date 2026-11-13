"""Train and evaluate AortaST (H&E-only spatial gene expression prediction).

Example
-------
python -m src.train --config configs/default.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from . import aortast as A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--panel", type=int, default=100)
    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = cfg["root"]
    dev, test = cfg["development_patients"], cfg["test_patients"]

    X, Y, P, T, genes = A.load_dataset(root, dev + test,
                                       cfg["feature_dir"], cfg["label_dir"])
    print(f"{len(X)} regions, {len(genes)} genes, {len(dev)} dev / {len(test)} test")

    # 1) spatially variable gene filter (no model information)
    svg = A.select_svg(dev, root, cfg["label_dir"])
    sel = [i for i, g in enumerate(genes) if g in set(svg)]
    genes_svg = [genes[i] for i in sel]
    Y_svg = Y[:, sel]
    print(f"SVG filter: {len(genes)} -> {len(genes_svg)} genes")

    # 2) rank by development out-of-fold AUC
    panel, gene_auc = A.rank_panel_by_auc(X, Y_svg, P, genes_svg, dev,
                                          panel_size=args.panel, epochs=args.epochs)
    pidx = [genes_svg.index(g) for g in panel]
    print(f"panel: {len(panel)} genes")

    # 3) ensemble on development, evaluate on held-out patients
    dev_idx = np.flatnonzero(np.isin(P, dev))
    test_idx = np.flatnonzero(np.isin(P, test))
    preds = []
    for sd in range(args.seeds):
        net, m = A.train_mlp(X[:, pidx], Y_svg[:, pidx], P, dev_idx,
                             epochs=args.epochs, seed=20261101 + sd)
        preds.append(A.predict(net, X[test_idx][:, pidx], m))
    raw = np.mean(preds, axis=0)
    sm = A.smooth_predictions(raw, P[test_idx], T[test_idx], args.smooth)

    truth, pat = Y_svg[test_idx][:, pidx], P[test_idx]
    auc = A.per_gene_auc(truth, sm, panel, pat)
    v = np.array([x for x in auc.values() if np.isfinite(x)])
    pcc = np.array([pearsonr(sm[:, j], truth[:, j])[0] for j in range(len(panel))
                    if sm[:, j].std() > 1e-8 and truth[:, j].std() > 1e-8])

    summary = {
        "model": "AortaST",
        "panel_size": len(panel),
        "n_test_regions": int(len(test_idx)),
        "auc_mean": float(v.mean()), "auc_median": float(np.median(v)),
        "auc_ge_075": int((v >= 0.75).sum()), "auc_ge_070": int((v >= 0.70).sum()),
        "pcc_mean": float(np.mean(pcc)), "pcc_median": float(np.median(pcc)),
        "pcc_ge_06": int((pcc >= 0.6).sum()),
        "panel": panel,
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    pd.DataFrame({"gene": panel,
                  "auc": [auc[g] for g in panel],
                  "pcc": [pearsonr(sm[:, j], truth[:, j])[0] for j in range(len(panel))]}
                 ).to_csv(out / "per_gene.csv", index=False)
    print(json.dumps({k: summary[k] for k in
                      ("auc_mean", "auc_median", "auc_ge_075", "pcc_mean", "pcc_median")},
                     indent=1))


if __name__ == "__main__":
    main()
