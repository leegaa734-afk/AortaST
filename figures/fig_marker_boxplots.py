#!/usr/bin/env python
"""Boxplots for the marker-gene panel.

Panels:
  A  AUC — 100-gene SVG panel vs 30 marker genes (combined)
  B  PCC — same comparison
  C  AUC by functional category (30 marker genes)
  D  PCC by functional category

Two versions are written: English labels (for the manuscript) and Chinese labels
(for the report).
"""
from __future__ import annotations

from pathlib import Path
import os
os.makedirs("results", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
A = pd.read_csv(BASE / "data" / "gene_panel_100.csv")
B = pd.read_csv(BASE / "data" / "marker_genes_30.csv")

CAT_EN = {
    "A. 收缩型平滑肌（中膜）": "SMC\n(media)",
    "B. 细胞外基质/间质（外膜）": "ECM\n(adventitia)",
    "C. 免疫/髓系": "Immune\n(myeloid)",
    "D. 血管内皮/周细胞": "Endo.\n(pericyte)",
    "E. 应激/代谢": "Stress\n(metabolic)",
}
CAT_CN = {
    "A. 收缩型平滑肌（中膜）": "平滑肌\n（中膜）",
    "B. 细胞外基质/间质（外膜）": "ECM\n（外膜）",
    "C. 免疫/髓系": "免疫\n（髓系）",
    "D. 血管内皮/周细胞": "内皮\n（周细胞）",
    "E. 应激/代谢": "应激\n（代谢）",
}
CAT_ORDER = list(CAT_EN)
COLORS = ["#2E5A87", "#C0504D", "#4F8A5B", "#8064A2", "#D9A441"]
C_100, C_30 = "#8FA9C4", "#C0504D"


def box(ax, groups, labels, colors, ylabel, title, ylim=None):
    bp = ax.boxplot(groups, patch_artist=True, widths=0.5, showfliers=False,
                    medianprops=dict(color="black", lw=1.4),
                    whiskerprops=dict(color="0.35"), capprops=dict(color="0.35"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor("0.25")
    rng = np.random.default_rng(0)
    for i, (vals, c) in enumerate(zip(groups, colors), start=1):
        x = i + rng.uniform(-0.14, 0.14, len(vals))

        ax.text(i, ax.get_ylim()[1] if ylim is None else ylim[1],
                f"n={len(vals)}\nμ={np.mean(vals):.3f}", ha="center", va="top", fontsize=8)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ylim:
        ax.set_ylim(*ylim)


def make(cat_labels, fname, suptitle):
    groups_cat = [B.loc[B.category == k, "auc"].to_numpy() for k in CAT_ORDER]
    groups_cat_p = [B.loc[B.category == k, "pcc"].to_numpy() for k in CAT_ORDER]
    labels_cat = [cat_labels[k] for k in CAT_ORDER]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.6))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    box(axes[0, 0], [A["auc"].to_numpy(), B["auc"].to_numpy()],
        [f"100 SVG\n({A.auc.mean():.3f})", f"30 marker\n({B.auc.mean():.3f})"],
        [C_100, C_30], "AUROC", "A   AUROC — combined", ylim=(0.40, 0.95))
    box(axes[0, 1], [A["pcc"].to_numpy(), B["pcc"].to_numpy()],
        [f"100 SVG\n({A.pcc.mean():.3f})", f"30 marker\n({B.pcc.mean():.3f})"],
        [C_100, C_30], "Pearson r", "B   PCC — combined", ylim=(-0.3, 0.95))
    box(axes[1, 0], groups_cat, labels_cat, COLORS, "AUROC", "C   AUROC — by category", ylim=(0.70, 0.90))
    box(axes[1, 1], groups_cat_p, labels_cat, COLORS, "Pearson r", "D   PCC — by category", ylim=(0.25, 0.90))

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = BASE / fname
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    pdf = out.with_suffix(".pdf")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print("saved", out, "and", pdf)
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams["font.family"] = "DejaVu Sans"
    make(CAT_EN, "results/fig_marker_boxplots_EN.png",
         "Marker-gene subset: AUROC and PCC (C1 model, 112 µm, held-out patients)")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    make(CAT_CN, "results/fig_marker_boxplots_CN.png",
         "Marker 基因子集：AUROC 与 PCC（C1 模型，112 µm，留出患者）")
