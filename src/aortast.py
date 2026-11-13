"""AortaST: H&E-only prediction of spatially resolved gene expression in human aorta.

Core library: data loading, spatially-variable gene selection, the MLP predictor,
and evaluation utilities.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# --------------------------------------------------------------------- data


def load_dataset(root, samples, feature_dir="data/features112", label_dir="data/labels112"):
    """Load region-level features and expression labels.

    Per sample, expects:
      ``{label_dir}/{sample}_labels.npz``  -> tile_ids, genes, Y  (regions x genes, log1p CP10K)
      ``{feature_dir}/{sample}.npz``       -> X (regions x 768), tile_ids
    """
    root = Path(root)
    Xs, Ys, Ps, Ts, genes = [], [], [], [], None
    for s in samples:
        lab = np.load(root / label_dir / f"{s}_labels.npz", allow_pickle=False)
        fea = np.load(root / feature_dir / f"{s}.npz", allow_pickle=False)
        ids = lab["tile_ids"].astype(str)
        assert ids.tolist() == fea["tile_ids"].astype(str).tolist(), f"tile id mismatch: {s}"
        if genes is None:
            genes = lab["genes"].astype(str).tolist()
        Xs.append(fea["X"])
        Ys.append(lab["Y"])
        Ps.extend([s] * len(ids))
        Ts.extend(ids.tolist())
    X = np.concatenate(Xs).astype(np.float32)
    Y = np.concatenate(Ys).astype(np.float32)
    P = np.array(Ps)
    T = np.array(Ts)
    keep = np.isfinite(Y).all(axis=0)          # genes missing from some sample are NaN
    genes = [g for g, k in zip(genes, keep) if k]
    Y = Y[:, keep]
    return X, Y, P, T, genes


# ------------------------------------------------------------- gene selection


def _bh(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    q = pv[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    res = np.empty(n)
    res[order] = np.minimum(q, 1.0)
    out[ok] = res
    return out


def _moran(y, ids):
    """Global Moran's I per column of ``y`` on the 16-um bin grid (Queen adjacency)."""
    coords = [(int(r), int(c)) for _, r, c in (t.rsplit("_", 2) for t in ids)]
    pos = {c: i for i, c in enumerate(coords)}
    rows, cols = [], []
    for i, (r, c) in enumerate(coords):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                j = pos.get((r + dr, c + dc))
                if j is not None:
                    rows.append(i)
                    cols.append(j)
    n = len(coords)
    w = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    s0 = float(w.sum())
    z = y - y.mean(axis=0, keepdims=True)
    den = np.sum(z * z, axis=0)
    num = np.sum(z * w.dot(z), axis=0)
    return np.divide(n * num, s0 * den, out=np.full(y.shape[1], np.nan), where=den > 0)


def select_svg(samples, root, label_dir="data/labels112", moran_min=0.10,
               q_cut=0.05, min_patients=9, min_moran=0.04):
    """Genome-wide spatial-variability filter.

    A gene is retained when its expression is spatially variable (Moran's I >= moran_min,
    BH q < q_cut) in at least ``min_patients`` samples and its per-sample minimum Moran's I
    is >= ``min_moran``.
    """
    root = Path(root)
    first = True
    support = None
    min_m = None
    for s in samples:
        z = np.load(root / label_dir / f"{s}_labels.npz", allow_pickle=False)
        genes = z["genes"].astype(str).tolist()
        m = _moran(z["Y"].astype(np.float64), z["tile_ids"].astype(str).tolist())
        n = len(m)
        exp = -1.0 / (n - 1)
        p = norm.sf((m - exp) / max(np.sqrt(max(1e-12, 1.0)), 1e-12))  # conservative
        q = _bh(p)
        hit = (q < q_cut) & (m >= moran_min)
        if first:
            support = np.zeros(len(genes), dtype=int)
            min_m = np.full(len(genes), np.inf)
            first = False
        support += hit.astype(int)
        min_m = np.fmin(min_m, m)
    keep = (support >= min_patients) & (min_m >= min_moran)
    return [g for g, k in zip(genes, keep) if k]


def rank_panel_by_auc(X, Y, P, genes, dev, panel_size=100, folds=5, epochs=200, seed=20261101):
    """Rank genes by development out-of-fold per-gene AUC and keep the top ``panel_size``."""
    dev_idx = np.flatnonzero(np.isin(P, dev))
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_aucs = []
    for k, (a, b) in enumerate(kf.split(dev)):
        tr = np.flatnonzero(np.isin(P, [dev[i] for i in a]))
        va = np.flatnonzero(np.isin(P, [dev[i] for i in b]))
        net, m = train_mlp(X, Y, P, tr, epochs, seed + k, arch="mlp128")
        fold_aucs.append(per_gene_auc(Y[va], predict(net, X[va], m), genes, P[va]))
    gene_auc = {g: float(np.nanmean([f[g] for f in fold_aucs])) for g in genes}
    ranked = sorted(genes, key=lambda g: (-gene_auc[g] if np.isfinite(gene_auc[g]) else 1))
    panel = [g for g in ranked[:panel_size] if np.isfinite(gene_auc[g])]
    return panel, gene_auc


# ------------------------------------------------------------------ the model


class MLP(nn.Module):
    """768 -> 128 -> G regression head (the AortaST predictor)."""

    def __init__(self, d_in, d_out, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, d_out)
        )

    def forward(self, x):
        return self.net(x)


def _patient_weights(p):
    w = np.zeros(len(p), dtype=np.float64)
    for s in np.unique(p):
        m = p == s
        w[m] = 1.0 / m.sum()
    return w / w.sum()


def train_mlp(X, Y, P, train_idx, epochs=200, seed=0, arch="mlp128", lr=5e-4, batch=128,
              weight_decay=1e-4, dropout=0.2):
    """Train one AortaST MLP with patient-balanced sampling and MSE loss."""
    torch.manual_seed(seed)
    net = MLP(X.shape[1], Y.shape[1], dropout=dropout).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    p = P[train_idx]
    probs = _patient_weights(p)
    xm = np.mean([X[train_idx][p == s].mean(0) for s in np.unique(p)], axis=0)
    xs = np.maximum(np.sqrt(np.mean([((X[train_idx][p == s] - xm) ** 2).mean(0)
                                     for s in np.unique(p)], axis=0)), 0.01)
    Xt = torch.tensor(((X[train_idx] - xm) / xs).astype(np.float32), device="cuda")
    Yt = torch.tensor(Y[train_idx].astype(np.float32), device="cuda")
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        idx = rng.choice(len(train_idx), len(train_idx), replace=True, p=probs)
        for b in range(0, len(idx), batch):
            sel = idx[b:b + batch]
            loss = ((net(Xt[sel]) - Yt[sel]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return net, {"xm": xm, "xs": xs}


def predict(net, X, m):
    with torch.no_grad():
        xb = torch.tensor(((X - m["xm"]) / m["xs"]).astype(np.float32), device="cuda")
        return net(xb).cpu().numpy()


# ---------------------------------------------------------------- evaluation


def per_gene_auc(truth, pred, genes, patients):
    """Per-gene AUC within each patient (median split), averaged across patients."""
    out = {}
    for j, g in enumerate(genes):
        vals = []
        for s in np.unique(patients):
            m = patients == s
            y = truth[m, j]
            if m.sum() < 10 or np.unique(y).size < 2:
                continue
            b = (y > np.median(y)).astype(int)
            if b.min() == b.max():
                continue
            vals.append(roc_auc_score(b, pred[m, j]))
        out[g] = float(np.mean(vals)) if vals else float("nan")
    return out


def smooth_predictions(pred, patients, tile_ids, radius=1):
    """Average each prediction with its 8 grid neighbours within the same patient."""
    if radius <= 0:
        return pred
    out = pred.copy()
    for s in np.unique(patients):
        m = patients == s
        grid = {}
        for k, t in enumerate(tile_ids[m]):
            _, r, c = t.rsplit("_", 2)
            grid[(int(r), int(c))] = k
        for (r, c), k in grid.items():
            nb = [grid.get((r + dr, c + dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
            nb = [i for i in nb if i is not None]
            out[m][k] = pred[m][nb].mean(axis=0)
    return out
