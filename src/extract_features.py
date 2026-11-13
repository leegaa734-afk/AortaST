"""Extract frozen CTransPath features for AortaST.

CTransPath weights and architecture are third-party and are **not** redistributed here.
Download them from the original authors (https://github.com/Xiyue-Wang/TransPath) and pass
``--ctranspath-dir`` pointing at the directory that contains ``ctrans/`` and ``ctranspath.pth``.

Example
-------
python -m src.extract_features --tiles data/tiles --labels data/labels112 \
    --samples con1 con3 con5 --c transpath-dir /path/to/ctrans --out data/features112
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", required=True, help="directory of 224-px PNG tiles named <sample>_<r>_<c>.png")
    ap.add_argument("--labels", required=True, help="directory of <sample>_labels.npz")
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--ctranspath-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    sys.path.insert(0, args.ctranspath_dir)
    from ctrans import CTransPath  # provided by the CTransPath release

    model = CTransPath(num_classes=0)
    ckpt = torch.load(Path(args.ctranspath_dir) / "ctranspath.pth", map_location="cpu",
                      weights_only=True)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    model = model.cuda().eval()

    tf = T.Compose([T.Resize(224, antialias=False), T.ToTensor(),
                    T.Normalize((.485, .456, .406), (.229, .224, .225))])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for s in args.samples:
        dst = out / f"{s}.npz"
        if dst.exists():
            print("skip", s)
            continue
        lab = np.load(Path(args.labels) / f"{s}_labels.npz", allow_pickle=False)
        ids = lab["tile_ids"].astype(str)
        X = np.zeros((len(ids), 768), dtype=np.float32)
        t0 = time.time()
        with torch.inference_mode():
            for b in range(0, len(ids), args.batch):
                chunk = ids[b:b + args.batch]
                batch = [tf(Image.open(Path(args.tiles) / f"{t}.png").convert("RGB")) for t in chunk]
                X[b:b + len(chunk)] = model(torch.stack(batch).cuda()).cpu().numpy()
        np.savez_compressed(dst, X=X, tile_ids=ids)
        print(f"{s}: {len(ids)} regions in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
