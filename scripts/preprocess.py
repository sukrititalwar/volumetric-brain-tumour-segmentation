"""Run the preprocessing pipeline for every dataset.

  python scripts/preprocess.py brats   # Dataset 1: BraTS2020 (3D, 369 subjects)
  python scripts/preprocess.py figshare  # 2D substitute for Dataset 2 (3,064 T1-CE slices, masks)
  python scripts/preprocess.py lgg       # 2D substitute for Dataset 3 (3,929 FLAIR slices, masks)
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from svae.data.preprocess import preprocess_2d, preprocess_brats  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")


def figshare_pairs():
    base = glob.glob(os.path.join(RAW, "brain-tumor-segmentation", "**", "images"), recursive=True)[0]
    pairs = []
    for ip in sorted(glob.glob(os.path.join(base, "*.png"))):
        mp = os.path.join(os.path.dirname(base), "masks", os.path.basename(ip))
        if os.path.exists(mp):
            # this release carries no patient IDs, so each slice is its own CV group
            pairs.append((ip, mp, os.path.basename(ip)))
    return pairs


def lgg_pairs():
    pairs = []
    for mp in sorted(glob.glob(os.path.join(RAW, "lgg-mri-segmentation", "**", "*_mask.tif"), recursive=True)):
        ip = mp.replace("_mask.tif", ".tif")
        if os.path.exists(ip):
            pairs.append((ip, mp, os.path.basename(os.path.dirname(mp))))  # patient folder = CV group
    return pairs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["brats", "figshare", "lgg"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--size2d", type=int, default=128)
    a = ap.parse_args()
    if a.dataset == "brats":
        preprocess_brats(os.path.join(RAW, "brats2020.zip"), os.path.join(OUT, "brats"), workers=a.workers)
    elif a.dataset == "figshare":
        p = figshare_pairs()
        print(f"figshare: {len(p)} image/mask pairs")
        preprocess_2d(p, os.path.join(OUT, "figshare.npz"), size=a.size2d, workers=a.workers)
    else:
        p = lgg_pairs()
        print(f"lgg: {len(p)} image/mask pairs, {len(set(g for *_, g in p))} patients")
        preprocess_2d(p, os.path.join(OUT, "lgg.npz"), size=a.size2d, workers=a.workers)
