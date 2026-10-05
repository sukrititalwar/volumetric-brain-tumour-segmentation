"""Datasets, augmentation (paper Step 6) and 5-fold cross-validation splits (Table 4)."""
import glob
import math
import os

import numpy as np
import torch
from sklearn.model_selection import GroupKFold, KFold
from torch.utils.data import Dataset

# Binary targets derived from harmonised BraTS labels {1: NCR/NET, 2: ED, 3: ET}
BRATS_TARGETS = {"wt": (1, 2, 3), "tc": (1, 3), "et": (3,)}


# ------------------------------------------------------------------------------------ augmentation

def augment(img, seg, rng, dims):
    """Step 6: random flip, rotation (90-degree in-plane + small arbitrary in 2D), scaling, intensity."""
    sp = tuple(range(1, 1 + dims))  # spatial axes of img (C, *spatial)
    for ax in sp:
        if rng.random() < 0.5:
            img, seg = np.flip(img, ax), np.flip(seg, ax - 1)
    if rng.random() < 0.5:
        k = int(rng.integers(1, 4))
        a, b = (sp[-2], sp[-1]) if dims == 3 else sp
        img, seg = np.rot90(img, k, (a, b)), np.rot90(seg, k, (a - 1, b - 1))
    if rng.random() < 0.3:  # random scaling (zoom about centre, crop/pad back)
        img, seg = _random_scale(img, seg, rng.uniform(0.85, 1.15))
    if rng.random() < 0.5:  # intensity scale & shift per channel
        c = img.shape[0]
        sh = (c,) + (1,) * dims
        img = img * rng.uniform(0.9, 1.1, sh) + rng.uniform(-0.1, 0.1, sh)
    return np.ascontiguousarray(img, dtype=np.float32), np.ascontiguousarray(seg)


def _random_scale(img, seg, s):
    from scipy import ndimage
    from .preprocess import center_crop_pad
    shape = seg.shape
    img = np.stack([ndimage.zoom(img[c], s, order=1) for c in range(img.shape[0])])
    seg = ndimage.zoom(seg, s, order=0)
    return center_crop_pad(img, shape), center_crop_pad(seg, shape)


def random_patch(img, seg, patch, rng, fg_prob=0.67):
    """Extract a 3D patch; with prob fg_prob centre it on a tumour voxel (class balancing)."""
    shp = seg.shape
    if rng.random() < fg_prob and seg.any():
        fg = np.argwhere(seg > 0)
        c = fg[rng.integers(len(fg))]
        start = [int(np.clip(ci - p // 2, 0, s - p)) for ci, p, s in zip(c, patch, shp)]
    else:
        start = [int(rng.integers(0, s - p + 1)) for p, s in zip(patch, shp)]
    sl = tuple(slice(a, a + p) for a, p in zip(start, patch))
    return img[(slice(None),) + sl], seg[sl]


# -------------------------------------------------------------------------------------- datasets

class BraTSDataset(Dataset):
    """Dataset 1 (memory-mapped). Training: random 3D patches. Evaluation: whole (cropped) volume."""

    def __init__(self, img, seg, idx, target="wt", patch=(64, 64, 64), train=True, samples_per_epoch=None, seed=0):
        self.img, self.seg, self.idx = img, seg, np.asarray(idx)
        self.patch, self.train, self.seed = tuple(patch), train, seed
        self.labels = BRATS_TARGETS[target]
        self.n = samples_per_epoch or len(self.idx)

    def __len__(self):
        return self.n if self.train else len(self.idx)

    # Pickling a np.memmap copies the whole array (2.8 GB) into every DataLoader worker;
    # instead ship the file paths and re-open the memory maps lazily in the worker.
    def __getstate__(self):
        st = self.__dict__.copy()
        st["img"], st["seg"] = self.img.filename, self.seg.filename
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)
        self.img = np.load(st["img"], mmap_mode="r")
        self.seg = np.load(st["seg"], mmap_mode="r")

    def __getitem__(self, k):
        rng = np.random.default_rng((self.seed, k, np.random.randint(1 << 30)))
        i = self.idx[int(rng.integers(len(self.idx)))] if self.train else self.idx[k]
        seg = np.isin(self.seg[i], self.labels).astype(np.uint8)
        img = self.img[i]
        if self.train:
            img, seg = random_patch(img, seg, self.patch, rng)
            img, seg = augment(img.astype(np.float32), seg, rng, dims=3)
        img = np.ascontiguousarray(img, dtype=np.float32)
        return torch.from_numpy(img), torch.from_numpy(seg[None].astype(np.float32))


class Slice2DDataset(Dataset):
    """Datasets 2 & 3 (2D slices)."""

    def __init__(self, img, seg, idx, train=True, seed=0):
        self.img, self.seg, self.idx, self.train, self.seed = img, seg, np.asarray(idx), train, seed

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        img, seg = self.img[i].astype(np.float32), self.seg[i]
        if self.train:
            rng = np.random.default_rng((self.seed, k, np.random.randint(1 << 30)))
            img, seg = augment(img, seg, rng, dims=2)
        return torch.from_numpy(np.ascontiguousarray(img)), torch.from_numpy(seg[None].astype(np.float32))


# -------------------------------------------------------------------------------------- CV splits

def kfold_splits(n, k=5, seed=42, groups=None):
    """5-fold CV (Table 4). Grouped by patient when groups are given so no patient leaks across folds."""
    idx = np.arange(n)
    if groups is not None and len(set(groups)) < n:
        rng = np.random.default_rng(seed)
        uniq = np.unique(groups)
        perm = {g: r for g, r in zip(uniq, rng.permutation(len(uniq)))}
        order = np.argsort([perm[g] for g in groups], kind="stable")
        return [(order[tr], order[te]) for tr, te in GroupKFold(k).split(idx[order], groups=np.asarray(groups)[order])]
    return list(KFold(k, shuffle=True, random_state=seed).split(idx))


def train_val_split(train_idx, val_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.permutation(train_idx)
    n_val = max(1, int(math.ceil(len(p) * val_frac)))
    return p[n_val:], p[:n_val]


def load_brats(root):
    """Memory-mapped (N,4,D,H,W) float16 images and (N,D,H,W) uint8 labels built by scripts/preprocess.py."""
    img = np.load(os.path.join(root, "brats_img.npy"), mmap_mode="r")
    seg = np.load(os.path.join(root, "brats_seg.npy"), mmap_mode="r")
    cases = open(os.path.join(root, "brats_cases.txt")).read().split()
    return img, seg, cases


def load_2d(path):
    d = np.load(path, allow_pickle=True)
    return d["img"], d["seg"], d["groups"]
