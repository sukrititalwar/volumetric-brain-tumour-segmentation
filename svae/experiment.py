"""Dataset registry + k-fold experiment runner shared by the scripts."""
import json
import os
from dataclasses import asdict, replace

import numpy as np

from .data.datasets import BraTSDataset, Slice2DDataset, kfold_splits, load_2d, load_brats, train_val_split
from .train import TrainConfig, evaluate, get_device, train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")

DATASETS = {
    # name: (paper role, dims)
    "brats": ("Dataset 1: BraTS2020, 3D multimodal MRI (369 subjects)", 3),
    "figshare": ("Dataset 2 substitute: Figshare/Cheng 3,064 T1-CE slices (glioma/meningioma/pituitary)", 2),
    "lgg": ("Dataset 3 substitute: TCGA-LGG FLAIR slices (110 patients, 3,929 slices)", 2),
}


class DataBundle:
    """Lazily loads a processed dataset and builds train/val/test Dataset objects for index sets."""

    def __init__(self, name, target="wt", patch=(64, 64, 64), samples_per_epoch=None):
        self.name, self.dims = name, DATASETS[name][1]
        if name == "brats":
            self.img, self.seg, self.ids = load_brats(os.path.join(PROC, "brats"))
            self.groups = None
            self.in_ch, self.patch = 4, tuple(patch)
        else:
            self.img, self.seg, groups = load_2d(os.path.join(PROC, f"{name}.npz"))
            self.groups = groups
            self.ids = [str(g) for g in groups]
            self.in_ch, self.patch = 1, tuple(self.img.shape[2:])
        self.n = len(self.img)
        self.target = target

    def train_set(self, idx, seed=0, samples_per_epoch=None):
        if self.dims == 3:
            return BraTSDataset(self.img, self.seg, idx, self.target, self.patch, True, samples_per_epoch, seed)
        return Slice2DDataset(self.img, self.seg, idx, True, seed)

    def eval_set(self, idx):
        if self.dims == 3:
            return BraTSDataset(self.img, self.seg, idx, self.target, self.patch, train=False)
        return Slice2DDataset(self.img, self.seg, idx, train=False)

    def folds(self, k=5, seed=42):
        return kfold_splits(self.n, k, seed, self.groups)

    def base_config(self, **kw):
        cfg = TrainConfig(dims=self.dims, in_ch=self.in_ch, patch=self.patch,
                          batch_size=2 if self.dims == 3 else 8)
        return replace(cfg, **kw)


def run_cv(bundle, cfg, out_dir, folds=None, k=5, seed=42, max_train=None, max_test=None, val_frac=0.15,
           save_preds=False):
    """K-fold CV (Table 4): each fold trains on 4/5 (with an internal validation split for model
    selection) and tests on the held-out 1/5. Writes per-fold metrics + training histories."""
    os.makedirs(out_dir, exist_ok=True)
    splits = bundle.folds(k, seed)
    folds = list(range(k)) if folds is None else folds
    results = []
    for f in folds:
        fold_dir = os.path.join(out_dir, f"fold{f}")
        res_path = os.path.join(fold_dir, "metrics.json")
        if os.path.exists(res_path):
            results.append(json.load(open(res_path)))
            print(f"[{cfg.model}] fold {f}: cached", flush=True)
            continue
        tr, te = splits[f]
        tr, va = train_val_split(tr, val_frac, seed + f)
        rng = np.random.default_rng(seed + f)
        if max_train:
            tr = rng.permutation(tr)[:max_train]
            va = rng.permutation(va)[:max(4, max_train // 6)]
        if max_test:
            te = rng.permutation(te)[:max_test]
        print(f"[{cfg.model}] {bundle.name} fold {f}: train={len(tr)} val={len(va)} test={len(te)}", flush=True)
        model, hist = train(replace(cfg, seed=cfg.seed + f), bundle.train_set(tr, seed=f), bundle.eval_set(va),
                            out_dir=fold_dir)
        ev = evaluate(model, bundle.eval_set(te), get_device(), return_preds=save_preds)
        per_case = ev.pop("per_case")
        preds = ev.pop("preds", None)
        res = {"fold": f, "metrics": ev, "per_case": per_case, "test_idx": [int(i) for i in te],
               "config": asdict(cfg)}
        if preds is not None:
            np.savez_compressed(os.path.join(fold_dir, "preds.npz"), idx=np.asarray(te), preds=np.stack(preds))
        json.dump(res, open(res_path, "w"))
        print(f"[{cfg.model}] fold {f} TEST: " + " ".join(f"{m}={ev[m]:.4f}" for m in ("dice", "iou", "accuracy", "sensitivity", "precision")), flush=True)
        results.append(res)
    return results
