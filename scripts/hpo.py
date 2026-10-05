"""Hyper-parameter optimisation of SVAEUnet++ with RUNU-ECO (or ECO / random search), Eq. (6).

Searches (hidden neurons, learning rate, steps per epoch) to minimise 1/IoU on a validation split
carved from the *training* part of CV fold 0 (the test fold is never touched).

Paper setting : --pop 10 --iters 50 --epochs 50 --budget-scale 1.0   (~500 full trainings)
Quick setting : --pop 6 --iters 5 --epochs 4 --budget-scale 0.1 --max-train 48
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from svae.data.datasets import train_val_split  # noqa: E402
from svae.experiment import DataBundle  # noqa: E402
from svae.optim.eco import OPTIMIZERS  # noqa: E402
from svae.train import HPO_LB, HPO_UB, decode, make_objective  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="brats")
    ap.add_argument("--optimizer", default="runu-eco", choices=list(OPTIMIZERS))
    ap.add_argument("--model", default="svaeunetpp")
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--budget-scale", type=float, default=1.0)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/hpo")
    a = ap.parse_args()

    b = DataBundle(a.dataset)
    tr, _ = b.folds()[0]
    tr, va = train_val_split(tr, 0.2, a.seed)
    rng = np.random.default_rng(a.seed)
    if a.max_train:
        tr = rng.permutation(tr)[:a.max_train]
    if a.max_val:
        va = rng.permutation(va)[:a.max_val]
    cfg = b.base_config(model=a.model, epochs=a.epochs, seed=a.seed)
    log = []
    fobj = make_objective(cfg, b.train_set(tr, seed=a.seed), b.eval_set(va), a.budget_scale, log)
    opt = OPTIMIZERS[a.optimizer](pop=a.pop, iters=a.iters, seed=a.seed)
    print(f"{opt.name} on {a.dataset}: pop={a.pop} iters={a.iters} epochs={a.epochs} "
          f"budget_scale={a.budget_scale} train={len(tr)} val={len(va)}", flush=True)
    t = time.time()
    res = opt.minimize(fobj, HPO_LB, HPO_UB)
    best = decode(res["best_x"])
    out = {"dataset": a.dataset, "optimizer": opt.name, "best_hparams": best, "best_fitness": res["best_f"],
           "best_iou": 1.0 / res["best_f"], "curve": res["curve"], "evaluations": log,
           "settings": vars(a), "seconds": time.time() - t}
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.dataset}_{a.optimizer}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"best {best} IoU={1 / res['best_f']:.4f} -> {path}")
