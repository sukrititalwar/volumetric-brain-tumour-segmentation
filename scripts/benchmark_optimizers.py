"""Compare ECO, RUNU-ECO and random search on standard benchmark functions (cheap convergence study,
complementing the HPO convergence curves of Fig. 10). Fitness values are kept >= 0 so Eq. (8) is
well defined.

  python scripts/benchmark_optimizers.py --runs 20 --pop 10 --iters 50
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from svae.optim.eco import OPTIMIZERS  # noqa: E402

FUNCS = {
    "sphere": (lambda x: float(np.sum(x ** 2)), 100),
    "rosenbrock": (lambda x: float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)), 30),
    "rastrigin": (lambda x: float(10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))), 5.12),
    "ackley": (lambda x: float(-20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e), 32),
    "griewank": (lambda x: float(np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1)))) + 1), 600),
    "schwefel_2_22": (lambda x: float(np.sum(np.abs(x)) + np.prod(np.abs(x))), 10),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--out", default="results/optimizer_benchmark.json")
    a = ap.parse_args()
    res = {}
    for fname, (f, bound) in FUNCS.items():
        res[fname] = {}
        for oname, O in OPTIMIZERS.items():
            finals, curves = [], []
            for r in range(a.runs):
                out = O(pop=a.pop, iters=a.iters, seed=r, verbose=False).minimize(f, [-bound] * a.dim, [bound] * a.dim)
                finals.append(out["best_f"])
                curves.append(out["curve"])
            res[fname][oname] = {"final": finals, "mean_curve": np.mean(curves, 0).tolist()}
            print(f"{fname:14s} {oname:9s} mean={np.mean(finals):.4g} median={np.median(finals):.4g} std={np.std(finals):.3g}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"))
