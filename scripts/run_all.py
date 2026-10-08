"""Orchestrates the whole paper pipeline, in order of importance, each step in a fresh process.

  python scripts/run_all.py --profile mini    # ~1 h smoke-scale run: 1 fold, small subsets, short training
  python scripts/run_all.py --profile quick   # reduced protocol that fits an 8 GB Apple-silicon laptop (~15 h)
  python scripts/run_all.py --profile paper   # the paper's settings (needs a large GPU and days of compute)

Steps per dataset: (1) RUNU-ECO and ECO hyper-parameter optimisation, (2) k-fold CV of the
RUNU-ECO-tuned SVAEUnet++, (3) baselines UNet / UNet++ / UNet3+, (4) ablations, then the report.
Every step caches its outputs, so the script can be re-run to resume.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

PROFILES = {
    "paper": dict(datasets=["brats", "figshare", "lgg"], hpo=dict(pop=10, iters=50, epochs=50, scale=1.0, max_train=None, max_val=None),
                  optimizers=["runu-eco", "eco"], epochs=50, step_scale=1.0, folds="0,1,2,3,4"),
    "quick": dict(datasets=["brats", "figshare", "lgg"], hpo=dict(pop=6, iters=4, epochs=3, scale=0.1, max_train=60, max_val=12),
                  optimizers=["runu-eco", "eco"], epochs=15, step_scale=0.3, folds="0,1,2"),
    # Subset sizes are per dimensionality: {3: BraTS subjects, 2: 2D slices}.
    "mini": dict(datasets=["brats", "figshare", "lgg"],
                 hpo=dict(pop=4, iters=2, epochs=2, scale=0.05, max_train={3: 60, 2: 300}, max_val={3: 12, 2: 60}),
                 optimizers=["runu-eco", "eco"], epochs=5, step_scale=0.1, folds="0",
                 max_train={3: 60, 2: 600}, max_test={3: 20, 2: 200}),
}
DIMS = {"brats": 3, "figshare": 2, "lgg": 2}


def _per_dims(v, ds):
    return v.get(DIMS[ds]) if isinstance(v, dict) else v
BASELINES = ["unet", "unetpp", "unet3p"]
ABLATIONS = ["efficient_unet", "efficient_unetpp"]  # + svaeunetpp (untuned) + tuned


def run(args, log):
    print(">>", " ".join(args), flush=True)
    with open(log, "a") as fh:
        r = subprocess.run([PY, "-u"] + args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        print(f"   FAILED ({r.returncode}), see {log}", flush=True)
    return r.returncode == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="quick", choices=list(PROFILES))
    ap.add_argument("--datasets", default=None)
    a = ap.parse_args()
    P = PROFILES[a.profile]
    res = os.path.join("results", a.profile)
    logs = os.path.join(ROOT, res, "logs")
    os.makedirs(logs, exist_ok=True)
    datasets = a.datasets.split(",") if a.datasets else P["datasets"]
    h = P["hpo"]
    for ds in datasets:
        log = os.path.join(logs, f"{ds}.log")
        # 1) HPO
        for opt in P["optimizers"]:
            hp_json = os.path.join(ROOT, res, "hpo", f"{ds}_{opt}.json")
            if not os.path.exists(hp_json):
                cmd = ["scripts/hpo.py", "--dataset", ds, "--optimizer", opt, "--pop", str(h["pop"]), "--iters", str(h["iters"]),
                       "--epochs", str(h["epochs"]), "--budget-scale", str(h["scale"]), "--out", os.path.join(res, "hpo")]
                if h["max_train"]:
                    cmd += ["--max-train", str(_per_dims(h["max_train"], ds)), "--max-val", str(_per_dims(h["max_val"], ds))]
                run(cmd, log)
        best = os.path.join(res, "hpo", f"{ds}_runu-eco.json")
        cv = ["--dataset", ds, "--epochs", str(P["epochs"]), "--folds", P["folds"], "--out", os.path.join(res, "cv")]
        for key, flag in (("max_train", "--max-train"), ("max_test", "--max-test")):
            if P.get(key):
                cv += [flag, str(_per_dims(P[key], ds))]
        default_steps = str(max(1, round(250 * P["step_scale"])))
        # 2) proposed model with RUNU-ECO-tuned HPs (steps/epoch scaled like the HPO budget)
        if os.path.exists(os.path.join(ROOT, best)):
            import json
            hp = json.load(open(os.path.join(ROOT, best)))["best_hparams"]
            run(["scripts/cross_validate.py", "--model", "svaeunetpp", "--tag", "runu_eco_svaeunetpp",
                 "--lr", str(hp["learning_rate"]), "--hidden-neurons", str(hp["hidden_neurons"]),
                 "--steps-per-epoch", str(max(1, round(hp["steps_per_epoch"] * P["step_scale"]))), "--save-preds"] + cv, log)
        eco = os.path.join(ROOT, res, "hpo", f"{ds}_eco.json")
        same = False
        if os.path.exists(eco) and os.path.exists(os.path.join(ROOT, best)):
            import json
            same = json.load(open(eco))["best_hparams"] == json.load(open(os.path.join(ROOT, best)))["best_hparams"]
            if same:
                print(f"   {ds}: ECO found the same hyper-parameters as RUNU-ECO - ECO-SVAEUnet++ CV skipped "
                      "(it would retrain an identical configuration)", flush=True)
        if os.path.exists(eco) and not same:
            import json
            hp = json.load(open(eco))["best_hparams"]
            run(["scripts/cross_validate.py", "--model", "svaeunetpp", "--tag", "eco_svaeunetpp",
                 "--lr", str(hp["learning_rate"]), "--hidden-neurons", str(hp["hidden_neurons"]),
                 "--steps-per-epoch", str(max(1, round(hp["steps_per_epoch"] * P["step_scale"])))] + cv, log)
        # 3) untuned SVAEUnet++ and ablations, then baselines (Table 5 fairness: same epochs & defaults)
        for m in ["svaeunetpp"] + ABLATIONS + BASELINES:
            run(["scripts/cross_validate.py", "--model", m, "--steps-per-epoch", default_steps]
                + (["--save-preds"] if m in BASELINES else []) + cv, log)
        run(["scripts/report.py", "--results", res], log)
    print("all done", flush=True)
