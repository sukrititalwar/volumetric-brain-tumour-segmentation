"""5-fold cross-validation of one model on one dataset (Tables 4, 6-9; Figs. 9, 11).

  python scripts/cross_validate.py --dataset brats --model svaeunetpp --hp results/hpo/brats_runu-eco.json
  python scripts/cross_validate.py --dataset brats --model unet --tag unet
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from svae.experiment import DataBundle, run_cv  # noqa: E402
from svae.models import MODELS  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="brats")
    ap.add_argument("--model", default="svaeunetpp", choices=MODELS)
    ap.add_argument("--hp", default=None, help="HPO result json; uses its best_hparams")
    ap.add_argument("--tag", default=None, help="result folder name (default: model name)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--hidden-neurons", type=int, default=None)
    ap.add_argument("--folds", default=None, help="comma list, e.g. 0,1 (default: all 5)")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--save-preds", action="store_true")
    ap.add_argument("--out", default="results/cv")
    a = ap.parse_args()

    b = DataBundle(a.dataset)
    hp = {}
    if a.hp:
        hp = json.load(open(a.hp))["best_hparams"]
    for k, v in (("steps_per_epoch", a.steps_per_epoch), ("learning_rate", a.lr), ("hidden_neurons", a.hidden_neurons)):
        if v is not None:
            hp[k] = v
    cfg = b.base_config(model=a.model, epochs=a.epochs, **hp)
    folds = [int(f) for f in a.folds.split(",")] if a.folds else None
    out = os.path.join(a.out, a.dataset, a.tag or a.model)
    run_cv(b, cfg, out, folds=folds, max_train=a.max_train, max_test=a.max_test, save_preds=a.save_preds)
