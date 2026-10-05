"""Aggregate all results into paper-style tables (Tables 6-9 + HPO) and figures (Figs. 8-11).

  python scripts/report.py --results results/quick
Writes <results>/report/REPORT.md and PNG figures.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from svae.metrics import METRIC_NAMES, describe  # noqa: E402

# fixed colour per entity (reference categorical palette, fixed order, never cycled)
COLORS = {"runu_eco_svaeunetpp": "#2a78d6", "unet": "#eb6834", "unetpp": "#1baf7a", "unet3p": "#eda100",
          "svaeunetpp": "#e87ba4", "eco_svaeunetpp": "#008300", "efficient_unet": "#4a3aa7", "efficient_unetpp": "#e34948",
          "RUNU-ECO": "#2a78d6", "ECO": "#008300", "RandomSearch": "#8a8985", "runu-eco": "#2a78d6", "eco": "#008300", "random": "#8a8985"}
LABEL = {"runu_eco_svaeunetpp": "RUNU-ECO-SVAEUnet++", "eco_svaeunetpp": "ECO-SVAEUnet++", "svaeunetpp": "SVAEUnet++ (untuned)",
         "unet": "UNet", "unetpp": "UNet++", "unet3p": "UNet 3+", "efficient_unet": "Efficient-UNet",
         "efficient_unetpp": "Efficient-UNet++"}
DS_LABEL = {"brats": "Dataset 1 - BraTS2020 (3D, WT)", "figshare": "Dataset 2* - Figshare T1-CE (2D)",
            "lgg": "Dataset 3* - TCGA-LGG FLAIR (2D)"}
COMPARE = ["unet", "unetpp", "unet3p", "runu_eco_svaeunetpp"]
ABLATION = ["efficient_unet", "efficient_unetpp", "svaeunetpp", "eco_svaeunetpp", "runu_eco_svaeunetpp"]
ALL = ["unet", "unetpp", "unet3p", "efficient_unet", "efficient_unetpp", "svaeunetpp", "eco_svaeunetpp", "runu_eco_svaeunetpp"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8985",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "figure.dpi": 130, "savefig.bbox": "tight"})


def load_cv(res):
    out = {}
    for p in glob.glob(os.path.join(res, "cv", "*", "*", "fold*", "metrics.json")):
        parts = p.split(os.sep)
        ds, tag = parts[-4], parts[-3]
        out.setdefault(ds, {}).setdefault(tag, []).append(json.load(open(p)))
    for ds in out:
        for tag in out[ds]:
            out[ds][tag].sort(key=lambda r: r["fold"])
    return out


def fold_values(runs, metric, positive_only=False):
    if positive_only:
        return [np.mean([c[metric] for c in r["per_case"] if c.get("gt_positive", True)]) for r in runs]
    return [r["metrics"][metric] for r in runs]


def pm(v, pct=True):
    v = np.asarray(v) * (100 if pct else 1)
    return f"{v.mean():.2f} ± {v.std():.2f}" if len(v) > 1 else f"{v.mean():.2f}"


def table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "|".join(["---"] * len(header)) + "|\n"
    return s + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def main(res):
    rep = os.path.join(res, "report")
    os.makedirs(rep, exist_ok=True)
    cv = load_cv(res)
    md = [f"# SVAEUnet++ / RUNU-ECO reproduction report (`{res}`)\n",
          "All numbers are percentages on held-out CV test folds (mean ± std over folds) unless stated. "
          "IoU = Jaccard = TP/(TP+FP+FN). BraTS metrics are per-subject on the whole-tumour (WT) region at the "
          "preprocessed 1.6 mm grid; 2D metrics are per-slice.\n"]

    # ---------------------------------------------------------------- HPO
    hpo = {}
    for p in sorted(glob.glob(os.path.join(res, "hpo", "*.json"))):
        h = json.load(open(p))
        hpo.setdefault(h["dataset"], {})[h["optimizer"]] = h
    if hpo:
        rows = []
        for ds, d in hpo.items():
            for opt, h in d.items():
                b = h["best_hparams"]
                rows.append([DS_LABEL.get(ds, ds), opt, b["hidden_neurons"], f"{b['learning_rate']:.2e}", b["steps_per_epoch"],
                             f"{h['best_iou']:.4f}", len(h["evaluations"]), f"{h['seconds'] / 60:.0f}"])
        md.append("## Hyper-parameter optimisation (Eq. 6, Algorithm 1)\n")
        md.append(table(["Dataset", "Optimizer", "Hidden neurons", "Learning rate", "Steps/epoch", "Best val IoU (proxy)",
                         "Trainings", "Minutes"], rows))
        fig, axes = plt.subplots(1, len(hpo), figsize=(4.2 * len(hpo), 3.2), squeeze=False)
        for ax, (ds, d) in zip(axes[0], hpo.items()):
            for opt, h in d.items():
                ax.plot(range(1, len(h["curve"]) + 1), h["curve"], marker="o", ms=4, lw=2, color=COLORS.get(opt), label=opt)
            ax.set_title(DS_LABEL.get(ds, ds), fontsize=9)
            ax.set_xlabel("iteration")
            ax.set_ylabel("best fitness  1/IoU  (lower is better)")
            ax.legend(frameon=False)
        fig.suptitle("Fig. 10 analogue - HPO convergence", fontsize=10)
        fig.savefig(os.path.join(rep, "fig10_hpo_convergence.png"))
        plt.close(fig)
        md.append("\n![HPO convergence](fig10_hpo_convergence.png)\n")

    # ---------------------------------------------------------------- optimizer benchmark
    bpath = os.path.join(os.path.dirname(res.rstrip("/")), "optimizer_benchmark.json")
    if os.path.exists(bpath):
        bm = json.load(open(bpath))
        rows, wins = [], {"eco": 0, "runu-eco": 0}
        for f, d in bm.items():
            rows.append([f] + [f"{np.mean(d[o]['final']):.3g} / {np.median(d[o]['final']):.3g}" for o in ("eco", "runu-eco", "random")])
            wins["runu-eco" if np.median(d["runu-eco"]["final"]) < np.median(d["eco"]["final"]) else "eco"] += 1
        md.append("## ECO vs RUNU-ECO on benchmark functions (10-D, pop 10, 50 iters, 20 seeds; mean / median final value)\n")
        md.append(table(["Function", "ECO", "RUNU-ECO", "Random search"], rows))
        md.append(f"\nRUNU-ECO has the lower median on {wins['runu-eco']}/{len(bm)} functions.\n")
        fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
        for ax, (f, d) in zip(axes.flat, bm.items()):
            for o in ("eco", "runu-eco", "random"):
                ax.semilogy(d[o]["mean_curve"], lw=2, color=COLORS[o], label=o)
            ax.set_title(f, fontsize=9)
        axes.flat[0].legend(frameon=False)
        fig.suptitle("ECO vs RUNU-ECO - mean best-so-far over 20 seeds", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(rep, "fig_optimizer_benchmark.png"))
        plt.close(fig)
        md.append("\n![optimizer benchmark](fig_optimizer_benchmark.png)\n")

    for ds in [d for d in ("brats", "figshare", "lgg") if d in cv]:
        R = cv[ds]
        md.append(f"\n## {DS_LABEL[ds]}\n")
        nf = {t: len(r) for t, r in R.items()}
        md.append("Folds completed: " + ", ".join(f"{LABEL.get(t, t)}={n}" for t, n in nf.items()) + "\n")
        # Table 6 analogue
        rows = [[LABEL[t]] + [pm(fold_values(R[t], m)) for m in ("dice", "precision", "sensitivity", "accuracy")]
                for t in COMPARE if t in R]
        md.append("\n**Table 6 analogue - k-fold comparison**\n\n" + table(["Model", "Dice", "Precision", "Recall", "Accuracy"], rows))
        if ds == "lgg":
            rows = [[LABEL[t]] + [pm(fold_values(R[t], m, True)) for m in ("dice", "iou", "precision", "sensitivity")]
                    for t in ALL if t in R]
            md.append("\n*LGG, tumour-positive slices only* (empty slices count as perfect in the table above)\n\n"
                      + table(["Model", "Dice", "IoU", "Precision", "Recall"], rows))
        # Table 7 analogue (over folds)
        rows = []
        for m in ("dice", "iou", "accuracy"):
            for t in ALL:
                if t in R:
                    d = describe(fold_values(R[t], m))
                    rows.append([m, LABEL[t]] + [f"{d[k]:.4f}" for k in ("best", "worst", "mean", "median", "std")])
        md.append("\n**Table 7 analogue - statistics over folds**\n\n" + table(["Metric", "Model", "Best", "Worst", "Mean", "Median", "Std"], rows))
        # Table 8 analogue
        rows = [[LABEL[t]] + [pm(fold_values(R[t], m)) for m in ("dice", "iou", "accuracy", "sensitivity", "f1")]
                for t in ABLATION if t in R]
        md.append("\n**Table 8 analogue - ablation**\n\n" + table(["Variant", "Dice", "IoU", "Accuracy", "Recall", "F1"], rows))
        # Table 9 analogue
        rows = [[m] + [f"{100 * np.mean(fold_values(R[t], m)):.3f}" for t in ALL if t in R] for m in METRIC_NAMES]
        md.append("\n**Table 9 analogue - all indicators (mean over folds, %)**\n\n"
                  + table(["Metric"] + [LABEL[t] for t in ALL if t in R], rows))
        # params / time
        rows = []
        for t in ALL:
            if t in R:
                hs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(res, "cv", ds, t, "fold*", "history.json")))]
                sec = np.mean([h["time"][-1] for h in hs]) if hs else float("nan")
                c = R[t][0]["config"]
                rows.append([LABEL[t], c["epochs"], c["steps_per_epoch"], f"{c['learning_rate']:.2e}", c["hidden_neurons"], f"{sec / 60:.1f}"])
        md.append("\n**Training settings actually used**\n\n" + table(["Model", "Epochs", "Steps/epoch", "LR", "Hidden neurons", "Train min/fold"], rows))
        _fig9(R, ds, rep)
        _fig11(res, ds, rep)
        _fig8(res, ds, R, rep)
        md.append(f"\n![metrics]({ds}_fig9_metrics.png)\n\n![loss]({ds}_fig11_loss.png)\n\n![qualitative]({ds}_fig8_qualitative.png)\n")
    open(os.path.join(rep, "REPORT.md"), "w").write("\n".join(md))
    print("wrote", os.path.join(rep, "REPORT.md"))


def _fig9(R, ds, rep):
    tags = [t for t in ALL if t in R]
    mets = ["accuracy", "dice", "iou", "f1", "sensitivity"]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    w = 0.8 / len(tags)
    for k, t in enumerate(tags):
        vals = [np.mean(fold_values(R[t], m)) for m in mets]
        errs = [np.std(fold_values(R[t], m)) for m in mets]
        ax.bar(np.arange(len(mets)) + (k - len(tags) / 2 + 0.5) * w, vals, w * 0.9, yerr=errs, color=COLORS[t],
               label=LABEL[t], error_kw={"lw": 0.8, "ecolor": "#52514e"})
    ax.set_xticks(range(len(mets)), ["Accuracy", "Dice", "IoU", "F1", "Recall"])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title(f"Fig. 9 analogue - {DS_LABEL[ds]}", fontsize=10, pad=34)
    fig.savefig(os.path.join(rep, f"{ds}_fig9_metrics.png"))
    plt.close(fig)


def _fig11(res, ds, rep):
    hs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(res, "cv", ds, "runu_eco_svaeunetpp", "fold*", "history.json")))]
    if not hs:
        return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for i, h in enumerate(hs):
        ax.plot(h["epoch"], h["train_loss"], color="#2a78d6", lw=2, alpha=0.9, label="train" if i == 0 else None)
        ax.plot(h["epoch"], h["val_loss"], color="#eb6834", lw=2, alpha=0.9, label="validation" if i == 0 else None)
    ax.set_xlabel("epoch")
    ax.set_ylabel("Dice + BCE loss")
    ax.legend(frameon=False)
    ax.set_title(f"Fig. 11 analogue - RUNU-ECO-SVAEUnet++ ({len(hs)} folds)\n{DS_LABEL[ds]}", fontsize=9)
    fig.savefig(os.path.join(rep, f"{ds}_fig11_loss.png"))
    plt.close(fig)


def _fig8(res, ds, R, rep, n=4):
    from svae.experiment import DataBundle
    tags = [t for t in ("runu_eco_svaeunetpp", "unet", "unetpp", "unet3p") if t in R]
    preds = {}
    for t in tags:
        p = os.path.join(res, "cv", ds, t, "fold0", "preds.npz")
        if os.path.exists(p):
            d = np.load(p)
            preds[t] = dict(zip(d["idx"].tolist(), d["preds"]))
    if "runu_eco_svaeunetpp" not in preds:
        return
    tags = [t for t in tags if t in preds]
    b = DataBundle(ds)
    common = [i for i in preds["runu_eco_svaeunetpp"] if all(i in preds[t] for t in tags)]
    pos = [i for i in common if np.isin(b.seg[i], b_labels(ds)).any()]
    pick = pos[:: max(1, len(pos) // n)][:n]
    fig, axes = plt.subplots(len(pick), 2 + len(tags), figsize=(2.1 * (2 + len(tags)), 2.1 * len(pick)), squeeze=False)
    for r, i in enumerate(pick):
        gt = np.isin(b.seg[i], b_labels(ds))
        img = b.img[i][0].astype(np.float32)  # FLAIR for BraTS
        if b.dims == 3:
            z = int(np.argmax(gt.sum((0, 1))))
            sl = lambda v: v[..., z]  # noqa: E731
        else:
            sl = lambda v: v  # noqa: E731
        base = np.rot90(sl(img))
        cols = [("Image", None), ("Ground truth", gt)] + [(LABEL[t], preds[t][i].astype(bool)) for t in tags]
        for c, (name, m) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(base, cmap="gray")
            if m is not None:
                mm = np.rot90(sl(m))
                ov = np.zeros(mm.shape + (4,))
                ov[mm] = matplotlib.colors.to_rgba("#eda100" if name == "Ground truth" else "#2a78d6", 0.55)
                ax.imshow(ov)
                if name != "Ground truth":
                    g = np.rot90(sl(gt))
                    dice = 2 * (mm & g).sum() / max(1, mm.sum() + g.sum())
                    ax.text(2, 10, f"Dice {dice:.2f}", color="white", fontsize=7)
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(name, fontsize=8)
    fig.suptitle(f"Fig. 8 analogue - {DS_LABEL[ds]} (fold-0 test cases)", fontsize=10)
    fig.savefig(os.path.join(rep, f"{ds}_fig8_qualitative.png"))
    plt.close(fig)


def b_labels(ds):
    return (1, 2, 3) if ds == "brats" else (1,)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/quick")
    main(ap.parse_args().results)
