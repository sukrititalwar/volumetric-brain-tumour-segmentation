"""Evaluation indicators (paper Sec. 6.2, Table 9) and the segmentation loss.

Note on the paper's equations: Eq. (10) labels 2TP/(2TP+FP+FN) as IoU, which is actually Dice/F1.
We use the standard definitions: IoU (Jaccard) = TP/(TP+FP+FN), Dice = F1 = 2TP/(2TP+FP+FN).
"""
import numpy as np
import torch
import torch.nn.functional as F

METRIC_NAMES = ["dice", "iou", "accuracy", "sensitivity", "specificity", "precision", "f1",
                "mcc", "fpr", "fnr", "npv", "fdr"]


def confusion(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.count_nonzero(pred & gt)
    fp = np.count_nonzero(pred & ~gt)
    fn = np.count_nonzero(~pred & gt)
    tn = pred.size - tp - fp - fn
    return tp, fp, fn, tn


def metrics_from_confusion(tp, fp, fn, tn):
    tp, fp, fn, tn = map(float, (tp, fp, fn, tn))
    d = lambda a, b, empty=1.0: a / b if b > 0 else empty  # noqa: E731 (empty/empty -> perfect)
    m = {
        "dice": d(2 * tp, 2 * tp + fp + fn),
        "iou": d(tp, tp + fp + fn),
        "accuracy": d(tp + tn, tp + tn + fp + fn),
        "sensitivity": d(tp, tp + fn),
        "specificity": d(tn, tn + fp),
        "precision": d(tp, tp + fp),
        "npv": d(tn, tn + fn),
    }
    m["f1"] = m["dice"]
    m["fpr"], m["fnr"], m["fdr"] = 1 - m["specificity"], 1 - m["sensitivity"], 1 - m["precision"]
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    m["mcc"] = (tp * tn - fp * fn) / den if den > 0 else float(m["dice"] == 1.0)
    return m


def case_metrics(pred, gt):
    return metrics_from_confusion(*confusion(pred, gt))


def summarize(per_case):
    """Mean over cases for every metric."""
    return {k: float(np.mean([c[k] for c in per_case])) for k in METRIC_NAMES}


def describe(values):
    """Best / Worst / Mean / Median / Std, as in Table 7."""
    v = np.asarray(values, dtype=float)
    return {"best": v.max(), "worst": v.min(), "mean": v.mean(), "median": float(np.median(v)), "std": v.std()}


class DiceBCELoss(torch.nn.Module):
    """Soft Dice + BCE, with weighted deep supervision when the model returns a list of outputs."""

    def __init__(self, ds_weights=(0.125, 0.25, 0.5, 1.0)):
        super().__init__()
        self.ds_weights = ds_weights

    @staticmethod
    def _single(logits, target):
        p = torch.sigmoid(logits)
        dims = tuple(range(2, p.ndim))
        inter = (p * target).sum(dims)
        dice = 1 - (2 * inter + 1) / (p.sum(dims) + target.sum(dims) + 1)
        return dice.mean() + F.binary_cross_entropy_with_logits(logits, target)

    def forward(self, out, target):
        if isinstance(out, (list, tuple)):
            w = self.ds_weights[-len(out):]
            return sum(wi * self._single(o, target) for wi, o in zip(w, out)) / sum(w)
        return self._single(out, target)
