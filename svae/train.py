"""Training / evaluation engine shared by every experiment.

The three hyper-parameters tuned by RUNU-ECO (chromosome length 3, Eq. 6 / Algorithm 1):
    hidden_neurons  in [5, 255]   -> hidden width of the SViT MLP (Eq. 4)
    learning_rate   in [1e-4, 1e-2] (Table 3; searched in log10 space)
    steps_per_epoch in [100, 500]
"""
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import DiceBCELoss, case_metrics, summarize
from .models import build_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    model: str = "svaeunetpp"
    dims: int = 3
    in_ch: int = 4
    base: int = 16
    patch: tuple = (64, 64, 64)
    hidden_neurons: int = 128
    learning_rate: float = 1e-3
    steps_per_epoch: int = 250
    epochs: int = 50
    batch_size: int = 2
    weight_decay: float = 1e-5
    num_workers: int = 2
    seed: int = 0
    amp: bool = False
    log_every: int = 0
    extra: dict = field(default_factory=dict)


def build(cfg: TrainConfig):
    kw = dict(in_ch=cfg.in_ch, n_classes=1, dims=cfg.dims, base=cfg.base)
    if cfg.model in ("svaeunetpp", "svit_eunetpp", "efficient_unet", "efficient_unetpp"):
        kw.update(patch_size=cfg.patch, mlp_hidden=int(cfg.hidden_neurons))
    return build_model(cfg.model, **kw)


def _pad_to_multiple(x, m=16):
    pads = []
    for s in reversed(x.shape[2:]):
        p = (-s) % m
        pads += [0, p]
    return torch.nn.functional.pad(x, pads), x.shape[2:]


@torch.no_grad()
def predict(model, x, device):
    """Whole-image / whole-volume inference (inputs padded to a multiple of 16)."""
    model.eval()
    xp, shape = _pad_to_multiple(x.to(device))
    logits = model(xp)
    sl = (slice(None), slice(None)) + tuple(slice(0, s) for s in shape)
    return torch.sigmoid(logits[sl]).float().cpu()


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=1, return_preds=False, loss_fn=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    per_case, preds, losses = [], [], []
    for x, y in loader:
        model.eval()
        xp, shape = _pad_to_multiple(x.to(device))
        logits = model(xp)
        logits = logits[(slice(None), slice(None)) + tuple(slice(0, s) for s in shape)]
        if loss_fn is not None:
            losses.append(loss_fn(logits, y.to(device)).item())
        p = (torch.sigmoid(logits).float().cpu() > 0.5)
        for pi, yi in zip(p.numpy(), y.numpy()):
            m = case_metrics(pi[0], yi[0] > 0.5)
            m["gt_positive"] = bool((yi[0] > 0.5).any())
            per_case.append(m)
            if return_preds:
                preds.append(pi[0].astype(np.uint8))
    out = summarize(per_case)
    out["per_case"] = per_case
    if loss_fn is not None:
        out["loss"] = float(np.mean(losses))
    if return_preds:
        out["preds"] = preds
    return out


def train(cfg: TrainConfig, train_set, val_set=None, out_dir=None, verbose=True):
    """Train with Adam (Table 3) + cosine LR; keep the checkpoint with the best validation IoU."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = get_device()
    model = build(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total = cfg.epochs * cfg.steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, total // 50)) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    loss_fn = DiceBCELoss()
    loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.num_workers, persistent_workers=cfg.num_workers > 0)
    it = iter(loader)
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_iou": [], "val_dice": [], "time": []}
    if not isinstance(train_set, torch.utils.data.Dataset) or len(train_set) < cfg.batch_size:
        raise ValueError("training set smaller than one batch")
    best, best_state = -1.0, None
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for _ in range(cfg.steps_per_epoch):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=cfg.amp):
                out = model(x)
                loss = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            opt.step()
            sched.step()
            if not math.isfinite(loss.item()):
                raise FloatingPointError("non-finite loss")
            losses.append(loss.item())
        hist["epoch"].append(ep)
        hist["train_loss"].append(float(np.mean(losses)))
        hist["time"].append(time.time() - t0)
        if val_set is not None:
            v = evaluate(model, val_set, device, loss_fn=loss_fn)
            hist["val_loss"].append(v["loss"])
            hist["val_iou"].append(v["iou"])
            hist["val_dice"].append(v["dice"])
            if v["iou"] > best:
                best = v["iou"]
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if verbose:
            msg = f"  ep {ep:3d}/{cfg.epochs} loss={hist['train_loss'][-1]:.4f}"
            if val_set is not None:
                msg += f" val_loss={hist['val_loss'][-1]:.4f} val_dice={hist['val_dice'][-1]:.4f} val_iou={hist['val_iou'][-1]:.4f}"
            print(msg + f" [{hist['time'][-1]:.0f}s]", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save({"cfg": asdict(cfg), "state_dict": model.state_dict()}, os.path.join(out_dir, "model.pt"))
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(hist, f)
    return model, hist


# ------------------------------------------------------------------------------- HPO objective

# search space: [hidden_neurons, log10(learning_rate), steps_per_epoch]
HPO_LB = [5, -4.0, 100]
HPO_UB = [255, -2.0, 500]


def decode(x):
    return {"hidden_neurons": int(round(x[0])), "learning_rate": float(10 ** x[1]), "steps_per_epoch": int(round(x[2]))}


def make_objective(base_cfg: TrainConfig, train_set, val_set, budget_scale=1.0, log=None):
    """Eq. (6): G = argmin 1/IoU over (hidden neurons, learning rate, steps per epoch).

    budget_scale < 1 shrinks the number of training steps per evaluation (low-fidelity proxy) so
    a population of 10 x 50 iterations is affordable on modest hardware.
    """
    cache = {}

    def fobj(x):
        hp = decode(x)
        key = tuple(hp.values())
        if key in cache:
            return cache[key]
        cfg = TrainConfig(**{**asdict(base_cfg), **hp})
        cfg.steps_per_epoch = max(1, int(round(hp["steps_per_epoch"] * budget_scale)))
        t = time.time()
        try:
            _, hist = train(cfg, train_set, val_set, verbose=False)
            iou = max(hist["val_iou"])  # IoU of the selected (best-validation) checkpoint
        except FloatingPointError:
            iou = 0.0
        f = 1.0 / max(iou, 1e-3)
        cache[key] = f
        if log is not None:
            log.append({**hp, "iou": iou, "fitness": f, "seconds": time.time() - t})
        print(f"    eval {hp} -> IoU={iou:.4f} fitness={f:.4f} ({time.time() - t:.0f}s)", flush=True)
        return f

    return fobj
