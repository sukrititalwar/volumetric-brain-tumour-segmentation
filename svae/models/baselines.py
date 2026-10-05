"""Baseline segmentation networks compared in the paper: UNet [30], UNet++ [30], UNet 3+ [36].

All are dimension-agnostic (dims=2 or 3) and use the same 5-level width schedule as SVAEUnet++.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvNormAct, DoubleConv, conv_nd


def _pool(dims):
    return nn.MaxPool3d(2) if dims == 3 else nn.MaxPool2d(2)


def _up(x, size, dims):
    return F.interpolate(x, size=size, mode="trilinear" if dims == 3 else "bilinear", align_corners=False)


class _Encoder(nn.Module):
    def __init__(self, dims, in_ch, ch):
        super().__init__()
        self.pool = _pool(dims)
        self.blocks = nn.ModuleList([DoubleConv(dims, in_ch if i == 0 else ch[i - 1], ch[i]) for i in range(len(ch))])

    def forward(self, x):
        feats = []
        for i, b in enumerate(self.blocks):
            x = b(x if i == 0 else self.pool(x))
            feats.append(x)
        return feats


class UNet(nn.Module):
    def __init__(self, in_ch=4, n_classes=1, dims=3, base=16, **_):
        super().__init__()
        self.dims = dims
        ch = [base * 2 ** i for i in range(5)]
        self.enc = _Encoder(dims, in_ch, ch)
        self.dec = nn.ModuleList([DoubleConv(dims, ch[i] + ch[i + 1], ch[i]) for i in range(4)])
        self.head = conv_nd(dims)(ch[0], n_classes, 1)

    def forward(self, x):
        f = self.enc(x)
        h = f[4]
        for i in range(3, -1, -1):
            h = self.dec[i](torch.cat([f[i], _up(h, f[i].shape[2:], self.dims)], 1))
        return self.head(h)


class UNetPP(nn.Module):
    """Zhou et al. UNet++ with dense nested skips (Eq. 5) and DoubleConv nodes."""

    def __init__(self, in_ch=4, n_classes=1, dims=3, base=16, **_):
        super().__init__()
        self.dims = dims
        ch = [base * 2 ** i for i in range(5)]
        self.enc = _Encoder(dims, in_ch, ch)
        self.nodes = nn.ModuleDict({f"{i}_{j}": DoubleConv(dims, ch[i] * j + ch[i + 1], ch[i])
                                    for j in range(1, 5) for i in range(5 - j)})
        self.head = conv_nd(dims)(ch[0], n_classes, 1)

    def forward(self, x):
        f = self.enc(x)
        X = {(i, 0): f[i] for i in range(5)}
        for j in range(1, 5):
            for i in range(5 - j):
                up = _up(X[(i + 1, j - 1)], f[i].shape[2:], self.dims)
                X[(i, j)] = self.nodes[f"{i}_{j}"](torch.cat([X[(i, k)] for k in range(j)] + [up], 1))
        return self.head(X[(0, 4)])


class UNet3P(nn.Module):
    """Huang et al. UNet 3+: every decoder node aggregates all encoder scales + deeper decoder nodes."""

    def __init__(self, in_ch=4, n_classes=1, dims=3, base=16, cat_ch=None, **_):
        super().__init__()
        self.dims = dims
        ch = [base * 2 ** i for i in range(5)]
        self.enc = _Encoder(dims, in_ch, ch)
        cat_ch = cat_ch or ch[0]
        up_ch = cat_ch * 5
        # for decoder level i: from encoder levels 0..i (downsampled/same), decoder levels i+1..4 (upsampled)
        self.proj = nn.ModuleDict()
        for i in range(4):
            for k in range(5):
                cin = ch[k] if (k <= i or k == 4) else up_ch
                self.proj[f"{i}_{k}"] = ConvNormAct(dims, cin, cat_ch)
            self.proj[f"fuse{i}"] = ConvNormAct(dims, up_ch, up_ch)
        self.head = conv_nd(dims)(up_ch, n_classes, 1)

    def forward(self, x):
        f = self.enc(x)
        D = {4: f[4]}
        pool = F.max_pool3d if self.dims == 3 else F.max_pool2d
        for i in range(3, -1, -1):
            size = f[i].shape[2:]
            parts = []
            for k in range(5):
                src = f[k] if k <= i else D[k]
                if k < i:
                    src = pool(src, 2 ** (i - k))
                elif k > i:
                    src = _up(src, size, self.dims)
                parts.append(self.proj[f"{i}_{k}"](src))
            D[i] = self.proj[f"fuse{i}"](torch.cat(parts, 1))
        return self.head(D[0])


class StridedStem(nn.Module):
    """Stride-2 conv stem + inner network at half resolution + logits upsampled back.

    Applied to the 3D baselines so they run under the same compute regime as SVAEUnet++ (whose
    EfficientNet stem is stride 2); full-resolution 3D UNet++/UNet3+ are 10-25x slower on MPS.
    """

    def __init__(self, inner_cls, in_ch=4, n_classes=1, dims=3, base=16, **kw):
        super().__init__()
        self.dims = dims
        self.stem = ConvNormAct(dims, in_ch, base, k=3, s=2)
        self.inner = inner_cls(in_ch=base, n_classes=n_classes, dims=dims, base=base, **kw)

    def forward(self, x):
        return _up(self.inner(self.stem(x)), x.shape[2:], self.dims)
