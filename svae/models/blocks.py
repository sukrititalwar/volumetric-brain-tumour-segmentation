"""Building blocks for SVAEUnet++ (dimension-agnostic: works for 2D and 3D).

Paper components mapped to code:
  * Spatial attention, Eq. (1):  O(I) = sigmoid(Conv_pxp([AvgPool_c(H); MaxPool_c(H)])) * H
  * ViT encoder, Eqs. (2)-(4):   z0 = [z_cls || z_patch] + y_pos
                                 a_l = z_{l-1} + MSA(LN(z_{l-1}))
                                 z_l = a_l + MLP(LN(a_l))     (MLP hidden = "hidden neurons", tuned by RUNU-ECO)
  * EfficientNet encoder:        MBConv (inverted residual, depthwise-separable conv, squeeze-excitation)
  * "Dilated dense residual block" to enlarge the receptive field (Sec. 5.1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


FACTORIZE_DW3D = True  # set False to use native grouped nn.Conv3d (e.g. on CUDA)


def conv_nd(dims):
    return nn.Conv3d if dims == 3 else nn.Conv2d


def norm_nd(dims, ch):
    # InstanceNorm is the standard choice for small-batch 3D medical segmentation.
    return (nn.InstanceNorm3d if dims == 3 else nn.InstanceNorm2d)(ch, affine=True)


class DepthwiseConv3dFactorized(nn.Module):
    """3D depthwise conv as in-plane (k x k) depthwise conv followed by a through-plane (k x 1) one.

    Rationale: on Apple-silicon (MPS) a grouped/depthwise nn.Conv3d is ~70x slower than this
    factorised (2+1)D form, which uses fast 2D kernels. Receptive field is the same k^3 box; the
    kernel is rank-1 separable between plane and depth.
    """

    def __init__(self, ch, k=3, stride=1, d=1):
        super().__init__()
        self.hw = nn.Conv2d(ch, ch, k, stride, padding=d * (k // 2), dilation=d, groups=ch, bias=False)
        self.dd = nn.Conv2d(ch, ch, (k, 1), (stride, 1), padding=(d * (k // 2), 0), dilation=(d, 1),
                            groups=ch, bias=False)

    def forward(self, x):
        B, C, D, H, W = x.shape
        y = self.hw(x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W))
        H2, W2 = y.shape[-2:]
        y = self.dd(y.reshape(B, D, C, H2, W2).permute(0, 4, 2, 1, 3).reshape(B * W2, C, D, H2))
        return y.reshape(B, W2, C, y.shape[2], H2).permute(0, 2, 3, 4, 1)


class ConvNormAct(nn.Sequential):
    def __init__(self, dims, cin, cout, k=3, s=1, d=1, groups=1, act=True):
        Conv = conv_nd(dims)
        if dims == 3 and k > 1 and groups == cin == cout and FACTORIZE_DW3D:
            conv = DepthwiseConv3dFactorized(cin, k, s, d)
        else:
            conv = Conv(cin, cout, k, s, padding=d * (k // 2), dilation=d, groups=groups, bias=False)
        layers = [conv, norm_nd(dims, cout)]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class DoubleConv(nn.Sequential):
    """Plain UNet conv block (used by baselines and the UNet++ decoder nodes)."""

    def __init__(self, dims, cin, cout):
        super().__init__(ConvNormAct(dims, cin, cout), ConvNormAct(dims, cout, cout))


class DSConv(nn.Sequential):
    """Depthwise-separable conv block (cheap decoder node, Sec. 5.1 '3D depth separable convolution')."""

    def __init__(self, dims, cin, cout):
        super().__init__(ConvNormAct(dims, cin, cout, k=1),
                         ConvNormAct(dims, cout, cout, k=3, groups=cout),
                         ConvNormAct(dims, cout, cout, k=1))


class SqueezeExcite(nn.Module):
    def __init__(self, dims, ch, reduced):
        super().__init__()
        Conv = conv_nd(dims)
        self.dims = dims
        self.fc1 = Conv(ch, reduced, 1)
        self.fc2 = Conv(reduced, ch, 1)

    def forward(self, x):
        s = x.mean(dim=tuple(range(2, x.ndim)), keepdim=True)
        return x * torch.sigmoid(self.fc2(F.silu(self.fc1(s))))


class MBConv(nn.Module):
    """EfficientNet Mobile inverted Bottleneck Conv (expand -> depthwise -> SE -> project)."""

    def __init__(self, dims, cin, cout, stride=1, expand=4, k=3, drop_path=0.0):
        super().__init__()
        mid = cin * expand
        self.use_res = stride == 1 and cin == cout
        self.block = nn.Sequential(
            ConvNormAct(dims, cin, mid, k=1) if expand != 1 else nn.Identity(),
            ConvNormAct(dims, mid, mid, k=k, s=stride, groups=mid),
            SqueezeExcite(dims, mid, max(1, cin // 4)),
            ConvNormAct(dims, mid, cout, k=1, act=False),
        )
        self.drop_path = drop_path

    def forward(self, x):
        y = self.block(x)
        if self.use_res:
            if self.training and self.drop_path > 0:
                keep = torch.rand(x.shape[0], *([1] * (x.ndim - 1)), device=x.device) > self.drop_path
                y = y * keep / (1 - self.drop_path)
            y = y + x
        return y


class DilatedDenseResidual(nn.Module):
    """Dilated dense residual block: dense concatenation of dilated depthwise convs (d=1,2,3) + residual."""

    def __init__(self, dims, ch, dilations=(1, 2, 3)):
        super().__init__()
        self.branches = nn.ModuleList()
        c = ch
        for d in dilations:
            self.branches.append(nn.Sequential(ConvNormAct(dims, c, ch, k=1),
                                               ConvNormAct(dims, ch, ch, k=3, d=d, groups=ch)))
            c += ch
        self.fuse = ConvNormAct(dims, c, ch, k=1, act=False)

    def forward(self, x):
        feats = [x]
        for b in self.branches:
            feats.append(b(torch.cat(feats, 1)))
        return F.silu(x + self.fuse(torch.cat(feats, 1)))


class SpatialAttention(nn.Module):
    """Eq. (1): channel-wise avg & max pooling, concat, p x p conv, sigmoid gate."""

    def __init__(self, dims, k=7):
        super().__init__()
        self.conv = conv_nd(dims)(2, 1, k, padding=k // 2, bias=False)

    def forward(self, h):
        desc = torch.cat([h.mean(1, keepdim=True), h.amax(1, keepdim=True)], 1)
        return h * torch.sigmoid(self.conv(desc))


class TransformerBlock(nn.Module):
    """Eqs. (3)-(4): pre-LN MSA and MLP with residual connections."""

    def __init__(self, dim, heads, mlp_hidden, drop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.msa = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(mlp_hidden, dim), nn.Dropout(drop))

    def forward(self, z):
        h = self.ln1(z)
        a = z + self.msa(h, h, h, need_weights=False)[0]
        return a + self.mlp(self.ln2(a))


class SpatialViT(nn.Module):
    """Spatial Vision Transformer (Sec. 4.1, Fig. 3).

    Spatial attention produces the feature descriptor, which is split into patches, linearly
    embedded, prefixed with a class token, given learnable position embeddings (Eq. 2), passed
    through `depth` transformer blocks, then folded back to a feature map and fused residually.
    Position embeddings are interpolated when the input size differs from `grid` (sliding-window /
    full-volume inference).
    """

    def __init__(self, dims, ch, grid, patch=2, embed=None, depth=2, heads=4, mlp_hidden=128, drop=0.0):
        super().__init__()
        self.dims, self.patch = dims, patch
        embed = embed or ch
        assert embed % heads == 0
        self.sa = SpatialAttention(dims)
        Conv = conv_nd(dims)
        self.to_tokens = Conv(ch, embed, patch, stride=patch)  # linear patch projection
        self.grid = tuple(g // patch for g in grid)
        n = 1
        for g in self.grid:
            n *= g
        self.cls = nn.Parameter(torch.zeros(1, 1, embed))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, embed))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.blocks = nn.Sequential(*[TransformerBlock(embed, heads, mlp_hidden, drop) for _ in range(depth)])
        self.ln = nn.LayerNorm(embed)
        Up = nn.ConvTranspose3d if dims == 3 else nn.ConvTranspose2d
        self.from_tokens = Up(embed, ch, patch, stride=patch)
        self.out_norm = norm_nd(dims, ch)

    def _pos(self, grid):
        if grid == self.grid:
            return self.pos
        cls_pos, patch_pos = self.pos[:, :1], self.pos[:, 1:]
        e = patch_pos.shape[-1]
        patch_pos = patch_pos.transpose(1, 2).reshape(1, e, *self.grid)
        mode = "trilinear" if self.dims == 3 else "bilinear"
        patch_pos = F.interpolate(patch_pos, size=grid, mode=mode, align_corners=False)
        return torch.cat([cls_pos, patch_pos.flatten(2).transpose(1, 2)], 1)

    def forward(self, x):
        h = self.sa(x)
        t = self.to_tokens(h)
        grid = tuple(t.shape[2:])
        b, e = t.shape[:2]
        z = t.flatten(2).transpose(1, 2)
        z = torch.cat([self.cls.expand(b, -1, -1), z], 1) + self._pos(grid)
        z = self.ln(self.blocks(z))[:, 1:]  # drop class token
        t = z.transpose(1, 2).reshape(b, e, *grid)
        return x + self.out_norm(self.from_tokens(t))
