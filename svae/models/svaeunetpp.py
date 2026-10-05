"""SVAEUnet++: Spatial-ViT-based Adaptive (3D) Efficient UNet++ (Sec. 5.1-5.2, Fig. 6).

Encoder : input layer + 4 identical down-sampling layers. Each down-sampling layer =
          strided conv (Fused-MBConv "pooling") -> MBConv (EfficientNet) -> Spatial ViT.
Bottleneck: dilated dense residual block.
Decoder : UNet++ nested dense skip pathways, Eq. (5):
          X^{i,0} = encoder level i
          X^{i,j} = H([X^{i,0}, ..., X^{i,j-1}, Up(X^{i+1,j-1})]),  j > 0
          Up = (bi/tri)linear interpolation + 1x1 conv; H = depthwise-separable conv block.
Heads   : 1x1 conv on X^{0,1..4} (deep supervision during training; last head at inference).

Ablation switches (Table 8): `decoder` in {"unet", "unetpp"}, `use_svit`, and `dims` in {2, 3}
give Efficient-UNet, Efficient-UNet++, 3D Efficient-UNet++, SViT-EUNet++ (= SVAEUnet++).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvNormAct, DSConv, DilatedDenseResidual, MBConv, SpatialViT, conv_nd


class Up(nn.Module):
    def __init__(self, dims, cin, cout):
        super().__init__()
        self.mode = "trilinear" if dims == 3 else "bilinear"
        self.proj = ConvNormAct(dims, cin, cout, k=1)

    def forward(self, x, size):
        return self.proj(F.interpolate(x, size=size, mode=self.mode, align_corners=False))


class SVAEUnetPP(nn.Module):
    def __init__(self, in_ch=4, n_classes=1, dims=3, base=16, patch_size=(64, 64, 64),
                 mlp_hidden=128, vit_depth=1, heads=4, max_tokens=512, use_svit=True,
                 decoder="unetpp", deep_supervision=True, drop_path=0.05, expand=4, stem_stride=None):
        super().__init__()
        assert decoder in ("unet", "unetpp")
        self.dims, self.decoder, self.deep_supervision = dims, decoder, deep_supervision
        L = 5
        ch = [base * 2 ** i for i in range(L)]  # e.g. 16,32,64,128,256
        # EfficientNet uses a stride-2 stem; we do so in 3D (memory) and upsample the logits at the end.
        self.stem_stride = stem_stride or (2 if dims == 3 else 1)
        size = [p // self.stem_stride for p in patch_size[:dims]]

        # ---- encoder -------------------------------------------------------------------------
        self.stem = nn.Sequential(ConvNormAct(dims, in_ch, ch[0], s=self.stem_stride),
                                  MBConv(dims, ch[0], ch[0], expand=1))
        self.down = nn.ModuleList()
        for i in range(1, L):
            size = [s // 2 for s in size]
            # Fused-MBConv style down-sampling (EfficientNetV2): strided dense 3x3 conv, so that the
            # expanded MBConv tensors live at the lower resolution (keeps 3D memory manageable).
            layers = [ConvNormAct(dims, ch[i - 1], ch[i], k=3, s=2),
                      MBConv(dims, ch[i], ch[i], expand=expand, drop_path=drop_path)]
            if use_svit:
                # smallest power-of-two patch keeping token count <= max_tokens
                p = 1
                while _prod([s // p for s in size]) > max_tokens and min(size) // (2 * p) >= 1:
                    p *= 2
                emb = max(heads * 8, min(ch[i], 128))
                emb -= emb % heads
                layers.append(SpatialViT(dims, ch[i], size, patch=p, embed=emb, depth=vit_depth,
                                         heads=heads, mlp_hidden=mlp_hidden))
            self.down.append(nn.Sequential(*layers))
        self.bottleneck = DilatedDenseResidual(dims, ch[-1])

        # ---- decoder -------------------------------------------------------------------------
        self.nodes = nn.ModuleDict()
        self.ups = nn.ModuleDict()
        for j in range(1, L):
            for i in range(0, L - j):
                if decoder == "unet" and i + j != L - 1:
                    continue
                n_skip = j if decoder == "unetpp" else 1
                self.ups[f"{i}_{j}"] = Up(dims, ch[i + 1], ch[i])
                self.nodes[f"{i}_{j}"] = DSConv(dims, ch[i] * (n_skip + 1), ch[i])
        Conv = conv_nd(dims)
        n_heads = L - 1 if (decoder == "unetpp" and deep_supervision) else 1
        self.heads = nn.ModuleList([Conv(ch[0], n_classes, 1) for _ in range(n_heads)])

    def forward(self, x):
        out = self._forward(x)
        if self.stem_stride == 1:
            return out
        mode = "trilinear" if self.dims == 3 else "bilinear"
        up = lambda o: F.interpolate(o, size=x.shape[2:], mode=mode, align_corners=False)  # noqa: E731
        return [up(o) for o in out] if isinstance(out, list) else up(out)

    def _forward(self, x):
        X = {}
        h = self.stem(x)
        X[(0, 0)] = h
        for i, d in enumerate(self.down, start=1):
            h = d(h)
            X[(i, 0)] = h
        X[(4, 0)] = self.bottleneck(X[(4, 0)])
        L = 5
        for j in range(1, L):
            for i in range(0, L - j):
                key = f"{i}_{j}"
                if key not in self.nodes:
                    continue
                if self.decoder == "unetpp":
                    skips = [X[(i, k)] for k in range(j)]
                    below = X[(i + 1, j - 1)]
                else:  # plain U-Net path: X^{i,j} from X^{i,0} and the node directly below
                    skips = [X[(i, 0)]]
                    below = X[(i + 1, j - 1)]
                up = self.ups[key](below, X[(i, 0)].shape[2:])
                X[(i, j)] = self.nodes[key](torch.cat(skips + [up], 1))
        if len(self.heads) > 1:
            outs = [hd(X[(0, j)]) for hd, j in zip(self.heads, range(1, L))]
            return outs if self.training else outs[-1]
        return self.heads[0](X[(0, L - 1)])


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p
