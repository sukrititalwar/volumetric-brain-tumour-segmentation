from .baselines import StridedStem, UNet, UNet3P, UNetPP
from .svaeunetpp import SVAEUnetPP


def build_model(name, **kw):
    """Model factory. Ablation variants of Table 8 are SVAEUnet++ with switches."""
    name = name.lower()
    baselines = {"unet": UNet, "unetpp": UNetPP, "unet++": UNetPP, "unet3p": UNet3P, "unet3+": UNet3P}
    if name in baselines:
        kw.pop("patch_size", None), kw.pop("mlp_hidden", None)
        if kw.get("dims", 3) == 3 and kw.pop("stem_stride", 2) == 2:
            return StridedStem(baselines[name], **kw)
        kw.pop("stem_stride", None)
        return baselines[name](**kw)
    if name == "efficient_unet":            # EfficientNet encoder + plain U-Net decoder, no SViT
        return SVAEUnetPP(use_svit=False, decoder="unet", **kw)
    if name == "efficient_unetpp":          # EfficientNet encoder + UNet++ decoder, no SViT
        return SVAEUnetPP(use_svit=False, decoder="unetpp", **kw)
    if name in ("svaeunetpp", "svit_eunetpp"):
        return SVAEUnetPP(use_svit=True, decoder="unetpp", **kw)
    raise ValueError(f"unknown model {name}")


MODELS = ["unet", "unetpp", "unet3p", "efficient_unet", "efficient_unetpp", "svaeunetpp"]
