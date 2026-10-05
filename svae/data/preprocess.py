"""Standardised preprocessing / harmonisation pipeline (paper Sec. 3.3, Steps 1-5).

Step 1  Spatial normalisation : 3D -> crop to brain bounding box, resample to fixed voxel spacing,
                                centre crop/pad to a fixed shape.   2D -> resize to fixed resolution.
Step 2  Intensity normalisation: per-image (per-modality) z-score  I_norm = (I - mu) / sigma  over brain voxels.
Step 3  Intensity clipping     : clip to [0.5, 99.5] percentiles *before* z-scoring.
Step 4  Modality harmonisation : BraTS modalities are already co-registered; each is normalised independently.
Step 5  Label harmonisation    : every dataset is mapped to a unified label space.
          BraTS {0,1,2,4} -> {0: bg, 1: NCR/NET, 2: ED, 3: ET}; binary targets (WT/TC/ET) derived at load time.
          2D datasets       -> {0: bg, 1: tumour}.
(Step 6, augmentation, lives in transforms.py.)
"""
import io
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage

MODALITIES = ("flair", "t1", "t1ce", "t2")


def clip_and_zscore(img, mask, lo=0.5, hi=99.5):
    out = np.zeros_like(img, dtype=np.float32)
    vals = img[mask]
    if vals.size == 0:
        return out
    a, b = np.percentile(vals, [lo, hi])
    vals = np.clip(vals, a, b)
    out[mask] = (vals - vals.mean()) / (vals.std() + 1e-8)
    return out


def center_crop_pad(arr, shape):
    """Crop/pad the trailing len(shape) axes of arr to `shape`, centred."""
    out = arr
    for ax_off, target in enumerate(shape):
        ax = arr.ndim - len(shape) + ax_off
        n = out.shape[ax]
        if n > target:
            s = (n - target) // 2
            out = np.take(out, range(s, s + target), axis=ax)
        elif n < target:
            before = (target - n) // 2
            pad = [(0, 0)] * out.ndim
            pad[ax] = (before, target - n - before)
            out = np.pad(out, pad)
    return out


# ------------------------------------------------------------------------------------------- BraTS

def _find_case_files(names):
    cases = {}
    for n in names:
        if not n.endswith(".nii") and not n.endswith(".nii.gz"):
            continue
        if "TrainingData" not in n:  # the validation set has no ground truth
            continue
        case = n.split("/")[-2]
        fn = n.split("/")[-1].lower()
        key = next((m for m in MODALITIES if fn.endswith(f"_{m}.nii") or fn.endswith(f"_{m}.nii.gz")), None)
        if key is None and ("seg" in fn):  # also catches subject 355's misnamed '..._Segm.nii'
            key = "seg"
        if key:
            cases.setdefault(case, {})[key] = n
    return {c: f for c, f in cases.items() if len(f) == 5}


def _load_nii(z, name):
    import nibabel as nib
    data = z.read(name)
    if name.endswith(".gz"):
        import gzip
        data = gzip.decompress(data)
    img = nib.Nifti1Image.from_bytes(data)
    return np.asarray(img.dataobj, dtype=np.float32), img.header.get_zooms()[:3]


def _process_brats_case(args):
    zip_path, case, files, out_dir, spacing, shape = args
    out_path = os.path.join(out_dir, f"{case}.npz")
    if os.path.exists(out_path):
        return case, "cached"
    z = zipfile.ZipFile(zip_path)
    vols, zooms = [], None
    for m in MODALITIES:
        v, zooms = _load_nii(z, files[m])
        vols.append(v)
    seg, _ = _load_nii(z, files["seg"])
    img = np.stack(vols)                                  # (4, 240, 240, 155)
    brain = (img > 0).any(0)
    # Step 1a: brain bounding-box crop
    idx = np.argwhere(brain)
    lo, hi = idx.min(0), idx.max(0) + 1
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    img, seg, brain = img[(slice(None),) + sl], seg[sl], brain[sl]
    # Steps 3,2,4: per-modality clip + z-score on brain voxels
    img = np.stack([clip_and_zscore(img[c], brain) for c in range(img.shape[0])])
    # Step 1b: resample to fixed spacing (linear for images, nearest for labels)
    zoom = np.array(zooms, dtype=np.float32) / np.array(spacing, dtype=np.float32)
    img = np.stack([ndimage.zoom(img[c], zoom, order=1) for c in range(img.shape[0])])
    seg = ndimage.zoom(seg, zoom, order=0)
    # Step 1c: crop / pad to fixed shape
    img = center_crop_pad(img, shape)
    seg = center_crop_pad(seg, shape)
    # Step 5: label harmonisation {0,1,2,4} -> {0,1,2,3}
    seg = seg.astype(np.uint8)
    seg[seg == 4] = 3
    np.savez_compressed(out_path, img=img.astype(np.float16), seg=seg)
    return case, f"ok {img.shape} tumour_vox={int((seg > 0).sum())}"


def preprocess_brats(zip_path, out_dir, spacing=(1.6, 1.6, 1.6), shape=(96, 112, 96), workers=4):
    os.makedirs(out_dir, exist_ok=True)
    cases = _find_case_files(zipfile.ZipFile(zip_path).namelist())
    print(f"BraTS2020: {len(cases)} training subjects with labels")
    jobs = [(zip_path, c, f, out_dir, spacing, shape) for c, f in sorted(cases.items())]
    with ProcessPoolExecutor(workers) as ex:
        for i, (case, msg) in enumerate(ex.map(_process_brats_case, jobs), 1):
            if i % 20 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] {case}: {msg}", flush=True)
    pack_brats(out_dir)


def pack_brats(out_dir):
    """Pack per-case npz files into two memory-mappable arrays (fast random access, low RAM)."""
    import glob
    files = sorted(glob.glob(os.path.join(out_dir, "*.npz")))
    first = np.load(files[0])
    img = np.lib.format.open_memmap(os.path.join(out_dir, "brats_img.npy"), "w+", np.float16,
                                    (len(files),) + first["img"].shape)
    seg = np.lib.format.open_memmap(os.path.join(out_dir, "brats_seg.npy"), "w+", np.uint8,
                                    (len(files),) + first["seg"].shape)
    for i, f in enumerate(files):
        d = np.load(f)
        img[i], seg[i] = d["img"], d["seg"]
    img.flush(), seg.flush()
    with open(os.path.join(out_dir, "brats_cases.txt"), "w") as fh:
        fh.write("\n".join(os.path.basename(f)[:-4] for f in files))
    print(f"packed {len(files)} cases -> brats_img.npy {img.shape}, brats_seg.npy {seg.shape}")


# ---------------------------------------------------------------------------------------------- 2D

def _process_2d_pair(args):
    img_path, mask_path, size = args
    from PIL import Image
    img = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)
    msk = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32)
    img = np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR), dtype=np.float32)
    msk = np.asarray(Image.fromarray(msk).resize((size, size), Image.NEAREST)) > 127
    fg = img > 0.02 * img.max() if img.max() > 0 else np.ones_like(img, bool)
    img = clip_and_zscore(img, fg)
    return img.astype(np.float16), msk.astype(np.uint8)


def preprocess_2d(pairs, out_path, size=128, workers=4):
    """pairs: list of (image_path, mask_path, group_id). group_id keeps a patient's slices in one fold."""
    jobs = [(i, m, size) for i, m, _ in pairs]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_process_2d_pair, jobs, chunksize=32))
    imgs = np.stack([r[0] for r in res])[:, None]
    segs = np.stack([r[1] for r in res])
    groups = np.array([g for _, _, g in pairs])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, img=imgs, seg=segs, groups=groups)
    print(f"saved {out_path}: {imgs.shape}, tumour-positive slices={(segs.reshape(len(segs), -1).max(1) > 0).sum()}")
