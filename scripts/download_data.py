"""Download every dataset into data/raw/ (no Kaggle account needed for these public datasets).

  python scripts/download_data.py brats figshare lgg      # what the pipeline trains on
  python scripts/download_data.py paper_d2 paper_d3       # the paper's own Drive folders (no masks, see README)

Dataset 1 (paper)  : BraTS2020 NIfTI. The paper's mirror (darksteeldragon/brats2020-nifti-format-for-deepmedic)
                     is 42.8 GB uncompressed; awsaf49/brats20-dataset-training-validation contains the same
                     369 training subjects as a 4.5 GB zip, which preprocessing reads without extracting.
Dataset 2 (paper)  : Drive folder 1pvd75_y-2dZRSc1qHEEoriJHLWGhHECP = Sartaj's 4-class *classification* set
                     (glioma/meningioma/pituitary/no tumour, 3,264 JPGs) - no segmentation masks.
  substitute       : Figshare (Cheng et al.) 3,064 T1-CE slices, same 3 tumour types, expert masks.
Dataset 3 (paper)  : Drive folder 1BoBoZV0ae9VuWg_EjNJccEBHpWeM9lQD = Bohaju "Brain Tumor" set
                     (3,762 JPGs + texture-feature CSV, binary label) - no segmentation masks.
  substitute       : TCGA-LGG FLAIR segmentation (Buda et al.), 110 patients, 3,929 slices, manual masks.
"""
import os
import subprocess
import sys
import zipfile

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/{}"
SOURCES = {
    "brats": ("awsaf49/brats20-dataset-training-validation", "brats2020.zip", False),
    "figshare": ("nikhilroxtomar/brain-tumor-segmentation", "brain-tumor-segmentation", True),
    "lgg": ("mateuszbuda/lgg-mri-segmentation", "lgg-mri-segmentation", True),
}
DRIVE = {"paper_d2": "https://drive.google.com/drive/folders/1pvd75_y-2dZRSc1qHEEoriJHLWGhHECP",
         "paper_d3": "https://drive.google.com/drive/folders/1BoBoZV0ae9VuWg_EjNJccEBHpWeM9lQD"}


def kaggle(slug, name, extract):
    zpath = os.path.join(RAW, name if name.endswith(".zip") else name + ".zip")
    if os.path.exists(zpath) or os.path.isdir(os.path.join(RAW, name)):
        print("exists:", name)
        return
    subprocess.check_call(["curl", "-L", "-A", "kagglehub/0.3", "-C", "-", "-o", zpath, KAGGLE.format(slug)])
    if extract:
        with zipfile.ZipFile(zpath) as z:
            z.extractall(os.path.join(RAW, name))
        os.remove(zpath)


if __name__ == "__main__":
    os.makedirs(RAW, exist_ok=True)
    for what in sys.argv[1:] or ["brats", "figshare", "lgg"]:
        if what in SOURCES:
            kaggle(*SOURCES[what])
        elif what in DRIVE:
            import gdown
            gdown.download_folder(DRIVE[what], output=os.path.join(RAW, what), quiet=False)
        else:
            raise SystemExit(f"unknown dataset {what}")
    # the LGG archive ships a duplicate nested copy; drop it
    dup = os.path.join(RAW, "lgg-mri-segmentation", "lgg-mri-segmentation")
    if os.path.isdir(dup):
        import shutil
        shutil.rmtree(dup)
