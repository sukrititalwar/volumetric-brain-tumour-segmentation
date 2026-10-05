# RUNU-ECO-SVAEUnet++ — implementation of Krishnaveni & Kollem (2026)

Reproduction of *"A heuristic-inspired spatial ViT-based adaptive 3D efficient Unet++ module for volumetric brain
tumour segmentation model with 3D brain images"*, Biomedical Signal Processing and Control 122 (2026) 110403.

The paper releases no code and leaves several components under-specified, so this repo implements every stage of
its pipeline and documents each interpretation and deviation below.

```
data download ─► preprocessing (Steps 1-6) ─► SVAEUnet++ ─► RUNU-ECO HPO (max IoU) ─► 5-fold CV ─► baselines / ablation / report
```

## Layout

| Path | What |
|---|---|
| `svae/models/blocks.py` | Spatial attention (Eq. 1), ViT encoder (Eqs. 2–4), MBConv (EfficientNet), dilated dense residual block |
| `svae/models/svaeunetpp.py` | **SVAEUnet++**: EfficientNet + SViT encoder, UNet++ nested dense decoder (Eq. 5), deep supervision. Switches give the Table 8 ablation variants |
| `svae/models/baselines.py` | UNet, UNet++, UNet 3+ (2D/3D) |
| `svae/optim/eco.py` | **ECO** (line-by-line port of the authors' MATLAB) and **RUNU-ECO** (Eq. 8), plus random search |
| `svae/data/preprocess.py` | Spatial normalisation, percentile clipping, z-score, modality and label harmonisation |
| `svae/data/datasets.py` | Patch sampling, augmentation (Step 6), grouped 5-fold CV |
| `svae/train.py` | Training/eval engine and the HPO objective `1/IoU` (Eq. 6) |
| `svae/metrics.py` | Dice, IoU, accuracy, sensitivity, specificity, precision, F1, MCC, FPR, FNR, NPV, FDR (Table 9) |
| `scripts/` | `download_data`, `preprocess`, `hpo`, `cross_validate`, `benchmark_optimizers`, `report`, `run_all` |

## Quick start

```bash
pip install -r requirements.txt
python scripts/download_data.py brats figshare lgg
python scripts/preprocess.py brats && python scripts/preprocess.py figshare && python scripts/preprocess.py lgg
python scripts/benchmark_optimizers.py
python scripts/run_all.py --profile quick      # reduced protocol (~10 h on an 8 GB M1)
python scripts/run_all.py --profile paper      # the paper's settings (GPU + days)
```

`run_all.py` caches every finished step, so you can re-run it to resume. The report is written to
`results/<profile>/report/REPORT.md`.

## Datasets

| Paper | What the link actually contains | Used here |
|---|---|---|
| **Dataset 1**: BraTS2020 (Kaggle `darksteeldragon/...`, 42.8 GB) | 369 labelled subjects; T1, T1ce, T2, FLAIR | Same 369 subjects from `awsaf49/brats20-dataset-training-validation` (4.5 GB zip). Target: whole tumour |
| **Dataset 2**: Drive `1pvd75_y…` | Sartaj's **classification** set (glioma/meningioma/pituitary/no tumour, 3,264 JPGs), **no masks** | **Figshare / Cheng et al.** 3,064 T1-CE slices of the same three tumour types, with expert masks |
| **Dataset 3**: Drive `1BoBoZV0…` | Bohaju "Brain Tumor" (3,762 JPGs + texture CSV, binary label), **no masks** | **TCGA-LGG** FLAIR (Buda et al.), 110 patients, 3,929 slices, with masks; folds grouped by patient |

Datasets 2 and 3 as published cannot give segmentation Dice/IoU because they have no ground-truth masks. So the paper's
numbers on them can't be reproduced as stated. I verified this by listing both Drive folders. Each was replaced by
the closest public dataset that has masks. `download_data.py paper_d2 paper_d3` still fetches the originals.

## Paper → code mapping and interpretations

| Paper | Implementation |
|---|---|
| Preprocessing Steps 1–5 | Brain bounding-box crop, resample to 1.6 mm isotropic, centre crop/pad to 96×112×96, clip to the 0.5–99.5 percentile, z-score per modality over brain voxels, BraTS labels {1,2,4} → WT/TC/ET. 2D: resize to 128², same clipping and z-score |
| Step 6 augmentation | Random flips, 90° rotations, scaling (0.85–1.15), intensity scale/shift |
| SViT (Sec. 4.1) | Spatial attention (Eq. 1) → patch embedding + class token + learned positions (Eq. 2) → pre-LN MSA/MLP blocks (Eqs. 3–4), folded back and fused residually. Used in every encoder down-sampling layer; patch size chosen so there are ≤ 512 tokens |
| EfficientNet encoder | MBConv with squeeze-excitation. Fused-MBConv down-sampling and a stride-2 stem (as in EfficientNetV2) |
| "3D depth separable convolution" | Depthwise-separable decoder nodes. On Apple silicon, 3D depthwise convs are factorised into (k×k)+(k×1) depthwise convs: native grouped `Conv3d` on MPS is about 70× slower. Set `blocks.FACTORIZE_DW3D=False` on CUDA |
| "Dilated dense residual block" | Bottleneck: dense concatenation of dilated (1, 2, 3) depthwise convs + residual |
| UNet++ decoder (Eq. 5) | Full nested X^{i,j} grid. Up-sampling is trilinear/bilinear interpolation + 1×1 conv. Deep supervision on X^{0,1..4} |
| RUNU-ECO (Eq. 8, Alg. 1) | ECO's uniform R1/R2 replaced by `N = (F_cur + F_best) / (2·F_worst)`. The extracted equation reads `er+ty / rt2`; this reading keeps N in (0, 1]. `runu_mode="square"` gives the literal rt² |
| Tuned hyper-parameters | Chromosome = (hidden neurons ∈ [5, 255] → SViT MLP width, learning rate ∈ [1e-4, 1e-2] (Table 3; Eq. 6's "0.01–0.99" is not a usable Adam LR), steps/epoch ∈ [100, 500]). Fitness = 1/IoU on a validation split taken from the training folds only |
| Evaluation | 5-fold CV (Table 4). Each fold holds out 15 % of its training part for checkpoint selection. Eq. 10 in the paper calls 2TP/(2TP+FP+FN) "IoU"; that is Dice. Here IoU is the standard TP/(TP+FP+FN) |
| Baselines | UNet, UNet++, UNet 3+ with the same widths, loss, schedule and epochs (Table 5 fairness). In 3D they get the same stride-2 stem as SVAEUnet++: full-resolution 3D UNet 3+ costs 11 s/step on MPS |
| Not implemented | FSA, SCO and SAA comparison optimizers (equations not given in the paper; I didn't want to invent them). ECO and random search are the comparison points. The `OPTIMIZERS` registry takes new ones |

## Compute note

The paper's protocol is 10 × 50 RUNU-ECO evaluations, each a full 50-epoch training, then 5-fold CV of 8+ models on
3 datasets. That is thousands of GPU-hours. The `quick` profile keeps every stage but shrinks the budgets:

* **HPO**: population 6, 4 iterations, 3 epochs, 10 % of the steps, 60 training and 12 validation subjects. This is a low-fidelity proxy.
* **CV**: 3 of the 5 folds, 15 epochs, 30 % of the steps per epoch.

Absolute scores are therefore below what longer training gives, but every model gets the same budget.

## Results

See `results/quick/report/REPORT.md`; the headline numbers are summarised at the end of the run.
