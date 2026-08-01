# High-Resolution Water Segmentation for Salmon Habitat
**MSc Data Science — Queen Mary University of London**
**Student:** Luana Marotti (250016521) | **Supervisor:** Dr. Iran Roman

---

## Overview
This repository contains the supporting code for the dissertation *High-Resolution Water Segmentation for Salmon Habitat*. The project evaluates three methods for detecting water in high-resolution imagery.

**Methods evaluated:**
- NDWI (Normalised Difference Water Index) — spectral baseline
- OWM (OmniWaterMask) — pretrained deep learning baseline
- U-Net + ResNet34 — fine-tuned model (this work), evaluated as best single fold and 5-fold ensemble

**Primary metric:** Micro F1 across 4,788 held-out test chips from 13 test scenes.

---

## Results
Pooled micro F1 on the 4,788 held-out test chips:

| Method | Micro F1 | Precision | Recall |
|--------|:-------:|:---------:|:------:|
| **U-Net ensemble (majority vote)** | **0.8629** | 0.8392 | 0.8879 |
| U-Net best model (t=0.25) | 0.8413 | 0.7995 | 0.8877 |
| NDWI (t=0.35) | 0.5231 | 0.3760 | 0.8589 |
| OWM | 0.3440 | 0.2093 | 0.9662 |
| Dummy 100% water (baseline) | 0.0472 | 0.0242 | 1.0000 |
| Dummy 0% water (baseline) | 0.0000 | 0.0000 | 0.0000 |

The dummy baselines are metric-pipeline sanity checks (see `generate_water_masks.py`), not competing methods.

---

## Data
The dataset is not included in this repository. All scripts expect the following directory structure under a base directory (`BASE_DIR`):

```
BASE_DIR/
├── images/
│   ├── chips_images/images/     # 17,445 .tif chips (8-band, 4 used for modelling, 512×512)
│   └── chips_masks/masks/       # Binary water masks (same filenames)
└── data/
    └── nhd/
        └── NHD_flowlines.geojson  # NHD hydrological network (used by OWM)
```

`BASE_DIR` is read from the `BASE_DIR` environment variable in all notebooks and scripts (with an Azure ML path as the hardcoded fallback). Always set `BASE_DIR` explicitly before launching Jupyter so local clones do not silently fall back to the inaccessible Azure path — e.g. `export BASE_DIR=/path/to/project`.

Normalisation statistics and the test chip list are saved to `results/unet_training/` on first run and reloaded automatically on subsequent runs — do not delete these files.

---

## Environment
All code was developed and run on Microsoft Azure ML with an NVIDIA H100 NVL GPU (100 GB VRAM) under the `owm` conda environment.

To recreate the environment locally:

```bash
conda env create -f environment.yml
conda activate owm
```

Alternatively, using pip (Python 3.12 required):

```bash
pip install -r requirements.txt
# For GPU/CUDA builds, install torch via the official index:
#   pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
```

**Key dependencies:**
- Python 3.12
- PyTorch 2.3.0 (CUDA 12.1)
- segmentation-models-pytorch
- rasterio
- scikit-learn
- omniwatermask
- opencv-python (cv2)
- geopandas, contextily, shapely
- pandas, numpy, matplotlib, tqdm

---

## File Structure

```
./
├── README.md                              # This file
│
├── unet_training_final.ipynb              # Full U-Net training pipeline (Models 1–3)
├── unet_stability_experiments_1.ipynb     # Stability experiments: loss/LR sweep (Tversky, BCE, Focal+Dice @ LR=1e-4)
├── unet_stability_experiments_2.ipynb     # Stability experiments: reduced aug + BS=64 + gradient clipping
│
├── ndwi_evaluation_final.ipynb            # NDWI baseline evaluation
├── owm_scenes_final.ipynb                 # OWM baseline evaluation on reconstructed scenes
├── owm_experiments.ipynb                  # OWM configuration investigation (4 rounds)
├── unet_diagnostic_final.ipynb            # U-Net inference, threshold selection, best model + ensemble evaluation
├── comparison_all_final.ipynb             # Final method comparison table and figures
├── dataset_analysis.ipynb                 # Temporal, spatial, seasonal and water fraction analysis
│
├── generate_water_masks.py                # Generates dummy mask sets for sanity checks and pipeline baseline comparison
└── generate_figures.ipynb                 # Generates dissertation figures (see note in Step 8 for which are used in the document)
```

---

## Reproducing Results

> Steps must be run in order. Most heavy steps are cached, so re-opening a notebook is safe and will skip already-computed outputs.

### Step 1 — Scene-level split and normalisation (run once)
Open `unet_training_final.ipynb` and run **cells 1–3** in order. These establish the train/val/test split at scene level and compute normalisation statistics from the training set.

> ⚠️ Do not re-run cells 1–3 after the split is established — this would change the random seed and invalidate the test set.

### Step 2 — NDWI baseline
Run all cells in `ndwi_evaluation_final.ipynb`. Evaluates NDWI at t=0.35 (development threshold, selected via sweep on 51 development scenes) on the 4,788 test chips.

### Step 3 — OWM baseline
Run all cells in `owm_scenes_final.ipynb`. Reconstructs each test scene from its chips, runs OWM with OSM/NHD calibration, extracts per-chip predictions, and computes metrics.

> ⚠️ OWM inference is slow (~30 min per scene). Results are cached — cells skip scenes that already have predictions.

### Step 4 — Dataset analysis
Run all cells in `dataset_analysis.ipynb`. Produces temporal, spatial, seasonal, and water fraction figures for the train/val vs test split. Results are saved to `results/dataset_analysis/`.

### Step 5 — U-Net training (Model 3 only — Models 1 and 2 are historical)
In `unet_training_final.ipynb`, run cells in this order:

| Cell | Description | Notes |
|------|-------------|-------|
| 11 | Define `WaterChipDatasetV3` | Must run before cells 12–14 |
| 12 | Two-round hyperparameter sweep | Do not re-run — results saved |
| 13 | Full 5-fold k-fold | Do not re-run — results saved |
| 14 | K-fold restart (folds 4+5) | Do not re-run — results saved |

### Step 6 — U-Net evaluation (best model + ensemble)
Run all cells in `unet_diagnostic_final.ipynb` in order. This notebook:
- Loads the global best fold checkpoint (`best_model.pth`) and all 5 fold checkpoints (`fold1_best.pth` through `fold5_best.pth`)
- Runs inference on all 4,788 test chips and saves probability maps for both strategies
- **Best single fold:** threshold selected via sweep on fold 5 validation scenes (never seen during fold 5 training)
- **Ensemble (majority vote):** each fold model selects its own threshold on its own validation scenes; binary predictions are combined by majority vote (3/5 models predict water)
- Analyses per-scene performance, probability distributions, and visual inspection
- Saves `final_summary.csv` to `results/unet_diagnostic_v3/` for use in the comparison notebook

> ⚠️ Requires all 5 fold checkpoints to exist before running. Run after cell 14 of `unet_training_final.ipynb` is complete.

### Step 7 — Final comparison
Run all cells in `comparison_all_final.ipynb` in order. This notebook:
- Loads U-Net results from `final_summary.csv`
- Produces the final comparison table and figures across all methods: NDWI, OWM, U-Net best model, and U-Net ensemble (majority vote)

> ⚠️ Cell 1b generates NDWI binary masks at t=0.35 from raw chips (~25 min, runs once). This must complete before Cell 2. The fold reconstruction uses sklearn KFold with shuffle=True, random_state=42.

### Step 8 — Figures
Run all cells in `generate_figures.ipynb` to regenerate figures. The notebook generates seven figures (training curves, method comparison, pipeline diagram, geographic map, acquisition timeline, per-scene scatter, and qualitative examples); the dissertation uses five of these (pipeline, map, timeline, per-scene scatter, and qualitative).

---

## Stability Experiments
The `unet_stability_experiments_1.ipynb` and `unet_stability_experiments_2.ipynb` notebooks document intermediate experiments run before the final hyperparameter sweep. `unet_stability_experiments_1.ipynb` covers the loss-function and learning-rate selection (Tversky and BCE at LR=4e-4, then Focal+Dice at LR=1e-4, which was adopted); `unet_stability_experiments_2.ipynb` covers the training-stability interventions (reduced augmentation vs BS=64 + gradient clipping — the latter resolved the instability, so full augmentation was retained). These are retained as a historical record and are not required to reproduce the final results.

---

## OWM Configuration Experiments
`owm_experiments.ipynb` documents the systematic investigation used to select the final OWM configuration. Four rounds were run across 31 development scenes:
- **Round 1** — vector source comparison (OSM only, OSM+NHD, NHD only) → OSM+NHD selected
- **Round 2** — component analysis (no vector, manual NDWI, water vector only)
- **Round 3** — U-Net model ablation (model+vector, model+vector+NDWI, model+NDWI only)
- **Round 4** — vector data quality assessment (OSM/NHD rasterised vs ground truth)

**Overall finding:** all OWM configurations were outperformed by a manually-thresholded NDWI baseline and by the fine-tuned U-Net; OWM was not adopted as the primary method.

**NOTE: Do not re-run** — results saved to `results/owm_experiments/`. This notebook was run on a separate 31-scene development set, not the final 13 test scenes used in `owm_scenes_final.ipynb`.

---

## Results Directory Layout
All outputs are written under `BASE_DIR/results/`. The directory is created automatically on first run and is not included in this repository.

```
results/
├── unet_training/
│   ├── test_chips.json              # Held-out test chip list (must not be deleted or changed)
│   └── normalisation_stats.json    # Per-channel mean/std computed from training chips
│
├── unet_training_v3/
│   ├── kfold/
│   │   ├── best_model.pth           # Global best fold checkpoint (used by unet_diagnostic_final)
│   │   ├── fold{N}_best.pth         # Per-fold checkpoints (N = 1–5)
│   │   └── fold{N}_history.csv      # Per-fold training curves
│   ├── sweep_bs64_test/             # Stability experiment outputs
│   ├── sweep_lowwd_test/
│   ├── sweep_bce_test/
│   └── sweep_lr1e4_test/
│
├── unet_diagnostic_v3/
│   ├── prob_maps_best/              # Raw sigmoid probabilities — best single fold
│   ├── pred_maps_ensemble_majority/ # Binary predictions — majority vote ensemble
│   ├── fold{N}_thresh.json          # Per-fold optimal thresholds
│   └── final_summary.csv            # Method-level summary (best model, ensemble, NDWI, OWM) — used by comparison_all_final
│
├── ndwi_chips/
│   └── masks_t0.35/                 # NDWI binary masks at t=0.35
│
├── owm_chips/                       # OWM per-chip predictions (reconstructed scenes)
├── owm_experiments/                 # OWM configuration experiment outputs
│
├── dummy_masks/
│   └── water_{pct}pct/              # Synthetic baseline masks (pct ∈ {000,020,040,050,060,080,100})
│
├── dataset_analysis/
│   ├── scene_metadata.csv           # Per-scene date, location, water fraction, split
│   └── *.png                        # Temporal, spatial, seasonal, water fraction figures
│
└── dissertation_figures/
    └── fig_*.png                    # All publication figures (generated by generate_figures.ipynb)
```

---

## Notes
- Random seeds are fixed (base value 42, combined with a per-chip hash in `generate_water_masks.py`) for reproducibility.
- The test scene list is saved to `results/unet_training/test_chips.json` and must not be changed.
- Normalisation statistics are computed from the training set only and saved to `results/unet_training/normalisation_stats.json`.
- The global best fold checkpoint is saved to `results/unet_training_v3/kfold/best_model.pth`.
- Per-fold checkpoints are saved to `results/unet_training_v3/kfold/fold{N}_best.pth`.
- Per-fold thresholds for the ensemble are saved to `results/unet_diagnostic_v3/fold{N}_thresh.json`.
- U-Net best model probability maps are saved to `results/unet_diagnostic_v3/prob_maps_best/`.
- U-Net ensemble majority vote binary predictions are saved to `results/unet_diagnostic_v3/pred_maps_ensemble_majority/`.
- NDWI binary masks at t=0.35 are saved to `results/ndwi_chips/masks_t0.35/`.

---

## License
This project is released under the MIT License. See [`LICENSE`](LICENSE) for details.
