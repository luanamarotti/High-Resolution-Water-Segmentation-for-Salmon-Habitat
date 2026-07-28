# Code Review

## Summary

Overall: the repository is well documented and the split logic is mostly consistent, but several notebooks contain **hard-coded metrics/paths** and a few **comparison bugs** that can silently misstate results.

Top issues, in priority order:

1. [HIGH] `comparison_all_final.ipynb` mixes **per-scene mean chip F1** for OWM/NDWI with **pooled micro F1** for U-Net in the same heatmap, so the cross-method scene comparison is not apples-to-apples.
2. [HIGH] `unet_diagnostic_final.ipynb` overwrites computed best-model outputs with **hard-coded threshold/metric constants** and assumes the “best model” is fold 5, which can silently drift on reruns.
3. [HIGH] `comparison_all_final.ipynb` leaves U-Net IoU as `None`, then plots it as `0` in the bar chart, producing a misleading figure.
4. [HIGH] `owm_experiments.ipynb` Round 3 “`unet+no_vector+manual_ndwi`” actually enables OWM’s internal NDWI **and** unions a manual NDWI mask, so the experiment label does not match the code path.
5. [MEDIUM] Reproducibility is weakened by repeated Azure-specific absolute paths, partially seeded standalone training cells, committed `__pycache__`, large committed notebook outputs, and the previous absence of `requirements.txt`.

## Per-file findings

### `environment.yml`

- [MEDIUM] Lines 8-30: dependency versions are only lightly pinned (`python`, `pytorch`, `pytorch-cuda`), so exact rebuilds are not guaranteed across time.
- [LOW] Line 10: `torchvision` is declared, but no static import/use was found anywhere in the reviewed `.py` file or notebook source.
- [LOW] Lines 25-30: this is the only environment spec in the repo; before this change there was no pip-oriented `requirements.txt`, which made reproduction harder for non-conda users.

### `generate_water_masks.py`

- [LOW] Lines 38-42: the cross-file note was stale; the `WATER_RATIOS` list in `comparison_all_final.ipynb` is in notebook **cell 6**, not cell 5. Updated.
- [LOW] Lines 63-64: `WATER_RATIOS` is in sync with `comparison_all_final.ipynb` (`[0, 20, 40, 50, 60, 80, 100]`) — no drift found.
- [LOW] Lines 77-108: no high-confidence raster I/O bug found; the script preserves chip CRS/transform and writes single-band masks correctly.

### `README.md`

- [MEDIUM] Line 32: README says `BASE_DIR` is used “in all notebooks and scripts”, but several later notebook cells hard-code Azure paths instead of honoring `BASE_DIR` (see `unet_training_final.ipynb` cells 12-14 and `owm_scenes_final.ipynb` cell 10).
- [MEDIUM] Line 174: `final_summary.csv` is described as “Per-chip metrics for all test chips”, but `unet_diagnostic_final.ipynb` cell 14 writes a **per-method summary table**, not per-chip rows.
- [LOW] The README does not mention that several notebooks embed large saved outputs, which materially increases clone/review size.

### `.gitignore`

- [LOW] Line 2: `__pycache__/` is already listed, so the ignore rule itself is correct.
- [LOW] Despite that rule, `__pycache__/generate_water_masks.cpython-312.pyc` was still tracked in Git. I removed it from the index with `git rm -r --cached __pycache__/`.

### `comparison_all_final.ipynb`

- [HIGH] Cell 7: U-Net rows are appended with `IoU=None`, but Cell 8 uses `summary_df[metric].fillna(0)` before plotting. Result: the U-Net IoU bars render as **zero**, which is misleading rather than “missing”.
- [HIGH] Cell 9: `df_owm.groupby('scene')['f1'].mean()` uses mean per-chip F1 for OWM, while `per_scene_micro_prob()` / `per_scene_micro_binary()` compute pooled micro F1 for U-Net. Those quantities are not directly comparable in the same heatmap.
- [MEDIUM] Cell 9: NDWI per-scene data would also be mean per-chip F1 (`df_ndwi.groupby('scene')['f1'].mean()`) if enabled, so the same inconsistency applies there too.
- [LOW] Cell 3: NDWI mask generation is duplicated here instead of reusing the already-defined NDWI evaluation outputs; this raises maintenance cost but is not necessarily wrong.
- [LOW] Cell 13: the final print message says `Saved to: .../final_comparison.csv` immediately after saving `final_comparison.png`; the message is stale.

### `dataset_analysis.ipynb`

- [LOW] Cell 1: uses the Azure path as fallback; reproducible only when `BASE_DIR` is set correctly or the same Azure mount exists.
- [LOW] Cell 2: the whole-scene leakage assertion is a good safeguard; no static split-leakage bug was found here.
- [LOW] Cell 8: the notebook recommends pinning `requirements.txt`, but the repo did not include one before this review.

### `generate_figures.ipynb`

- [MEDIUM] Cell 4: the comparison figure uses hard-coded arrays (`0.5231`, `0.3440`, `0.8413`, `0.8629`, etc.) instead of loading `final_summary.csv`; figures can drift from the actual outputs if upstream metrics change.
- [MEDIUM] Cell 8: qualitative scene labels and `THRESH_BEST = 0.25` are hard-coded, so the figure is not data-driven and may silently become inconsistent with refreshed results.
- [LOW] Cell 6: map color scaling is normalized only to current min/max test F1, which makes cross-run visual comparisons harder.
- [LOW] Cell 1: same Azure fallback path issue as other notebooks.

### `ndwi_evaluation_final.ipynb`

- [LOW] Cell 1: uses Azure fallback path.
- [LOW] Static logic is internally consistent: honest baseline remains `t=0.35`, upper bound remains test-optimised `t=0.70`, and the headline 0.523/0.601 numbers are consistent within this notebook.

### `owm_experiments.ipynb`

- [HIGH] Cell 19: `unet+no_vector+manual_ndwi` calls `run_owm_debug(... use_ndwi=True ...)` and then unions the result with a second manual NDWI mask. The experiment label suggests one manual NDWI contribution, but the code path includes **internal NDWI plus manual NDWI**, changing the experiment definition.
- [MEDIUM] Cell 15: Round 2 summary always includes `summarise(df_osm_nhd)` as the reference row label, even though the notebook claims `ROUND2_VECTOR = winner` can be overridden. If the winner/override is not `osm_plus_nhd`, the summary label/data pairing becomes misleading.
- [MEDIUM] Cells 22-23: Round 4 downloads live OSM data from Overpass mirrors at review/runtime, so results are not snapshot-reproducible.
- [LOW] Cell 25: buffering is hard-coded in EPSG:32610; if scenes span outside that UTM zone, buffer distances become only approximate.
- [LOW] Cell 1: same Azure fallback path issue as other notebooks.

### `owm_scenes_final.ipynb`

- [MEDIUM] Cell 9 reports and plots **per-scene macro F1** (`owm_scene_results.csv` is grouped means), while downstream comparison code wants scene-level cross-method comparisons; this contributes to the mismatch in `comparison_all_final.ipynb`.
- [MEDIUM] Cell 10 hard-codes `VISUALS_DIR` to an absolute Azure path instead of deriving it from `BASE_DIR`, so the display step is not portable.
- [LOW] Cell 2 reconstructs full-scene transforms carefully; no high-confidence CRS/transform bug was confirmed statically.
- [LOW] Cell 8 deletes “extra” prediction files under `results/owm_chips/`; intended, but destructive if that directory is reused for anything besides the current test-chip set.

### `unet_diagnostic_final.ipynb`

- [HIGH] Cell 10 hard-codes:
  - `thresh_best_honest = 0.25`
  - `f1_best_honest = 0.8413`
  - `p_best_honest = 0.7995`
  - `r_best_honest = 0.8877`
  
  These overwrite the values computed earlier in Cell 4 and can silently desynchronize the notebook from actual model outputs.
- [HIGH] Cells 2 and 4: the notebook loads `best_model.pth` as the “best single fold” but performs threshold selection specifically with `fold5_best.pth`. That only remains correct if the global best checkpoint is always fold 5; the code does not verify that assumption.
- [MEDIUM] Cell 11 uses `0.601` as the NDWI baseline reference line in the per-scene bar chart, but elsewhere the notebook treats `0.523` as the honest NDWI baseline. This mixes the test-optimised upper bound with the deployment-honest baseline.
- [MEDIUM] Cell 14 falls back to hard-coded OWM metrics (`0.344`, `0.209`, `0.966`) when `owm_micro.csv` is missing; useful operationally, but it weakens provenance and can drift.
- [LOW] Cell 12 samples only the **first** 500 land pixels per chip for land probability histograms, which may bias the land distribution if spatial ordering matters.
- [LOW] Cell 1: Azure fallback path.

### `unet_stability_experiments_1.ipynb`

- [MEDIUM] Cell 3 seeds `random` and `numpy`, but not `torch`, despite training a PyTorch model in a self-contained experiment cell. That weakens reproducibility for model initialization and data-loader order.
- [LOW] The historical/superseded cell is clearly marked; no additional high-confidence correctness bug found.

### `unet_stability_experiments_2.ipynb`

- [MEDIUM] Cells 2-5 repeat the same partial seeding pattern (`random`, `numpy`, but not `torch`) in standalone training experiments, again weakening reproducibility.
- [LOW] Repeated near-duplicate training code across Cells 3-5 increases maintenance overhead and drift risk.
- [LOW] Azure fallback paths recur throughout.

### `unet_training_final.ipynb`

- [MEDIUM] Cells 12-14 are written as largely standalone training blocks, but they seed only `random` and `numpy`; unlike Cell 2, they do **not** call `torch.manual_seed()` / `torch.cuda.manual_seed_all()`. That means reruns of the final sweep/k-fold stages are not fully deterministic.
- [MEDIUM] Cells 12-14 hard-code `BASE_DIR` to the Azure mount instead of using the environment-variable-based setup from Cell 2, so the notebook is less portable than the README claims.
- [LOW] Cell 10/11 diagnostic F1 explicitly excludes land-only chips; this is documented, so not a bug, but readers can easily confuse it with headline metrics if they skip the note.
- [LOW] Cell 14 (the model-2 “K-Fold Training Curves” helper) points at an absolute path; acceptable as historical record, but not portable.

### `__pycache__/generate_water_masks.cpython-312.pyc`

- [LOW] A compiled artifact was committed even though `.gitignore` already excludes `__pycache__/`. This is pure repo-noise and should not be versioned.

## Cross-file / consistency findings

- [LOW] `WATER_RATIOS` sync check: **values are consistent** between `generate_water_masks.py` and `comparison_all_final.ipynb` (`[0, 20, 40, 50, 60, 80, 100]`).
- [LOW] The script comment pointed to notebook “cell 5”, but the actual `WATER_RATIOS` definition is in notebook **cell 6**. Updated in `generate_water_masks.py`.
- [LOW] The `TEST_SCENES` set and scene-ID parsing logic are repeated consistently across the U-Net notebooks reviewed; no static split drift was found in those lists.
- [MEDIUM] Metric-definition drift exists across notebooks:
  - U-Net final comparisons are mostly **pooled micro** metrics.
  - OWM per-scene outputs are **macro means over chips**.
  - `comparison_all_final.ipynb` mixes these in one heatmap.
- [MEDIUM] NDWI baseline usage is inconsistent:
  - `ndwi_evaluation_final.ipynb` clearly separates honest baseline `0.523` from upper bound `0.601`.
  - `unet_diagnostic_final.ipynb` uses `0.523` in some cells and `0.601` in the per-scene chart.
- [MEDIUM] Hard-coded result constants are repeated in multiple places (`unet_diagnostic_final.ipynb`, `generate_figures.ipynb`), creating multiple opportunities for silent drift.

## Reproducibility & environment findings

- [MEDIUM] Absolute Azure paths appear throughout the notebooks as fallbacks, and some cells hard-code them outright instead of using `BASE_DIR`.
- [MEDIUM] Import-vs-environment check: all non-stdlib imports found in the reviewed source are covered either directly or transitively by `environment.yml`:
  - direct: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `rasterio`, `geopandas`, `shapely`, `contextily`, `requests`, `tqdm`, `segmentation-models-pytorch`, `omniwatermask`, `opencv-python`, `torch`, `torchvision`
  - transitive/packaged via other entries: `PIL` via `pillow`, `IPython` via `jupyterlab`/`ipykernel`
  - no missing third-party import was confirmed statically.
- [LOW] `torchvision` appears unused in the reviewed static source.
- [MEDIUM] Before this review there was no `requirements.txt`; a best-effort one has now been added, but `environment.yml` remains the authoritative environment because CUDA/channel details are conda-specific.
- [MEDIUM] `owm_experiments.ipynb` Round 4 depends on live Overpass responses, so exact reruns are time-dependent.
- [MEDIUM] Several standalone PyTorch training cells do not set `torch` seeds, so even with identical data and paths they are not fully reproducible.

## Repo hygiene findings

- [LOW] `.gitignore` already includes `__pycache__/` (line 2), but a `.pyc` file was still tracked. Fixed by removing it from the Git index.
- [LOW] The repo contains very large committed notebooks with outputs, notably:
  - `owm_scenes_final.ipynb` (~12.5 MB)
  - `comparison_all_final.ipynb` (~8.6 MB)
  - `unet_diagnostic_final.ipynb` (~5.2 MB)
  
  This suggests embedded outputs/figures are contributing substantial repository bloat.
- [LOW] No high-confidence secrets, API keys, or tokens were found in the reviewed text/source cells.
- [LOW] The committed `__pycache__` entry and large notebook outputs both increase review friction without improving reproducibility.

## Would require re-running (owner decision needed)

- Verifying whether `best_model.pth` is always fold 5 in saved outputs, or whether the current threshold-selection logic already depends on an accidental fold alignment.
- Confirming whether the hard-coded metrics in `generate_figures.ipynb` still exactly match the current saved CSV outputs.
- Recomputing per-scene OWM/NDWI metrics as pooled micro F1 to replace the current mixed-definition heatmap.
- Checking whether Overpass Round 4 outputs remain stable over time or differ materially from the originally saved results.
- Confirming the “2026 anomaly” discussed in `dataset_analysis.ipynb` against original acquisition metadata.

## Recommended next steps

- [ ] Fix `comparison_all_final.ipynb` so every method uses the **same per-scene metric definition** (prefer pooled micro F1).
- [ ] Remove hard-coded best-model metrics/thresholds from `unet_diagnostic_final.ipynb`; derive them from computed outputs only.
- [ ] Explicitly verify which fold produced `best_model.pth`, then threshold that fold’s validation scenes rather than assuming fold 5.
- [ ] Correct the Round 3 OWM experiment definition for `unet+no_vector+manual_ndwi`.
- [ ] Refactor standalone training/evaluation cells to honor `BASE_DIR` and seed `torch` consistently.
- [ ] Consider stripping heavy notebook outputs from Git or keeping rendered artifacts outside the notebooks.
