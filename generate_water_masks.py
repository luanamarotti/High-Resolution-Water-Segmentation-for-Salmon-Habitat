"""
Generate dummy water/no-water mask images for test chips.

NOTE: Already run. 33,516 dummy mask files saved to results/dummy_masks/.
Do not re-run unless you need to regenerate the masks from scratch.

For each test chip, 7 mask variants are created at the following target water pixel ratios:
    Y = 0%, 20%, 40%, 50%, 60%, 80%, 100%
For fractional targets (20%–80%), the actual water fraction is approximately the
target ratio due to per-pixel Bernoulli sampling (0% and 100% are exact).
Used as sanity checks to verify the metrics pipeline is working correctly.

Expected behaviour:
    0%  target → F1 = 0.000, Recall = 0.000
    100% target → F1 = 1.000 only if ground truth is also all-water (rarely)
    Random masks should produce metrics near the water fraction baseline.

Output layout:
    Each input chip is written unchanged in name into a per-ratio subfolder:
        results/dummy_masks/water_<pct>pct/<original_chip_name>.tif
    (e.g. results/dummy_masks/water_040pct/<chip>.tif for the 40% ratio)

Usage:
    python generate_water_masks.py

Output folders:
    results/dummy_masks/water_000pct/
    results/dummy_masks/water_020pct/
    ...etc
"""
import os
import json
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

# ── Paths ──
# Override by setting the BASE_DIR environment variable
BASE_DIR        = Path(os.environ.get(
    'BASE_DIR',
    '/mnt/batch/tasks/shared/LS_root/mounts/clusters/v-lmarotti1/code/Users/v-lmarotti/OmniWaterMask'
))
CHIPS_DIR       = BASE_DIR / 'images/chips_images/images'
OUTPUT_DIR      = BASE_DIR / 'results/dummy_masks'
TEST_CHIPS_JSON = BASE_DIR / 'results/unet_training/test_chips.json'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Water ratios to generate
WATER_RATIOS = [0.0, 0.20, 0.40, 0.50, 0.60, 0.80, 1.0]
SEED_BASE    = 42

def generate_mask(height, width, water_ratio, seed=None):
    if water_ratio == 0.0:
        return np.zeros((height, width), dtype=np.uint8)
    if water_ratio == 1.0:
        return np.ones((height, width), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return (rng.random((height, width)) < water_ratio).astype(np.uint8)

def process_chip(chip_path, output_dir):
    with rasterio.open(chip_path) as src:
        height, width = src.height, src.width
        profile = src.profile.copy()
    profile.update(count=1, dtype='uint8', compress='deflate', nodata=None)
    stem = chip_path.stem
    for ratio in WATER_RATIOS:
        pct = int(round(ratio * 100))
        ratio_dir = output_dir / f'water_{pct:03d}pct'
        ratio_dir.mkdir(parents=True, exist_ok=True)
        out_path = ratio_dir / chip_path.name
        seed = SEED_BASE + hash((stem, pct)) % (2**31)
        mask = generate_mask(height, width, ratio, seed=seed)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(mask[np.newaxis, :, :])

def main():
    with open(TEST_CHIPS_JSON) as f:
        test_chip_names = json.load(f)
    test_chips = [CHIPS_DIR / name for name in test_chip_names]
    print(f'Test chips: {len(test_chips)}')
    print(f'Water ratios: {[int(r*100) for r in WATER_RATIOS]}%')
    print(f'Output dir: {OUTPUT_DIR}')
    print(f'Total output files: {len(test_chips) * len(WATER_RATIOS):,}')
    for chip_path in tqdm(test_chips, desc='Generating dummy masks'):
        process_chip(chip_path, OUTPUT_DIR)
    print('Done.')

if __name__ == '__main__':
    main()
