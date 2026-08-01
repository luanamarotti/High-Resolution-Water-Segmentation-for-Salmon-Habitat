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
    python generate_water_masks.py --base-dir /path/to/project
    BASE_DIR=/path/to/project python generate_water_masks.py

Output folders:
    results/dummy_masks/water_000pct/
    results/dummy_masks/water_020pct/
    ...etc

Reproducibility:
    Masks for fractional ratios use a deterministic, per-chip seed derived from
    SEED_BASE and the chip filename via MD5 (NOT Python's built-in hash(), which
    is randomised per-process through PYTHONHASHSEED). 0% and 100% masks are
    always exact regardless of seed.

Cross-file dependency:
    WATER_RATIOS below must stay in sync with the WATER_RATIOS list in
    comparison_all_final.ipynb (currently [0, 20, 40, 50, 60, 80, 100]).
    Adding a ratio here without updating the notebook will silently exclude the
    new variant from the pipeline comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

# Water ratios to generate.
# IMPORTANT: Keep in sync with WATER_RATIOS in comparison_all_final.ipynb.
WATER_RATIOS = [0.0, 0.20, 0.40, 0.50, 0.60, 0.80, 1.0]
SEED_BASE    = 42
_MAX_SEED    = 2**31


def generate_mask(height: int, width: int, water_ratio: float, seed: Optional[int] = None) -> np.ndarray:
    if water_ratio == 0.0:
        return np.zeros((height, width), dtype=np.uint8)
    if water_ratio == 1.0:
        return np.ones((height, width), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    return (rng.random((height, width)) < water_ratio).astype(np.uint8)


def process_chip(chip_path: Path, output_dir: Path) -> None:
    with rasterio.open(chip_path) as src:
        # Build a clean minimal profile rather than copying all source metadata
        # (e.g. photometric tags, BIGTIFF, interleaving) from the multi-band chip.
        profile = {
            'driver':    'GTiff',
            'dtype':     'uint8',
            'width':     src.width,
            'height':    src.height,
            'count':     1,
            'crs':       src.crs,
            'transform': src.transform,
            'compress':  'deflate',
            'nodata':    None,
        }
    height = profile['height']
    width  = profile['width']
    # chip_stem is used only for deterministic seed generation, not for output naming.
    chip_stem = chip_path.stem
    for ratio in WATER_RATIOS:
        assert 0.0 <= ratio <= 1.0, f'ratio {ratio} is outside [0, 1]'
        pct       = int(round(ratio * 100))
        ratio_dir = output_dir / f'water_{pct:03d}pct'
        ratio_dir.mkdir(parents=True, exist_ok=True)
        out_path  = ratio_dir / chip_path.name
        # Use MD5 for a deterministic hash that is stable across Python processes
        # (unlike the built-in hash(), which is randomised via PYTHONHASHSEED).
        hash_int  = int(hashlib.md5(f'{chip_stem}{pct}'.encode()).hexdigest(), 16)
        seed      = SEED_BASE + hash_int % _MAX_SEED
        mask      = generate_mask(height, width, ratio, seed=seed)
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(mask[np.newaxis, :, :])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate dummy water/no-water mask images for test chips.'
    )
    parser.add_argument(
        '--base-dir', type=Path,
        default=Path(os.environ['BASE_DIR']) if 'BASE_DIR' in os.environ else None,
        help='Project root directory (default: $BASE_DIR env var)',
    )
    parser.add_argument('--chips-dir',       type=Path, help='Override chips image directory')
    parser.add_argument('--output-dir',      type=Path, help='Override output directory')
    parser.add_argument('--test-chips-json', type=Path, help='Override test chips JSON path')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.base_dir is None and (
        args.chips_dir is None or args.output_dir is None or args.test_chips_json is None
    ):
        raise ValueError(
            'Provide --base-dir (or set $BASE_DIR) so that all paths can be resolved, '
            'or supply --chips-dir, --output-dir, and --test-chips-json individually.'
        )

    base_dir        = args.base_dir
    chips_dir       = args.chips_dir       or base_dir / 'images/chips_images/images'
    output_dir      = args.output_dir      or base_dir / 'results/dummy_masks'
    test_chips_json = args.test_chips_json or base_dir / 'results/unet_training/test_chips.json'

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(test_chips_json) as f:
        test_chip_names = json.load(f)

    test_chips = [chips_dir / name for name in test_chip_names]

    missing = [p for p in test_chips if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f'{len(missing)} chip(s) not found under {chips_dir}. '
            f'First missing: {missing[0].name}'
        )

    log.info(f'Test chips:         {len(test_chips)}')
    log.info(f'Water ratios:       {[int(r * 100) for r in WATER_RATIOS]}%')
    log.info(f'Output dir:         {output_dir}')
    log.info(f'Total output files: {len(test_chips) * len(WATER_RATIOS):,}')

    failed = []
    for chip_path in tqdm(test_chips, desc='Generating dummy masks'):
        try:
            process_chip(chip_path, output_dir)
        except Exception as e:
            failed.append((chip_path.name, str(e)))

    if failed:
        log.warning(f'WARNING: {len(failed)} chip(s) failed:')
        for name, err in failed[:10]:
            log.warning(f'  {name}: {err}')

    log.info('Done.')


if __name__ == '__main__':
    main()
