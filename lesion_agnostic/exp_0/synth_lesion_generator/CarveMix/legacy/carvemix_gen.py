#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python3.10 carvemix_gen.py \
  --healthy-dir data/preprocessed/healthy \
  --library-dir library/output_cm \
  --output-dir generated_t1_cm \
  --name-filter t1 \
  --seed 42 \
  --soft-sigma 4.5 \
  --boundary-sigma 1.2 \
  --patch-sigma 0.0 \
  --local-mix 0.15
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    label,
    rotate,
    zoom,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def strip_suffixes(name: str) -> str:
    for s in (".nii.gz", ".nii"):
        if name.endswith(s):
            return name[:-len(s)]
    return name


def load_nifti(path: Path):
    img = nib.load(str(path))
    return img.get_fdata().astype(np.float32), img


def save_nifti(data, ref_img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ref_img.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data.astype(np.float32), ref_img.affine, header), str(path))


def find_nifti(root: Path, filt: Optional[str]):
    out = []
    for p in root.rglob("*"):
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            if filt is None or filt.lower() in p.name.lower():
                out.append(p)
    return sorted(out)


def load_lib_items(lib_dir: Path):
    items = sorted((lib_dir / "items").glob("*.npz"))
    if not items:
        raise RuntimeError("No library items found")
    return items


def load_lib(item_path: Path):
    f = np.load(item_path, allow_pickle=True)
    return {k: f[k] for k in f.files}


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------

def brain_mask(img):
    return (img > 1e-6).astype(np.uint8)


def paste(arr, shape, center, fill_value=0):
    out = np.full(shape, fill_value, dtype=arr.dtype)

    sx, sy, sz = arr.shape
    cx, cy, cz = center

    x0 = cx - sx // 2
    y0 = cy - sy // 2
    z0 = cz - sz // 2

    x1, y1, z1 = x0 + sx, y0 + sy, z0 + sz

    ox0, oy0, oz0 = max(0, x0), max(0, y0), max(0, z0)
    ox1, oy1, oz1 = min(shape[0], x1), min(shape[1], y1), min(shape[2], z1)

    if ox0 >= ox1 or oy0 >= oy1 or oz0 >= oz1:
        return out

    mx0, my0, mz0 = ox0 - x0, oy0 - y0, oz0 - z0
    mx1, my1, mz1 = mx0 + (ox1 - ox0), my0 + (oy1 - oy0), mz0 + (oz1 - oz0)

    out[ox0:ox1, oy0:oy1, oz0:oz1] = arr[mx0:mx1, my0:my1, mz0:mz1]
    return out


def largest_cc(mask):
    lbl, n = label(mask)
    if n == 0:
        return mask.astype(np.uint8)
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    return (lbl == counts.argmax()).astype(np.uint8)


def transform(mask, patch, rng, scale_range, rot_range, allow_flip=False):
    scale = float(rng.uniform(*scale_range))

    mask = zoom(mask.astype(np.float32), scale, order=0) > 0.5
    patch = zoom(patch.astype(np.float32), scale, order=1)

    for axes in [(0, 1), (0, 2), (1, 2)]:
        ang = float(rng.uniform(*rot_range))
        mask = rotate(mask.astype(np.float32), ang, axes=axes, order=0, reshape=True, mode="constant", cval=0) > 0.5
        patch = rotate(patch.astype(np.float32), ang, axes=axes, order=1, reshape=True, mode="constant", cval=0)

    if allow_flip:
        for axis in range(3):
            if rng.random() < 0.5:
                mask = np.flip(mask, axis=axis)
                patch = np.flip(patch, axis=axis)

    mask = largest_cc(mask)
    patch[mask == 0] = 0.0

    return mask.astype(np.uint8), patch.astype(np.float32)


def find_valid_center(mask_small, brain, rng, tries=100, min_inside_ratio=0.98):
    coords = np.argwhere(brain > 0)
    if len(coords) == 0:
        return None

    for _ in range(tries):
        c = coords[rng.integers(len(coords))]
        placed = paste(mask_small, brain.shape, c, fill_value=0)
        inside_ratio = (placed * brain).sum() / (placed.sum() + 1e-8)
        if inside_ratio >= min_inside_ratio:
            return tuple(int(x) for x in c)
    return None


def make_ring(mask, dilate_iters=4):
    dil = binary_dilation(mask.astype(bool), iterations=dilate_iters)
    ring = np.logical_and(dil, ~mask.astype(bool))
    return ring.astype(np.uint8)


# -----------------------------------------------------------------------------
# Improved pathology intensity modelling
# -----------------------------------------------------------------------------

def masked_gaussian_smooth(arr: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """
    Smooth only inside the mask without bleeding zeros inward.
    """
    arr = arr.astype(np.float32)
    mask = mask.astype(np.float32)

    num = gaussian_filter(arr * mask, sigma=sigma)
    den = gaussian_filter(mask, sigma=sigma)
    out = np.zeros_like(arr, dtype=np.float32)

    valid = den > 1e-6
    out[valid] = num[valid] / den[valid]
    return out


def adapt_patch_for_t1_realism(
    patch: np.ndarray,
    patch_mask: np.ndarray,
    healthy_img: np.ndarray,
    placed_mask: np.ndarray,
    brain: np.ndarray,
    rng: np.random.Generator,
    local_mix: float = 0.3,
    dark_strength: float = 1.1,
    texture_strength: float = 0.15,
    smooth_sigma: float = 2.5,
    center_darkening_strength: float = 0.65,
) -> np.ndarray:
    """
    Build a realistic non-contrast T1 tumor appearance.

    Main idea:
    - use source patch only as weak texture prior
    - strongly smooth it to remove seams / partitions
    - anchor lesion to local healthy context
    - force lesion darker than surrounding tissue
    """

    out = patch.astype(np.float32).copy()
    mask = patch_mask.astype(bool)

    src_vals = out[mask]
    if src_vals.size == 0:
        return out

    ring = make_ring(placed_mask, dilate_iters=4)
    tgt_vals = healthy_img[ring > 0]

    if tgt_vals.size == 0:
        tgt_vals = healthy_img[placed_mask > 0]
    if tgt_vals.size == 0:
        return out

    ctx_mean = float(np.mean(tgt_vals))
    ctx_std = float(np.std(tgt_vals))
    if ctx_std < 1e-6:
        ctx_std = 1.0

    # ------------------------------------------------------------------
    # 1. Smooth the patch inside mask to kill internal seams/fragmentation
    # ------------------------------------------------------------------
    smooth_patch = masked_gaussian_smooth(out, patch_mask, sigma=smooth_sigma)

    vals = smooth_patch[mask]

    # ------------------------------------------------------------------
    # 2. Make source patch only a weak texture prior
    # ------------------------------------------------------------------
    vals_mean = float(np.mean(vals))
    vals_std = float(np.std(vals))
    if vals_std < 1e-6:
        vals_std = 1.0
    texture = (vals - vals_mean) / vals_std

    # ------------------------------------------------------------------
    # 3. Force lesion darker than local tissue in T1
    # ------------------------------------------------------------------
    target_mean = ctx_mean - dark_strength * ctx_std
    target_std = texture_strength * ctx_std

    vals = target_mean + texture * target_std

    # ------------------------------------------------------------------
    # 4. Mix slightly with local healthy intensities
    # ------------------------------------------------------------------
    local_vals = healthy_img[placed_mask > 0]
    if local_vals.size == vals.size:
        vals = (1.0 - local_mix) * vals + local_mix * local_vals

    out[mask] = vals
    out[~mask] = 0.0

    # ------------------------------------------------------------------
    # 5. Add mild heterogeneity, but much weaker than before
    # ------------------------------------------------------------------
    noise = rng.normal(0.0, 0.03 * ctx_std, size=vals.shape).astype(np.float32)
    out[mask] = out[mask] + noise

    # ------------------------------------------------------------------
    # 6. Make center slightly darker than periphery
    # ------------------------------------------------------------------
    dist = distance_transform_edt(mask).astype(np.float32)
    if dist.max() > 0:
        dist = dist / dist.max()
        out[mask] = out[mask] - center_darkening_strength * ctx_std * dist[mask]

    # ------------------------------------------------------------------
    # 7. Clamp to realistic brain range
    # ------------------------------------------------------------------
    brain_vals = healthy_img[brain > 0]
    if brain_vals.size > 0:
        p2, p98 = np.percentile(brain_vals, [2, 98])
        out[mask] = np.clip(out[mask], p2, p98)

    out[~mask] = 0.0
    return out.astype(np.float32)


# -----------------------------------------------------------------------------
# Blending
# -----------------------------------------------------------------------------

def make_soft_mask(mask, sigma):
    soft = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    if soft.max() > 0:
        soft = soft / soft.max()
    return soft.astype(np.float32)


def blend(healthy, mask, patch, sigma):
    soft = make_soft_mask(mask, sigma=sigma)
    out = healthy.copy()

    support = soft > 1e-6
    out[support] = healthy[support] * (1.0 - soft[support]) + patch[support] * soft[support]
    out[~support] = healthy[~support]
    return out.astype(np.float32), soft


def smooth_boundary_only(img, mask, sigma=1.0):
    if sigma <= 0:
        return img.astype(np.float32)

    dil = binary_dilation(mask.astype(bool), iterations=1)
    ero = binary_erosion(mask.astype(bool), iterations=1)
    boundary = np.logical_xor(dil, ero)

    smoothed = gaussian_filter(img.astype(np.float32), sigma=sigma)
    out = img.copy()
    out[boundary] = smoothed[boundary]
    return out.astype(np.float32)


def validate_synthetic_output(healthy, synthetic, mask):
    diff = np.abs(synthetic - healthy)
    if float(diff.sum()) <= 0:
        raise RuntimeError("Synthetic image is identical to healthy image.")
    if int(mask.sum()) == 0:
        raise RuntimeError("Placed mask is empty.")
    if np.count_nonzero(synthetic) < 0.5 * np.count_nonzero(healthy):
        raise RuntimeError("Synthetic image lost most of the healthy brain.")


# -----------------------------------------------------------------------------
# MAIN GENERATOR
# -----------------------------------------------------------------------------

def generate_one(h_path, lib_path, out_dirs, rng, args):
    healthy, nii = load_nifti(h_path)
    brain = brain_mask(healthy)

    lib = load_lib(lib_path)

    patch = lib["patch_img"].astype(np.float32)
    mask = lib["patch_mask"].astype(np.uint8)

    patch[mask == 0] = 0.0

    mask, patch = transform(
        mask,
        patch,
        rng,
        (args.scale_min, args.scale_max),
        (args.rot_min, args.rot_max),
        allow_flip=args.allow_flip,
    )

    center = find_valid_center(mask, brain, rng, tries=args.max_placement_tries, min_inside_ratio=args.min_inside_ratio)
    if center is None:
        logging.warning("Could not find valid center for %s", h_path.name)
        return False

    placed_mask = paste(mask, healthy.shape, center, fill_value=0).astype(np.uint8)
    placed_patch = paste(patch, healthy.shape, center, fill_value=0.0).astype(np.float32)

    placed_patch[placed_mask == 0] = 0.0

    placed_patch = adapt_patch_for_t1_realism(
        patch=placed_patch,
        patch_mask=placed_mask,
        healthy_img=healthy,
        placed_mask=placed_mask,
        brain=brain,
        rng=rng,
        local_mix=args.local_mix,
    )

    if args.patch_sigma > 0:
        placed_patch = masked_gaussian_smooth(placed_patch, placed_mask, sigma=args.patch_sigma)
        placed_patch[placed_mask == 0] = 0.0

    synthetic, soft = blend(healthy, placed_mask, placed_patch, args.soft_sigma)
    synthetic = smooth_boundary_only(synthetic, placed_mask, sigma=args.boundary_sigma)

    validate_synthetic_output(healthy, synthetic, placed_mask)

    diff = np.abs(synthetic - healthy).astype(np.float32)

    stem = strip_suffixes(h_path.name)
    save_nifti(healthy, nii, out_dirs["healthy"] / f"{stem}.nii.gz")
    save_nifti(synthetic, nii, out_dirs["synthetic"] / f"{stem}_synthetic.nii.gz")
    save_nifti(placed_mask.astype(np.float32), nii, out_dirs["mask"] / f"{stem}_mask.nii.gz")
    save_nifti(diff, nii, out_dirs["diff"] / f"{stem}_diff.nii.gz")

    return True


# -----------------------------------------------------------------------------

def main():
    setup_logging()
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    healthy = find_nifti(args.healthy_dir, args.name_filter)
    lib = load_lib_items(args.library_dir)

    out_dirs = {
        "healthy": args.output_dir / "healthy",
        "synthetic": args.output_dir / "synthetic",
        "mask": args.output_dir / "masks",
        "diff": args.output_dir / "diff",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0

    for i, h in enumerate(healthy):
        item = random.choice(lib)
        logging.info("[%d/%d] %s + %s", i + 1, len(healthy), h.name, item.name)
        ok = generate_one(h, item, out_dirs, rng, args)
        if ok:
            success += 1
        else:
            failed += 1

    logging.info("DONE. success=%d failed=%d", success, failed)


# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--healthy-dir", type=Path, required=True)
    p.add_argument("--library-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--name-filter", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--scale-min", type=float, default=0.90)
    p.add_argument("--scale-max", type=float, default=1.10)
    p.add_argument("--rot-min", type=float, default=-10.0)
    p.add_argument("--rot-max", type=float, default=10.0)
    p.add_argument("--allow-flip", action="store_true")

    p.add_argument("--max-placement-tries", type=int, default=100)
    p.add_argument("--min-inside-ratio", type=float, default=0.98)

    p.add_argument("--soft-sigma", type=float, default=4.5)
    p.add_argument("--boundary-sigma", type=float, default=1.2)
    p.add_argument("--patch-sigma", type=float, default=0.0)
    p.add_argument("--local-mix", type=float, default=0.15)

    return p.parse_args()


if __name__ == "__main__":
    main()