#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations


import argparse
import json
import logging
import random
from pathlib import Path
from typing import Optional


import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    gaussian_filter,
    label,
    rotate,
    zoom,
)




# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )




def strip_suffixes(name: str) -> str:
    for s in (".nii.gz", ".nii", ".gz", ".npy", ".npz", ".json"):
        if name.endswith(s):
            return name[:-len(s)]
    return name




def load_nifti(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    return data, img




def save_nifti(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ref_img.header.copy()
    header.set_data_dtype(np.float32)
    out = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, header)
    nib.save(out, str(out_path))




def save_json(obj: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)




def find_nifti_images(root: Path, name_filter: Optional[str] = None) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        lower = p.name.lower()
        if not (lower.endswith(".nii.gz") or lower.endswith(".nii")):
            continue
        if name_filter is not None and name_filter.lower() not in lower:
            continue
        files.append(p)
    return sorted(files)




def load_library_items(library_dir: Path) -> list[Path]:
    items_dir = library_dir / "items"
    if not items_dir.exists():
        raise FileNotFoundError(f"Library items directory not found: {items_dir}")


    item_files = sorted(items_dir.glob("*.npz"))
    if len(item_files) == 0:
        raise FileNotFoundError(f"No .npz tumor library items found in: {items_dir}")


    return item_files




def load_library_item(item_path: Path) -> dict:
    item = np.load(item_path, allow_pickle=True)
    return {k: item[k] for k in item.files}




# -----------------------------------------------------------------------------
# Brain mask / placement
# -----------------------------------------------------------------------------


def make_brain_mask(img: np.ndarray, nonzero_threshold: float = 1e-6) -> np.ndarray:
    return (img > nonzero_threshold).astype(np.uint8)




def valid_placement(candidate_mask: np.ndarray, brain_mask: np.ndarray, min_inside_ratio: float) -> bool:
    mask_voxels = int(candidate_mask.sum())
    if mask_voxels == 0:
        return False
    inside = int((candidate_mask * brain_mask).sum())
    return (inside / float(mask_voxels)) >= min_inside_ratio




def random_center_from_brain(brain_mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int]:
    coords = np.argwhere(brain_mask > 0)
    if coords.size == 0:
        raise ValueError("Brain mask is empty.")
    idx = int(rng.integers(0, len(coords)))
    c = coords[idx]
    return int(c[0]), int(c[1]), int(c[2])




def paste_array_at_center(
    arr_small: np.ndarray,
    out_shape: tuple[int, int, int],
    center: tuple[int, int, int],
    fill_value: float = 0.0,
) -> np.ndarray:
    out = np.full(out_shape, fill_value, dtype=arr_small.dtype)


    sx, sy, sz = arr_small.shape
    cx, cy, cz = center


    x0 = cx - sx // 2
    y0 = cy - sy // 2
    z0 = cz - sz // 2


    x1 = x0 + sx
    y1 = y0 + sy
    z1 = z0 + sz


    ox0 = max(0, x0)
    oy0 = max(0, y0)
    oz0 = max(0, z0)
    ox1 = min(out_shape[0], x1)
    oy1 = min(out_shape[1], y1)
    oz1 = min(out_shape[2], z1)


    if ox0 >= ox1 or oy0 >= oy1 or oz0 >= oz1:
        return out


    mx0 = ox0 - x0
    my0 = oy0 - y0
    mz0 = oz0 - z0
    mx1 = mx0 + (ox1 - ox0)
    my1 = my0 + (oy1 - oy0)
    mz1 = mz0 + (oz1 - oz0)


    out[ox0:ox1, oy0:oy1, oz0:oz1] = arr_small[mx0:mx1, my0:my1, mz0:mz1]
    return out




def sample_valid_center(
    transformed_mask: np.ndarray,
    brain_mask: np.ndarray,
    rng: np.random.Generator,
    max_tries: int,
    min_inside_ratio: float,
) -> Optional[tuple[int, int, int]]:
    for _ in range(max_tries):
        center = random_center_from_brain(brain_mask, rng)
        placed_mask = paste_array_at_center(transformed_mask, brain_mask.shape, center, fill_value=0)
        if valid_placement(placed_mask.astype(np.uint8), brain_mask, min_inside_ratio):
            return center
    return None




# -----------------------------------------------------------------------------
# Mask + patch transforms
# -----------------------------------------------------------------------------


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    lbl, n = label(mask)
    if n == 0:
        return mask.astype(np.uint8)
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    largest = counts.argmax()
    return (lbl == largest).astype(np.uint8)




def resize_binary_mask(mask: np.ndarray, scale_factor: float) -> np.ndarray:
    if abs(scale_factor - 1.0) < 1e-6:
        return mask.astype(np.uint8)
    out = zoom(mask.astype(np.float32), zoom=scale_factor, order=0)
    return (out > 0.5).astype(np.uint8)




def resize_image_patch(patch: np.ndarray, scale_factor: float) -> np.ndarray:
    if abs(scale_factor - 1.0) < 1e-6:
        return patch.astype(np.float32)
    return zoom(patch.astype(np.float32), zoom=scale_factor, order=1)




def rotate_binary_mask(mask: np.ndarray, angle_deg: float, axes: tuple[int, int]) -> np.ndarray:
    if abs(angle_deg) < 1e-6:
        return mask.astype(np.uint8)
    out = rotate(
        mask.astype(np.float32),
        angle=angle_deg,
        axes=axes,
        reshape=True,
        order=0,
        mode="constant",
        cval=0.0,
    )
    return (out > 0.5).astype(np.uint8)




def rotate_image_patch(patch: np.ndarray, angle_deg: float, axes: tuple[int, int]) -> np.ndarray:
    if abs(angle_deg) < 1e-6:
        return patch.astype(np.float32)
    return rotate(
        patch.astype(np.float32),
        angle=angle_deg,
        axes=axes,
        reshape=True,
        order=1,
        mode="constant",
        cval=0.0,
    ).astype(np.float32)




def transform_mask_and_patch(
    mask: np.ndarray,
    patch: np.ndarray,
    rng: np.random.Generator,
    scale_range: tuple[float, float],
    rotation_range_deg: tuple[float, float],
    allow_flip: bool,
) -> tuple[np.ndarray, np.ndarray]:
    mask_out = mask.astype(np.uint8)
    patch_out = patch.astype(np.float32)


    scale = float(rng.uniform(scale_range[0], scale_range[1]))
    mask_out = resize_binary_mask(mask_out, scale)
    patch_out = resize_image_patch(patch_out, scale)


    for axes in [(0, 1), (0, 2), (1, 2)]:
        angle = float(rng.uniform(rotation_range_deg[0], rotation_range_deg[1]))
        mask_out = rotate_binary_mask(mask_out, angle, axes)
        patch_out = rotate_image_patch(patch_out, angle, axes)


    if allow_flip:
        for axis in range(3):
            if rng.random() < 0.5:
                mask_out = np.flip(mask_out, axis=axis)
                patch_out = np.flip(patch_out, axis=axis)


    mask_out = largest_connected_component(mask_out)
    patch_out[mask_out == 0] = 0.0
    return mask_out.astype(np.uint8), patch_out.astype(np.float32)




# -----------------------------------------------------------------------------
# Patch extraction + normalization
# -----------------------------------------------------------------------------


def crop_from_bbox(arr: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    x0, y0, z0 = [int(v) for v in bbox_min]
    x1, y1, z1 = [int(v) for v in bbox_max]
    return arr[x0:x1, y0:y1, z0:z1]




def zscore_nonzero(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32).copy()
    nz = out[out != 0]
    if nz.size == 0:
        return out
    mean = float(np.mean(nz))
    std = float(np.std(nz))
    if std < 1e-8:
        std = 1.0
    out[out != 0] = (out[out != 0] - mean) / std
    return out




def match_patch_to_healthy_context(
    patch: np.ndarray,
    patch_mask: np.ndarray,
    healthy_img: np.ndarray,
    placed_mask: np.ndarray,
) -> np.ndarray:
    """
    Make the real BraTS patch live in the healthy intensity regime.
    """
    patch = patch.astype(np.float32).copy()


    src_vals = patch[patch_mask > 0]
    tgt_vals = healthy_img[placed_mask > 0]


    if src_vals.size == 0 or tgt_vals.size == 0:
        return patch


    src_mean = float(np.mean(src_vals))
    src_std = float(np.std(src_vals))
    tgt_mean = float(np.mean(tgt_vals))
    tgt_std = float(np.std(tgt_vals))


    if src_std < 1e-8:
        src_std = 1.0
    if tgt_std < 1e-8:
        tgt_std = 1.0


    vals = patch[patch_mask > 0]
    vals = (vals - src_mean) / src_std
    vals = vals * tgt_std + tgt_mean
    patch[patch_mask > 0] = vals
    return patch




# -----------------------------------------------------------------------------
# Blending
# -----------------------------------------------------------------------------


def make_boundary_region(mask: np.ndarray) -> np.ndarray:
    dil = binary_dilation(mask, iterations=1)
    ero = binary_erosion(mask, iterations=1)
    return np.logical_xor(dil, ero).astype(np.uint8)




def soft_blend_patch_into_healthy(
    healthy_img: np.ndarray,
    placed_mask: np.ndarray,
    placed_patch: np.ndarray,
    soft_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    soft_mask = gaussian_filter(placed_mask.astype(np.float32), sigma=soft_sigma)
    if soft_mask.max() > 0:
        soft_mask = soft_mask / soft_mask.max()


    synthetic = healthy_img.copy()
    support = soft_mask > 1e-6


    # Start from healthy everywhere, only blend on support
    synthetic[support] = (
        healthy_img[support] * (1.0 - soft_mask[support]) +
        placed_patch[support] * soft_mask[support]
    )


    synthetic[~support] = healthy_img[~support]
    return synthetic.astype(np.float32), soft_mask.astype(np.float32)




def smooth_boundary_only(img: np.ndarray, placed_mask: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img.astype(np.float32)
    boundary = make_boundary_region(placed_mask)
    smooth = gaussian_filter(img, sigma=sigma)
    out = img.copy()
    out[boundary > 0] = smooth[boundary > 0]
    return out.astype(np.float32)




# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------


def validate_synthetic_output(
    healthy_img: np.ndarray,
    synthetic_img: np.ndarray,
    placed_mask: np.ndarray,
) -> None:
    diff = np.abs(synthetic_img - healthy_img)
    if float(diff.sum()) <= 0:
        raise RuntimeError("Synthetic image is identical to healthy image.")


    if np.count_nonzero(synthetic_img) < 0.5 * np.count_nonzero(healthy_img):
        raise RuntimeError("Synthetic image lost most of the healthy brain.")


    if int(placed_mask.sum()) == 0:
        raise RuntimeError("Placed mask is empty.")


    logging.info(
        "Synthetic sanity check passed. diff_sum=%.4f diff_max=%.4f",
        float(diff.sum()),
        float(diff.max()),
    )




# -----------------------------------------------------------------------------
# Core generation
# -----------------------------------------------------------------------------


def generate_one_synthetic(
    healthy_path: Path,
    library_item_path: Path,
    out_healthy_dir: Path,
    out_synthetic_dir: Path,
    out_mask_dir: Path,
    out_diff_dir: Path,
    out_meta_dir: Path,
    rng: np.random.Generator,
    scale_range: tuple[float, float],
    rotation_range_deg: tuple[float, float],
    allow_flip: bool,
    max_placement_tries: int,
    min_inside_ratio: float,
    soft_sigma: float,
    boundary_sigma: float,
) -> bool:
    log = logging.getLogger()


    healthy_img, healthy_nii = load_nifti(healthy_path)
    brain_mask = make_brain_mask(healthy_img)


    if brain_mask.sum() == 0:
        log.warning("Empty brain mask for healthy image: %s", healthy_path)
        return False


    lib = load_library_item(library_item_path)


    tumor_mask_small = lib["mask"].astype(np.uint8)
    bbox_min = lib["bbox_min"]
    bbox_max = lib["bbox_max"]
    brats_img_path = Path(str(lib["img_path"]))


    brats_img, _ = load_nifti(brats_img_path)
    tumor_patch_small = crop_from_bbox(brats_img, bbox_min, bbox_max).astype(np.float32)


    tumor_mask_small, tumor_patch_small = transform_mask_and_patch(
        tumor_mask_small,
        tumor_patch_small,
        rng=rng,
        scale_range=scale_range,
        rotation_range_deg=rotation_range_deg,
        allow_flip=allow_flip,
    )


    center = sample_valid_center(
        transformed_mask=tumor_mask_small,
        brain_mask=brain_mask,
        rng=rng,
        max_tries=max_placement_tries,
        min_inside_ratio=min_inside_ratio,
    )


    if center is None:
        log.warning("Could not find valid placement for: %s", healthy_path.name)
        return False


    placed_mask = paste_array_at_center(
        tumor_mask_small,
        brain_mask.shape,
        center,
        fill_value=0,
    ).astype(np.uint8)


    placed_patch = paste_array_at_center(
        tumor_patch_small,
        brain_mask.shape,
        center,
        fill_value=0.0,
    ).astype(np.float32)


    placed_patch = match_patch_to_healthy_context(
        patch=placed_patch,
        patch_mask=placed_mask,
        healthy_img=healthy_img,
        placed_mask=placed_mask,
    )


    synthetic_img, soft_mask = soft_blend_patch_into_healthy(
        healthy_img=healthy_img,
        placed_mask=placed_mask,
        placed_patch=placed_patch,
        soft_sigma=soft_sigma,
    )


    synthetic_img = smooth_boundary_only(
        img=synthetic_img,
        placed_mask=placed_mask,
        sigma=boundary_sigma,
    )


    validate_synthetic_output(healthy_img, synthetic_img, placed_mask)


    diff_img = np.abs(synthetic_img - healthy_img).astype(np.float32)


    stem = strip_suffixes(healthy_path.name)


    out_healthy_path = out_healthy_dir / f"{stem}.nii.gz"
    out_synthetic_path = out_synthetic_dir / f"{stem}_synthetic.nii.gz"
    out_mask_path = out_mask_dir / f"{stem}_synthetic_mask.nii.gz"
    out_diff_path = out_diff_dir / f"{stem}_synthetic_diff.nii.gz"
    out_meta_path = out_meta_dir / f"{stem}_metadata.json"


    save_nifti(healthy_img, healthy_nii, out_healthy_path)
    save_nifti(synthetic_img, healthy_nii, out_synthetic_path)
    save_nifti(placed_mask.astype(np.float32), healthy_nii, out_mask_path)
    save_nifti(diff_img, healthy_nii, out_diff_path)


    meta = {
        "healthy_path": str(healthy_path),
        "library_item_path": str(library_item_path),
        "library_case_id": str(lib["case_id"]),
        "library_item_id": str(lib["item_id"]),
        "library_img_path": str(lib["img_path"]),
        "library_seg_path": str(lib["seg_path"]),
        "mask_type": str(lib["mask_type"]),
        "tumor_stats": {
            "mean": float(lib["tumor_mean"]),
            "std": float(lib["tumor_std"]),
            "p10": float(lib["tumor_p10"]),
            "p90": float(lib["tumor_p90"]),
            "voxel_count_library": int(lib["voxel_count"]),
        },
        "generation": {
            "scale_range": [float(scale_range[0]), float(scale_range[1])],
            "rotation_range_deg": [float(rotation_range_deg[0]), float(rotation_range_deg[1])],
            "allow_flip": bool(allow_flip),
            "max_placement_tries": int(max_placement_tries),
            "min_inside_ratio": float(min_inside_ratio),
            "soft_sigma": float(soft_sigma),
            "boundary_sigma": float(boundary_sigma),
        },
        "placement": {
            "placed_mask_voxel_count": int(placed_mask.sum()),
            "brain_overlap_voxel_count": int((placed_mask * brain_mask).sum()),
        },
        "sanity": {
            "healthy_nonzero": int(np.count_nonzero(healthy_img)),
            "synthetic_nonzero": int(np.count_nonzero(synthetic_img)),
            "diff_sum": float(np.abs(synthetic_img - healthy_img).sum()),
            "diff_max": float(np.abs(synthetic_img - healthy_img).max()),
        },
    }
    save_json(meta, out_meta_path)
    return True




# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic tumor-bearing MRIs from healthy scans.")
    parser.add_argument("--healthy-dir", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name-filter", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-min", type=float, default=0.85)
    parser.add_argument("--scale-max", type=float, default=1.15)
    parser.add_argument("--rot-min", type=float, default=-10.0)
    parser.add_argument("--rot-max", type=float, default=10.0)
    parser.add_argument("--allow-flip", action="store_true")
    parser.add_argument("--max-placement-tries", type=int, default=100)
    parser.add_argument("--min-inside-ratio", type=float, default=0.98)
    parser.add_argument("--soft-sigma", type=float, default=2.0)
    parser.add_argument("--boundary-sigma", type=float, default=0.6)
    return parser.parse_args()




def main() -> None:
    setup_logging()
    args = parse_args()


    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)


    healthy_files = find_nifti_images(args.healthy_dir, args.name_filter)
    if len(healthy_files) == 0:
        raise FileNotFoundError(f"No healthy NIfTI files found in {args.healthy_dir}")


    if args.num_samples is not None:
        healthy_files = healthy_files[:args.num_samples]


    library_items = load_library_items(args.library_dir)


    out_healthy_dir = args.output_dir / "healthy"
    out_synthetic_dir = args.output_dir / "synthetic"
    out_mask_dir = args.output_dir / "masks"
    out_diff_dir = args.output_dir / "diff"
    out_meta_dir = args.output_dir / "metadata"


    out_healthy_dir.mkdir(parents=True, exist_ok=True)
    out_synthetic_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_diff_dir.mkdir(parents=True, exist_ok=True)
    out_meta_dir.mkdir(parents=True, exist_ok=True)


    success = 0
    failed = 0


    logging.info("Healthy scans found: %d", len(healthy_files))
    logging.info("Library items found: %d", len(library_items))


    for i, healthy_path in enumerate(healthy_files, start=1):
        item_path = random.choice(library_items)
        logging.info("[%d/%d] Generating synthetic version for %s using %s",
                     i, len(healthy_files), healthy_path.name, item_path.name)


        ok = generate_one_synthetic(
            healthy_path=healthy_path,
            library_item_path=item_path,
            out_healthy_dir=out_healthy_dir,
            out_synthetic_dir=out_synthetic_dir,
            out_mask_dir=out_mask_dir,
            out_diff_dir=out_diff_dir,
            out_meta_dir=out_meta_dir,
            rng=rng,
            scale_range=(args.scale_min, args.scale_max),
            rotation_range_deg=(args.rot_min, args.rot_max),
            allow_flip=args.allow_flip,
            max_placement_tries=args.max_placement_tries,
            min_inside_ratio=args.min_inside_ratio,
            soft_sigma=args.soft_sigma,
            boundary_sigma=args.boundary_sigma,
        )


        if ok:
            success += 1
        else:
            failed += 1


    summary = {
        "healthy_dir": str(args.healthy_dir),
        "library_dir": str(args.library_dir),
        "output_dir": str(args.output_dir),
        "num_requested": len(healthy_files),
        "num_success": success,
        "num_failed": failed,
        "seed": args.seed,
    }
    save_json(summary, args.output_dir / "summary.json")


    logging.info("Done. Success=%d, Failed=%d", success, failed)




if __name__ == "__main__":
    main()

