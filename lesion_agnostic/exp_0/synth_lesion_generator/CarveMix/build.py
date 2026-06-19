#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a BraTS tumor mask library for synthetic lesion generation.

For each BraTS case, this script:
1. Loads the segmentation (`seg`) and matching T1 image
2. Creates the mask you want (default: whole tumor, i.e. seg > 0)
3. Computes:
   - binary mask
   - bounding box
   - voxel count
   - tumor intensity stats from the matching T1 image
4. Saves:
   - one compressed `.npz` file per tumor item
   - a global `metadata.csv`
   - a global `metadata.json`

Default first-use choice for T1:
    mask = seg > 0

Supported mask types:
- whole       : seg > 0
- edema       : seg == 2
- core        : seg in {1, 4}
- enhancing   : seg == 4
- necrotic    : seg == 1

Expected input structure:
You can point the script at:
- a BraTS root folder containing both images and segs somewhere underneath, OR
- separate image and seg roots

Example usage:
py -3.10 exp_0\synth_lesion_generator\build.py `
    --brats-img-dir data\preprocessed\BrainAgeNeXt\IXI `
    --brats-seg-dir data\library\BraTS_Masks `
    --output-dir data\library\BraTS_Masks_3_CM `
    --mask-type whole `

If your T1 and seg files are in the same root:
py -3.10 exp_0\synth_lesion_generator\build.py `
    --brats-img-dir data\raw\CM_BraTS_Masks `
    --brats-seg-dir data\raw\CM_BraTS_Masks `
    --output-dir data\library\Guizard_CarveMix\npz_files\ `
    --mask-type whole `
    --image-filter t1n

Outputs:
output-dir/
├── items/
│   ├── CASEID_000001.npz
│   ├── CASEID_000002.npz
│   └── ...
├── metadata.csv
└── metadata.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import label


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class TumorLibraryItem:
    case_id: str
    item_id: str
    img_path: str
    seg_path: str
    mask_type: str
    bbox_x0: int
    bbox_x1: int
    bbox_y0: int
    bbox_y1: int
    bbox_z0: int
    bbox_z1: int
    voxel_count: int
    tumor_mean: float
    tumor_std: float
    tumor_p10: float
    tumor_p90: float
    img_shape_x: int
    img_shape_y: int
    img_shape_z: int
    affine_00: float
    affine_01: float
    affine_02: float
    affine_03: float
    affine_10: float
    affine_11: float
    affine_12: float
    affine_13: float
    affine_20: float
    affine_21: float
    affine_22: float
    affine_23: float
    affine_30: float
    affine_31: float
    affine_32: float
    affine_33: float


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )


def strip_all_suffixes(name: str) -> str:
    for s in (".nii.gz", ".nii", ".gz", ".npy", ".npz", ".json", ".csv"):
        if name.endswith(s):
            return name[:-len(s)]
    return name


def load_nifti(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    return data, img


def find_nifti_files(root: Path, name_filter: Optional[str] = None) -> list[Path]:
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


def normalize_case_key(path: Path) -> str:
    """
    Turn a filename into a robust matching key.
    Removes common modality suffixes like _seg, _t1, _t1n, _t1ce, _flair, _t2.
    """
    stem = strip_all_suffixes(path.name)

    suffixes = [
        "_seg",
        "-seg",
        "_label",
        "-label",
        "_labels",
        "-labels",
        "_t1",
        "-t1",
        "_t1n",
        "-t1n",
        "_t1ce",
        "-t1ce",
        "_t1c",
        "-t1c",
        "_flair",
        "-flair",
        "_t2",
        "-t2",
    ]

    changed = True
    while changed:
        changed = False
        lower = stem.lower()
        for suf in suffixes:
            if lower.endswith(suf):
                stem = stem[: -len(suf)]
                changed = True
                break

    return stem


def get_case_id(path: Path) -> str:
    return normalize_case_key(path)


def get_mask_from_seg(seg: np.ndarray, mask_type: str) -> np.ndarray:
    if mask_type == "whole":
        return (seg > 0).astype(np.uint8)
    if mask_type == "edema":
        return (seg == 2).astype(np.uint8)
    if mask_type == "core":
        return np.isin(seg, [1, 4]).astype(np.uint8)
    if mask_type == "enhancing":
        return (seg == 4).astype(np.uint8)
    if mask_type == "necrotic":
        return (seg == 1).astype(np.uint8)
    raise ValueError(f"Unknown mask_type: {mask_type}")


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = label(mask)
    if n == 0:
        return mask.astype(np.uint8)

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = counts.argmax()
    return (labeled == largest).astype(np.uint8)


def compute_bbox(mask: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("Mask is empty; cannot compute bounding box.")
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return (
        (int(mins[0]), int(mins[1]), int(mins[2])),
        (int(maxs[0]), int(maxs[1]), int(maxs[2])),
    )


def crop_to_bbox(arr: np.ndarray, bbox_min: tuple[int, int, int], bbox_max: tuple[int, int, int]) -> np.ndarray:
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max
    return arr[x0:x1, y0:y1, z0:z1]


def robust_intensity_stats(img: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    vals = img[mask > 0]
    if vals.size == 0:
        raise ValueError("Mask contains no voxels in image.")
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals) + 1e-8),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
    }


def maybe_normalize_image(img: np.ndarray, mode: str) -> np.ndarray:
    """
    Optional intensity normalization before computing stats.

    mode:
    - none
    - zscore_nonzero
    - clip_1_99_unit
    """
    if mode == "none":
        return img.astype(np.float32)

    out = img.astype(np.float32).copy()
    nz = out[out != 0]

    if nz.size == 0:
        return out

    if mode == "zscore_nonzero":
        mean = float(np.mean(nz))
        std = float(np.std(nz))
        if std < 1e-8:
            std = 1.0
        out[out != 0] = (out[out != 0] - mean) / std
        return out

    if mode == "clip_1_99_unit":
        p1 = float(np.percentile(nz, 1))
        p99 = float(np.percentile(nz, 99))
        if p99 <= p1:
            out[:] = 0.0
            return out
        out = np.clip(out, p1, p99)
        out = (out - p1) / (p99 - p1)
        out[img == 0] = 0.0
        return out.astype(np.float32)

    raise ValueError(f"Unknown normalization mode: {mode}")


def save_metadata_csv(items: list[TumorLibraryItem], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not items:
        raise ValueError("No items to save to CSV.")

    fieldnames = list(asdict(items[0]).keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(asdict(item))


def save_metadata_json(items: list[TumorLibraryItem], out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([asdict(item) for item in items], f, indent=2)


# -----------------------------------------------------------------------------
# Matching
# -----------------------------------------------------------------------------

def build_seg_map(seg_files: list[Path]) -> dict[str, Path]:
    seg_map: dict[str, Path] = {}
    for seg_path in seg_files:
        key = normalize_case_key(seg_path)
        seg_map[key] = seg_path
    return seg_map


# -----------------------------------------------------------------------------
# Main library build logic
# -----------------------------------------------------------------------------

def build_library(
    brats_img_dir: Path,
    brats_seg_dir: Path,
    output_dir: Path,
    mask_type: str,
    image_filter: str,
    seg_filter: str,
    min_voxels: int,
    largest_cc: bool,
    normalize_mode: str,
    limit: Optional[int] = None,
) -> None:
    log = logging.getLogger()

    items_dir = output_dir / "items"
    items_dir.mkdir(parents=True, exist_ok=True)

    img_files = find_nifti_files(brats_img_dir, image_filter)
    seg_files = find_nifti_files(brats_seg_dir, seg_filter)

    if limit is not None:
        img_files = img_files[:limit]

    if len(img_files) == 0:
        raise FileNotFoundError(f"No matching image files found in {brats_img_dir} with filter={image_filter}")
    if len(seg_files) == 0:
        raise FileNotFoundError(f"No matching seg files found in {brats_seg_dir} with filter={seg_filter}")

    seg_map = build_seg_map(seg_files)

    log.info("Found %d image files", len(img_files))
    log.info("Found %d seg files", len(seg_files))

    metadata_items: list[TumorLibraryItem] = []
    skipped_no_seg = 0
    skipped_empty = 0
    skipped_small = 0

    for idx, img_path in enumerate(img_files, start=1):
        case_id = get_case_id(img_path)
        seg_path = seg_map.get(case_id)

        log.info("[%d/%d] Processing case_id=%s", idx, len(img_files), case_id)

        if seg_path is None:
            log.warning("No matching seg found for image: %s", img_path)
            skipped_no_seg += 1
            continue

        img, img_nii = load_nifti(img_path)
        seg, seg_nii = load_nifti(seg_path)

        if img.shape != seg.shape:
            log.warning(
                "Shape mismatch for %s: img shape=%s, seg shape=%s",
                case_id, img.shape, seg.shape
            )
            continue

        img = maybe_normalize_image(img, normalize_mode)

        mask = get_mask_from_seg(seg, mask_type)
        if largest_cc:
            mask = largest_connected_component(mask)

        voxel_count = int(mask.sum())
        if voxel_count == 0:
            log.warning("Empty mask after extraction for %s", case_id)
            skipped_empty += 1
            continue

        if voxel_count < min_voxels:
            log.warning("Mask too small for %s: %d voxels < %d", case_id, voxel_count, min_voxels)
            skipped_small += 1
            continue

        bbox_min, bbox_max = compute_bbox(mask)
        mask_crop = crop_to_bbox(mask, bbox_min, bbox_max).astype(np.uint8)
        patch_img = crop_to_bbox(img, bbox_min, bbox_max).astype(np.float32)

        stats = robust_intensity_stats(img, mask)

        item_id = f"{case_id}_{idx:06d}"
        item_path = items_dir / f"{item_id}.npz"

        np.savez_compressed(
            item_path,
            case_id=case_id,
            item_id=item_id,
            img_path=str(img_path),
            seg_path=str(seg_path),
            mask_type=mask_type,
            mask=mask_crop,
            patch_mask=mask_crop,        # (new, explicit)
            patch_img=patch_img,        
            bbox_min=np.array(bbox_min, dtype=np.int32),
            bbox_max=np.array(bbox_max, dtype=np.int32),
            voxel_count=np.int32(voxel_count),
            tumor_mean=np.float32(stats["mean"]),
            tumor_std=np.float32(stats["std"]),
            tumor_p10=np.float32(stats["p10"]),
            tumor_p90=np.float32(stats["p90"]),
            img_shape=np.array(img.shape, dtype=np.int32),
            affine=np.array(img_nii.affine, dtype=np.float32),
        )

        affine = np.asarray(img_nii.affine, dtype=np.float32)

        metadata_items.append(
            TumorLibraryItem(
                case_id=case_id,
                item_id=item_id,
                img_path=str(img_path),
                seg_path=str(seg_path),
                mask_type=mask_type,
                bbox_x0=bbox_min[0],
                bbox_x1=bbox_max[0],
                bbox_y0=bbox_min[1],
                bbox_y1=bbox_max[1],
                bbox_z0=bbox_min[2],
                bbox_z1=bbox_max[2],
                voxel_count=voxel_count,
                tumor_mean=stats["mean"],
                tumor_std=stats["std"],
                tumor_p10=stats["p10"],
                tumor_p90=stats["p90"],
                img_shape_x=int(img.shape[0]),
                img_shape_y=int(img.shape[1]),
                img_shape_z=int(img.shape[2]),
                affine_00=float(affine[0, 0]),
                affine_01=float(affine[0, 1]),
                affine_02=float(affine[0, 2]),
                affine_03=float(affine[0, 3]),
                affine_10=float(affine[1, 0]),
                affine_11=float(affine[1, 1]),
                affine_12=float(affine[1, 2]),
                affine_13=float(affine[1, 3]),
                affine_20=float(affine[2, 0]),
                affine_21=float(affine[2, 1]),
                affine_22=float(affine[2, 2]),
                affine_23=float(affine[2, 3]),
                affine_30=float(affine[3, 0]),
                affine_31=float(affine[3, 1]),
                affine_32=float(affine[3, 2]),
                affine_33=float(affine[3, 3]),
            )
        )

    if len(metadata_items) == 0:
        raise RuntimeError(
            "No valid library items were created. "
            "Check your image/seg filters, mask type, and input directories."
        )

    save_metadata_csv(metadata_items, output_dir / "metadata.csv")
    save_metadata_json(metadata_items, output_dir / "metadata.json")

    summary = {
        "num_items": len(metadata_items),
        "skipped_no_seg": skipped_no_seg,
        "skipped_empty": skipped_empty,
        "skipped_small": skipped_small,
        "mask_type": mask_type,
        "image_filter": image_filter,
        "seg_filter": seg_filter,
        "min_voxels": min_voxels,
        "largest_connected_component_only": largest_cc,
        "normalize_mode": normalize_mode,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("Library build complete.")
    log.info("Items created: %d", len(metadata_items))
    log.info("Skipped (no seg): %d", skipped_no_seg)
    log.info("Skipped (empty): %d", skipped_empty)
    log.info("Skipped (too small): %d", skipped_small)
    log.info("Saved metadata to: %s", output_dir)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BraTS tumor mask library for synthetic lesion generation.")
    parser.add_argument(
        "--brats-img-dir",
        type=Path,
        required=True,
        help="Directory containing BraTS T1 images (search is recursive)."
    )
    parser.add_argument(
        "--brats-seg-dir",
        type=Path,
        required=True,
        help="Directory containing BraTS seg files (search is recursive)."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the tumor library will be saved."
    )
    parser.add_argument(
        "--mask-type",
        type=str,
        default="whole",
        choices=["whole", "edema", "core", "enhancing", "necrotic"],
        help="Which mask to extract from BraTS seg."
    )
    parser.add_argument(
        "--image-filter",
        type=str,
        default="t1",
        help="Substring used to find matching image files, e.g. 't1' or 't1n'."
    )
    parser.add_argument(
        "--seg-filter",
        type=str,
        default="seg",
        help="Substring used to find segmentation files."
    )
    parser.add_argument(
        "--min-voxels",
        type=int,
        default=300,
        help="Minimum number of voxels required to keep a tumor item."
    )
    parser.add_argument(
        "--largest-cc",
        action="store_true",
        help="Keep only the largest connected component of the extracted tumor mask."
    )
    parser.add_argument(
        "--normalize-mode",
        type=str,
        default="none",
        choices=["none", "zscore_nonzero", "clip_1_99_unit"],
        help="Optional intensity normalization before computing tumor stats."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of image files processed."
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    if not args.brats_img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {args.brats_img_dir}")
    if not args.brats_seg_dir.exists():
        raise FileNotFoundError(f"Seg directory not found: {args.brats_seg_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    build_library(
        brats_img_dir=args.brats_img_dir,
        brats_seg_dir=args.brats_seg_dir,
        output_dir=args.output_dir,
        mask_type=args.mask_type,
        image_filter=args.image_filter,
        seg_filter=args.seg_filter,
        min_voxels=args.min_voxels,
        largest_cc=args.largest_cc,
        normalize_mode=args.normalize_mode,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()