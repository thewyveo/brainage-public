#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CarveMix generation with GliGAN pairings as preferred IXI<->BraTS pairs.

IMPORTANT:
This version preserves original CarveMix-style lesion location.

That means:
- no random tumor center is sampled;
- if the library item is full-volume, it is used directly;
- if the library item is cropped, it is pasted back into the original
  aligned coordinates using bbox_min/bbox_max.

Behavior:
1. Read GliGAN successful pairings CSV.
2. For each GliGAN pair:
   - find matching IXI MRI in --healthy-dir
   - find matching CarveMix library item in --library-dir/items
   - try that exact pair first
   - if that preferred tumor item fails or cannot be found, retry same IXI with another unused tumor item
3. No healthy MRI is successfully used more than once.
4. No tumor library item is attempted more than once.
5. If successful count is below --target-total, generate random additional pairs from unused IXIs and unused tumors.
6. Save all logs, all attempts, successful pairings, and failed attempts.

Example:
py -3.10 carvemix_from_gligan_pairings_original_location.py `
  --healthy-dir "data\\preprocessed\\CarveMix\\IXI" `
  --library-dir "data\\library\\BraTS_Masks" `
  --gligan-pairings-csv "successful_ixi_brats_pairings.csv" `
  --output-dir "data\\generated_carvemix_matched_original_location" `
  --target-total 550 `
  --extra-random-if-under 50 `
  --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import label as cc_label


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "carvemix_pairing_generation.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return log_path


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def strip_suffixes(name: str) -> str:
    for s in (".nii.gz", ".nii", ".gz", ".npz", ".npy", ".json"):
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

    items = sorted(items_dir.glob("*.npz"))

    if len(items) == 0:
        raise FileNotFoundError(f"No .npz library items found under: {items_dir}")

    return items


def load_npz(path: Path) -> dict:
    f = np.load(path, allow_pickle=True)
    return {k: f[k] for k in f.files}


def normalize_id_text(x: str) -> str:
    x = str(x).replace("\\", "/")
    x = Path(x).name
    x = strip_suffixes(x)
    x = x.replace("_preprocessed", "")
    x = x.replace("_brain_n4_rigid", "")
    x = x.replace("-seg", "")
    x = x.replace("_seg", "")
    return x


def extract_ixi_id(text: str) -> Optional[str]:
    text = normalize_id_text(text)
    m = re.search(r"(IXI\d{3,4}(?:-[A-Za-z]+-\d+)?)", text)
    if not m:
        return None
    return m.group(1)


def extract_gli_id(text: str) -> Optional[str]:
    text = normalize_id_text(text)
    m = re.search(r"(?:BraTS-)?(GLI-\d{5}-\d{3})", text)
    if not m:
        return None
    return m.group(1)


# -----------------------------------------------------------------------------
# Read GliGAN pairings
# -----------------------------------------------------------------------------

def parse_pairings_csv(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    pairs = []

    for idx, row in df.iterrows():
        row_dict = {c: row[c] for c in df.columns}

        ixi = None
        gli = None

        for c in ["ixi_id", "IXI_ID", "healthy_stem", "healthy_id"]:
            if c in df.columns and pd.notna(row[c]):
                ixi = extract_ixi_id(str(row[c]))
                if ixi:
                    break

        for c in ["gli_id", "GLI_ID", "label_stem", "brats_id", "BraTS_ID"]:
            if c in df.columns and pd.notna(row[c]):
                gli = extract_gli_id(str(row[c]))
                if gli:
                    break

        if ("case_dir" in df.columns) and pd.notna(row["case_dir"]):
            case_dir = str(row["case_dir"])
            if ixi is None:
                ixi = extract_ixi_id(case_dir)
            if gli is None:
                gli = extract_gli_id(case_dir)

        if ixi is None or gli is None:
            logging.warning(
                "Could not parse IXI/GLI from row %s. Parsed IXI=%s GLI=%s row=%s",
                idx, ixi, gli, row_dict
            )
            continue

        pairs.append({
            "pairing_row": int(idx),
            "ixi_id": ixi,
            "gli_id": gli,
            "raw_row": row_dict,
        })

    seen_ixi = set()
    unique_pairs = []

    for p in pairs:
        if p["ixi_id"] in seen_ixi:
            logging.warning("Duplicate IXI in pairings CSV, keeping first only: %s", p["ixi_id"])
            continue

        seen_ixi.add(p["ixi_id"])
        unique_pairs.append(p)

    return unique_pairs


def build_healthy_index(healthy_files: list[Path]) -> dict[str, Path]:
    idx = {}

    for p in healthy_files:
        ixi = extract_ixi_id(p.name)

        if ixi is None:
            logging.warning("Could not extract IXI ID from healthy file: %s", p)
            continue

        if ixi in idx:
            logging.warning("Duplicate healthy IXI ID found; keeping first: %s", ixi)
            continue

        idx[ixi] = p

    return idx


def build_library_index(library_items: list[Path]) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = {}

    for p in library_items:
        gli = extract_gli_id(p.name)

        if gli is None:
            logging.warning("Could not extract GLI ID from library item: %s", p)
            continue

        idx.setdefault(gli, []).append(p)

    return idx


# -----------------------------------------------------------------------------
# Original CarveMix core
# -----------------------------------------------------------------------------

def get_distance(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    f = mask.astype(bool)
    dist_func = ndimage.distance_transform_edt

    distance = np.where(
        f,
        -dist_func(f, sampling=spacing),
        dist_func(1 - f.astype(np.uint8), sampling=spacing)
    )

    return distance.astype(np.float32)


def normalization(values: np.ndarray, valid_mask: Optional[np.ndarray] = None):
    arr = values.astype(np.float32)

    if valid_mask is None:
        valid = np.isfinite(arr)
    else:
        valid = (valid_mask > 0) & np.isfinite(arr)

    vals = arr[valid]

    if vals.size == 0:
        return arr.copy(), 0.0, 1.0

    mean = float(np.mean(vals))
    std = float(np.std(vals))

    if std < 1e-8:
        std = 1.0

    normed = (arr - mean) / (std + 1e-8)
    return normed.astype(np.float32), mean, std


def sample_carvemix_lambda(distance: np.ndarray, rng: np.random.Generator) -> float:
    c = float(rng.beta(1.0, 1.0))
    c = (c - 0.5) * 2.0

    min_dis = float(np.min(distance))

    if c > 0:
        lam = c * min_dis / 2.0
    else:
        lam = c * min_dis

    return float(lam)


def make_carvemix_mask(
    tumor_mask: np.ndarray,
    spacing: tuple[float, float, float],
    rng: np.random.Generator,
):
    distance = get_distance(tumor_mask, spacing)
    lam = sample_carvemix_lambda(distance, rng)
    mix_mask = (distance < lam).astype(np.float32)

    return mix_mask, lam, distance


# -----------------------------------------------------------------------------
# Patch loading and original-location placement
# -----------------------------------------------------------------------------

def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    lbl, n = cc_label(mask > 0)

    if n == 0:
        return mask.astype(np.uint8)

    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    largest = int(counts.argmax())

    return (lbl == largest).astype(np.uint8)


def crop_from_bbox(arr: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    x0, y0, z0 = [int(v) for v in bbox_min]
    x1, y1, z1 = [int(v) for v in bbox_max]
    return arr[x0:x1, y0:y1, z0:z1]


def load_patch_and_mask(item_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    lib = load_npz(item_path)

    meta = {
        "library_item_path": str(item_path),
        "keys": list(lib.keys()),
    }

    if "patch_img" in lib and "patch_mask" in lib:
        patch = lib["patch_img"].astype(np.float32)
        mask = lib["patch_mask"].astype(np.uint8)

        meta["library_format"] = "patch_img_patch_mask"

        if "bbox_min" in lib and "bbox_max" in lib:
            meta["bbox_min"] = [int(v) for v in lib["bbox_min"]]
            meta["bbox_max"] = [int(v) for v in lib["bbox_max"]]

    elif "mask" in lib and "bbox_min" in lib and "bbox_max" in lib and "img_path" in lib:
        mask = lib["mask"].astype(np.uint8)
        bbox_min = lib["bbox_min"]
        bbox_max = lib["bbox_max"]
        img_path = Path(str(lib["img_path"]))

        source_img, _ = load_nifti(img_path)
        patch = crop_from_bbox(source_img, bbox_min, bbox_max).astype(np.float32)

        meta["library_format"] = "bbox_img_path"
        meta["source_img_path"] = str(img_path)
        meta["bbox_min"] = [int(v) for v in bbox_min]
        meta["bbox_max"] = [int(v) for v in bbox_max]

    else:
        raise KeyError(
            f"Unsupported library item format: {item_path}\n"
            f"Found keys: {list(lib.keys())}\n"
            f"Expected either patch_img+patch_mask or mask+bbox_min+bbox_max+img_path."
        )

    mask = largest_connected_component(mask)
    patch[mask == 0] = 0.0

    if int(mask.sum()) == 0:
        raise RuntimeError(f"Empty patch mask in library item: {item_path}")

    return patch.astype(np.float32), mask.astype(np.uint8), meta


def make_brain_mask(img: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    return (img > threshold).astype(np.uint8)


def place_at_original_location(
    patch: np.ndarray,
    patch_mask: np.ndarray,
    out_shape: tuple[int, int, int],
    lib_meta: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Original CarveMix-style placement.

    If patch/mask are already full-volume, use them directly.
    If patch/mask are cropped, paste them back to their original aligned location
    using bbox_min/bbox_max.
    """

    if patch.shape == out_shape and patch_mask.shape == out_shape:
        full_patch = patch.astype(np.float32)
        full_mask = patch_mask.astype(np.uint8)
        full_patch[full_mask == 0] = 0.0

        placement_meta = {
            "placement_mode": "original_location_full_volume",
            "bbox_min": None,
            "bbox_max": None,
        }

        return full_patch, full_mask, placement_meta

    if "bbox_min" not in lib_meta or "bbox_max" not in lib_meta:
        raise RuntimeError(
            "Cannot preserve original CarveMix location because this library item "
            "is cropped but has no bbox_min/bbox_max metadata. "
            "Regenerate the library with bbox_min and bbox_max saved."
        )

    bbox_min = [int(v) for v in lib_meta["bbox_min"]]
    bbox_max = [int(v) for v in lib_meta["bbox_max"]]

    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max

    if x0 < 0 or y0 < 0 or z0 < 0:
        raise RuntimeError(f"Invalid negative bbox_min: {bbox_min}")

    if x1 > out_shape[0] or y1 > out_shape[1] or z1 > out_shape[2]:
        raise RuntimeError(
            f"bbox_max exceeds target image shape. bbox_max={bbox_max}, out_shape={out_shape}"
        )

    expected_shape = (x1 - x0, y1 - y0, z1 - z0)

    if patch.shape != expected_shape or patch_mask.shape != expected_shape:
        raise RuntimeError(
            f"Patch shape does not match bbox size for original-location placement. "
            f"patch.shape={patch.shape}, mask.shape={patch_mask.shape}, "
            f"bbox_shape={expected_shape}"
        )

    full_patch = np.zeros(out_shape, dtype=np.float32)
    full_mask = np.zeros(out_shape, dtype=np.uint8)

    full_patch[x0:x1, y0:y1, z0:z1] = patch
    full_mask[x0:x1, y0:y1, z0:z1] = patch_mask

    full_patch[full_mask == 0] = 0.0

    placement_meta = {
        "placement_mode": "original_location_bbox",
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
    }

    return full_patch, full_mask, placement_meta


# -----------------------------------------------------------------------------
# Generate one synthetic sample
# -----------------------------------------------------------------------------

def output_stem_for(healthy_path: Path, item_path: Path) -> str:
    h = strip_suffixes(healthy_path.name)
    gli = extract_gli_id(item_path.name)

    if gli is None:
        gli = strip_suffixes(item_path.name)

    return f"{h}__BraTS-{gli}"


def generate_one(
    healthy_path: Path,
    item_path: Path,
    out_dirs: dict[str, Path],
    rng: np.random.Generator,
    args: argparse.Namespace,
    source_phase: str,
    preferred_gli_id: Optional[str],
) -> tuple[bool, dict]:
    meta_out = {
        "healthy_path": str(healthy_path),
        "library_item_path": str(item_path),
        "source_phase": source_phase,
        "preferred_gli_id": preferred_gli_id,
        "actual_gli_id": extract_gli_id(item_path.name),
    }

    healthy, healthy_nii = load_nifti(healthy_path)
    brain = make_brain_mask(healthy)

    if int(brain.sum()) == 0:
        return False, {**meta_out, "failure_reason": "empty_brain_mask"}

    patch, patch_mask, lib_meta = load_patch_and_mask(item_path)

    placed_patch, placed_tumor_mask, placement_meta = place_at_original_location(
        patch=patch,
        patch_mask=patch_mask,
        out_shape=healthy.shape,
        lib_meta=lib_meta,
    )

    placed_patch[placed_tumor_mask == 0] = 0.0

    if int(placed_tumor_mask.sum()) == 0:
        return False, {**meta_out, "failure_reason": "empty_original_location_tumor_mask"}

    spacing = tuple(float(x) for x in healthy_nii.header.get_zooms()[:3])

    carvemix_mask, lam, _ = make_carvemix_mask(
        tumor_mask=placed_tumor_mask,
        spacing=spacing,
        rng=rng,
    )

    carvemix_mask = carvemix_mask.astype(np.float32)

    if int(carvemix_mask.sum()) == 0:
        return False, {**meta_out, "failure_reason": "empty_carvemix_mask"}

    healthy_norm, healthy_mean, healthy_std = normalization(healthy, valid_mask=brain)
    patch_norm, patch_mean, patch_std = normalization(placed_patch, valid_mask=placed_tumor_mask)

    synthetic_norm = healthy_norm * (carvemix_mask == 0) + patch_norm * carvemix_mask
    synthetic = synthetic_norm * healthy_std + healthy_mean

    synthetic = synthetic.astype(np.float32)
    synthetic[brain == 0] = healthy[brain == 0]

    diff = np.abs(synthetic - healthy).astype(np.float32)

    if float(diff.sum()) <= 0:
        return False, {**meta_out, "failure_reason": "synthetic_identical_to_healthy"}

    stem = output_stem_for(healthy_path, item_path)

    out_healthy = out_dirs["healthy"] / f"{stem}.nii.gz"
    out_synthetic = out_dirs["synthetic"] / f"{stem}_carvemix.nii.gz"
    out_tumor_mask = out_dirs["tumor_mask"] / f"{stem}_tumor_mask.nii.gz"
    out_carvemix_mask = out_dirs["carvemix_mask"] / f"{stem}_carvemix_mask.nii.gz"
    out_diff = out_dirs["diff"] / f"{stem}_diff.nii.gz"
    out_meta = out_dirs["metadata"] / f"{stem}_metadata.json"

    save_nifti(healthy, healthy_nii, out_healthy)
    save_nifti(synthetic, healthy_nii, out_synthetic)
    save_nifti(placed_tumor_mask.astype(np.float32), healthy_nii, out_tumor_mask)
    save_nifti(carvemix_mask.astype(np.float32), healthy_nii, out_carvemix_mask)
    save_nifti(diff, healthy_nii, out_diff)

    meta = {
        **meta_out,
        "success": True,
        "output_stem": stem,
        "outputs": {
            "healthy": str(out_healthy),
            "synthetic": str(out_synthetic),
            "tumor_mask": str(out_tumor_mask),
            "carvemix_mask": str(out_carvemix_mask),
            "diff": str(out_diff),
            "metadata": str(out_meta),
        },
        "placement": placement_meta,
        "spacing": [float(x) for x in spacing],
        "lambda": float(lam),
        "healthy_mean": float(healthy_mean),
        "healthy_std": float(healthy_std),
        "patch_mean": float(patch_mean),
        "patch_std": float(patch_std),
        "tumor_mask_voxels": int(placed_tumor_mask.sum()),
        "carvemix_mask_voxels": int(carvemix_mask.sum()),
        "diff_sum": float(diff.sum()),
        "diff_max": float(diff.max()),
        "library_meta": lib_meta,
    }

    save_json(meta, out_meta)

    return True, meta


# -----------------------------------------------------------------------------
# Pairing runner
# -----------------------------------------------------------------------------

def choose_unused_random_item(
    library_items: list[Path],
    attempted_items: set[str],
    rng_py: random.Random,
) -> Optional[Path]:
    candidates = [p for p in library_items if str(p) not in attempted_items]

    if not candidates:
        return None

    return rng_py.choice(candidates)


def attempt_until_success_for_healthy(
    healthy_path: Path,
    preferred_item: Optional[Path],
    preferred_gli_id: Optional[str],
    library_items: list[Path],
    attempted_items: set[str],
    out_dirs: dict[str, Path],
    rng_np: np.random.Generator,
    rng_py: random.Random,
    args: argparse.Namespace,
    source_phase: str,
    max_fallbacks: int,
) -> tuple[bool, Optional[dict], list[dict]]:
    attempts = []
    candidate_items = []

    if preferred_item is not None and str(preferred_item) not in attempted_items:
        candidate_items.append(("preferred", preferred_item))

    for fallback_idx in range(max_fallbacks):
        already_candidate = {str(p) for _, p in candidate_items}
        item = choose_unused_random_item(
            library_items=library_items,
            attempted_items=attempted_items | already_candidate,
            rng_py=rng_py,
        )

        if item is None:
            break

        candidate_items.append((f"fallback_{fallback_idx + 1}", item))

    for attempt_type, item_path in candidate_items:
        attempted_items.add(str(item_path))

        logging.info(
            "Trying %s | IXI=%s | item=%s | preferred_gli=%s | actual_gli=%s",
            attempt_type,
            healthy_path.name,
            item_path.name,
            preferred_gli_id,
            extract_gli_id(item_path.name),
        )

        try:
            ok, meta = generate_one(
                healthy_path=healthy_path,
                item_path=item_path,
                out_dirs=out_dirs,
                rng=rng_np,
                args=args,
                source_phase=source_phase,
                preferred_gli_id=preferred_gli_id,
            )
        except Exception as e:
            ok = False
            meta = {
                "healthy_path": str(healthy_path),
                "library_item_path": str(item_path),
                "source_phase": source_phase,
                "preferred_gli_id": preferred_gli_id,
                "actual_gli_id": extract_gli_id(item_path.name),
                "failure_reason": repr(e),
            }

        attempt_row = {
            "source_phase": source_phase,
            "attempt_type": attempt_type,
            "healthy_path": str(healthy_path),
            "ixi_id": extract_ixi_id(healthy_path.name),
            "preferred_gli_id": preferred_gli_id,
            "library_item_path": str(item_path),
            "actual_gli_id": extract_gli_id(item_path.name),
            "success": bool(ok),
            "failure_reason": meta.get("failure_reason", ""),
            "output_stem": meta.get("output_stem", ""),
        }

        attempts.append(attempt_row)

        if ok:
            logging.info("SUCCESS | %s | %s", healthy_path.name, item_path.name)
            return True, meta, attempts

        logging.warning(
            "FAILED | %s | %s | reason=%s",
            healthy_path.name,
            item_path.name,
            meta.get("failure_reason", "unknown"),
        )

    return False, None, attempts


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CarveMix generation using GliGAN pairings as preferred pairs, preserving original lesion location."
    )

    parser.add_argument("--healthy-dir", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--gligan-pairings-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name-filter", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--max-fallbacks-per-ixi",
        type=int,
        default=50,
        help="If preferred BraTS item fails, try up to this many unused random tumor items for the same IXI."
    )

    parser.add_argument(
        "--target-total",
        type=int,
        default=550,
        help="Target number of successful outputs. If below this after GliGAN-matched phase, top up randomly."
    )

    parser.add_argument(
        "--extra-random-if-under",
        type=int,
        default=50,
        help="If final count after matched phase is under target-total, generate up to this many extra random unused pairs."
    )

    parser.add_argument(
        "--num-gligan-pairs",
        type=int,
        default=None,
        help="Optional cap on how many GliGAN pairings to attempt first."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_path = setup_logging(args.output_dir)

    rng_np = np.random.default_rng(args.seed)
    rng_py = random.Random(args.seed)
    random.seed(args.seed)

    healthy_files = find_nifti_images(args.healthy_dir, args.name_filter)
    library_items = load_library_items(args.library_dir)

    if len(healthy_files) == 0:
        raise FileNotFoundError(f"No healthy NIfTI files found in: {args.healthy_dir}")

    if len(library_items) == 0:
        raise FileNotFoundError(f"No library items found in: {args.library_dir / 'items'}")

    healthy_index = build_healthy_index(healthy_files)
    library_index = build_library_index(library_items)
    gligan_pairs = parse_pairings_csv(args.gligan_pairings_csv)

    if args.num_gligan_pairs is not None:
        gligan_pairs = gligan_pairs[:int(args.num_gligan_pairs)]

    out_dirs = {
        "healthy": args.output_dir / "healthy",
        "synthetic": args.output_dir / "synthetic",
        "tumor_mask": args.output_dir / "tumor_masks",
        "carvemix_mask": args.output_dir / "carvemix_masks",
        "diff": args.output_dir / "diff",
        "metadata": args.output_dir / "metadata",
    }

    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    logging.info("Log file: %s", log_path)
    logging.info("Healthy files available: %d", len(healthy_files))
    logging.info("Healthy IXI IDs indexed: %d", len(healthy_index))
    logging.info("Library items available: %d", len(library_items))
    logging.info("Library GLI IDs indexed: %d", len(library_index))
    logging.info("GliGAN pairings loaded: %d", len(gligan_pairs))
    logging.info("Target total: %d", args.target_total)
    logging.info("Extra random if under target: %d", args.extra_random_if_under)

    attempted_items: set[str] = set()
    used_healthy_success: set[str] = set()

    all_attempts = []
    successes = []
    failed_ixi = []

    # -------------------------------------------------------------------------
    # Phase 1: Follow GliGAN pairings whenever possible
    # -------------------------------------------------------------------------

    for pair_idx, pair in enumerate(gligan_pairs, start=1):
        ixi_id = pair["ixi_id"]
        gli_id = pair["gli_id"]

        if ixi_id in used_healthy_success:
            logging.warning(
                "[%d/%d] IXI already successfully used, skipping: %s",
                pair_idx,
                len(gligan_pairs),
                ixi_id,
            )
            continue

        healthy_path = healthy_index.get(ixi_id)

        if healthy_path is None:
            logging.warning(
                "[%d/%d] Missing healthy MRI for IXI=%s",
                pair_idx,
                len(gligan_pairs),
                ixi_id,
            )
            failed_ixi.append({
                "source_phase": "gligan_matched",
                "ixi_id": ixi_id,
                "preferred_gli_id": gli_id,
                "reason": "missing_healthy",
            })
            continue

        preferred_items = library_index.get(gli_id, [])
        preferred_item = None

        for cand in preferred_items:
            if str(cand) not in attempted_items:
                preferred_item = cand
                break

        if preferred_item is None:
            logging.warning(
                "[%d/%d] Preferred GLI not available/unused for IXI=%s GLI=%s. Will use fallback.",
                pair_idx,
                len(gligan_pairs),
                ixi_id,
                gli_id,
            )

        logging.info(
            "[%d/%d] Matched phase | IXI=%s | preferred GLI=%s | preferred item=%s",
            pair_idx,
            len(gligan_pairs),
            ixi_id,
            gli_id,
            preferred_item.name if preferred_item else "NONE",
        )

        ok, meta, attempts = attempt_until_success_for_healthy(
            healthy_path=healthy_path,
            preferred_item=preferred_item,
            preferred_gli_id=gli_id,
            library_items=library_items,
            attempted_items=attempted_items,
            out_dirs=out_dirs,
            rng_np=rng_np,
            rng_py=rng_py,
            args=args,
            source_phase="gligan_matched",
            max_fallbacks=args.max_fallbacks_per_ixi,
        )

        all_attempts.extend(attempts)

        if ok and meta is not None:
            used_healthy_success.add(ixi_id)
            successes.append({
                "source_phase": "gligan_matched",
                "ixi_id": ixi_id,
                "preferred_gli_id": gli_id,
                "actual_gli_id": meta.get("actual_gli_id"),
                "healthy_path": meta.get("healthy_path"),
                "library_item_path": meta.get("library_item_path"),
                "output_stem": meta.get("output_stem"),
                "metadata_path": meta.get("outputs", {}).get("metadata"),
            })
        else:
            failed_ixi.append({
                "source_phase": "gligan_matched",
                "ixi_id": ixi_id,
                "preferred_gli_id": gli_id,
                "reason": "all_attempts_failed_or_no_unused_items",
            })

    # -------------------------------------------------------------------------
    # Phase 2: If under target, generate extra random unused IXI/tumor pairs
    # Note: pair choice is random, but lesion location is still original-location.
    # -------------------------------------------------------------------------

    count_after_matched = len(successes)
    logging.info("Matched phase done. Successes=%d", count_after_matched)

    if count_after_matched < args.target_total and args.extra_random_if_under > 0:
        needed_to_target = args.target_total - count_after_matched
        extra_n = min(args.extra_random_if_under, needed_to_target)

        unused_healthy = [
            p for p in healthy_files
            if (
                extract_ixi_id(p.name) is not None
                and extract_ixi_id(p.name) not in used_healthy_success
            )
        ]

        rng_py.shuffle(unused_healthy)

        logging.info(
            "Under target. Attempting up to %d random extra successful outputs from %d unused healthy MRIs.",
            extra_n,
            len(unused_healthy),
        )

        extra_successes = 0

        for h_idx, healthy_path in enumerate(unused_healthy, start=1):
            if extra_successes >= extra_n:
                break

            ixi_id = extract_ixi_id(healthy_path.name)

            logging.info(
                "[extra %d/%d] Random top-up | IXI=%s | successes=%d/%d",
                h_idx,
                len(unused_healthy),
                ixi_id,
                extra_successes,
                extra_n,
            )

            ok, meta, attempts = attempt_until_success_for_healthy(
                healthy_path=healthy_path,
                preferred_item=None,
                preferred_gli_id=None,
                library_items=library_items,
                attempted_items=attempted_items,
                out_dirs=out_dirs,
                rng_np=rng_np,
                rng_py=rng_py,
                args=args,
                source_phase="random_topup",
                max_fallbacks=args.max_fallbacks_per_ixi,
            )

            all_attempts.extend(attempts)

            if ok and meta is not None:
                used_healthy_success.add(ixi_id)
                extra_successes += 1
                successes.append({
                    "source_phase": "random_topup",
                    "ixi_id": ixi_id,
                    "preferred_gli_id": None,
                    "actual_gli_id": meta.get("actual_gli_id"),
                    "healthy_path": meta.get("healthy_path"),
                    "library_item_path": meta.get("library_item_path"),
                    "output_stem": meta.get("output_stem"),
                    "metadata_path": meta.get("outputs", {}).get("metadata"),
                })

        logging.info("Random top-up done. Extra successes=%d", extra_successes)

    # -------------------------------------------------------------------------
    # Save reports
    # -------------------------------------------------------------------------

    attempts_df = pd.DataFrame(all_attempts)
    successes_df = pd.DataFrame(successes)
    failed_df = pd.DataFrame(failed_ixi)

    attempts_csv = args.output_dir / "all_attempts.csv"
    successes_csv = args.output_dir / "successful_carvemix_pairings.csv"
    failed_csv = args.output_dir / "failed_ixi_pairings.csv"

    attempts_df.to_csv(attempts_csv, index=False)
    successes_df.to_csv(successes_csv, index=False)
    failed_df.to_csv(failed_csv, index=False)

    summary = {
        "healthy_dir": str(args.healthy_dir),
        "library_dir": str(args.library_dir),
        "gligan_pairings_csv": str(args.gligan_pairings_csv),
        "output_dir": str(args.output_dir),
        "seed": int(args.seed),
        "target_total": int(args.target_total),
        "extra_random_if_under": int(args.extra_random_if_under),
        "num_healthy_files": int(len(healthy_files)),
        "num_library_items": int(len(library_items)),
        "num_gligan_pairings_loaded": int(len(gligan_pairs)),
        "num_success": int(len(successes)),
        "num_failed_ixi": int(len(failed_ixi)),
        "num_item_attempts": int(len(all_attempts)),
        "no_successful_healthy_reuse": True,
        "no_attempted_tumor_item_reuse": True,
        "placement_mode": "original_location_carvemix_style",
        "attempts_csv": str(attempts_csv),
        "successes_csv": str(successes_csv),
        "failed_csv": str(failed_csv),
        "log_path": str(log_path),
    }

    save_json(summary, args.output_dir / "summary.json")

    logging.info("Done.")
    logging.info("Successes: %d", len(successes))
    logging.info("Failed IXIs: %d", len(failed_ixi))
    logging.info("Item attempts: %d", len(all_attempts))
    logging.info("Saved attempts: %s", attempts_csv)
    logging.info("Saved successes: %s", successes_csv)
    logging.info("Saved failed IXIs: %s", failed_csv)
    logging.info("Saved summary: %s", args.output_dir / "summary.json")


if __name__ == "__main__":
    main()