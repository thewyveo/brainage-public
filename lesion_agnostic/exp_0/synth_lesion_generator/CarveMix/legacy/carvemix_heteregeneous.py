#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
One-to-one CarveMix-style synthetic tumor generation.


Core logic preserved from original CarveMix:
1. take tumor patch + tumor mask
2. compute signed distance transform on tumor mask
3. sample lambda
4. create CarveMix mask M = distance < lambda
5. mix: healthy * (1-M) + tumor_patch * M
6. each tumor item and healthy MRI is used at most once


Expected library format:
library_dir/items/*.npz


Each .npz should contain either:


Format A:
   patch_img
   patch_mask


or Format B:
   mask
   bbox_min
   bbox_max
   img_path


Example:
python carvemix_heterogeneous.py \
 --healthy-dir data/preprocessed/healthy \
 --library-dir data/library/BraTS_Masks \
 --output-dir data/generated/carvemix_heterogeneous \
 --name-filter t1 \
 --seed 42
"""


from __future__ import annotations


import argparse
import json
import logging
import random
from pathlib import Path
from typing import Optional


import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.ndimage import label as cc_label




# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------


def setup_logging() -> None:
   logging.basicConfig(
       level=logging.INFO,
       format="%(asctime)s %(levelname)s: %(message)s"
   )




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




# -----------------------------------------------------------------------------
# Original CarveMix core
# -----------------------------------------------------------------------------


def get_distance(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
   """
   Same signed-distance idea as original CarveMix.


   Inside mask  -> negative distance
   Outside mask -> positive distance
   """
   f = mask.astype(bool)
   dist_func = ndimage.distance_transform_edt


   distance = np.where(
       f,
       -dist_func(f, sampling=spacing),
       dist_func(1 - f.astype(np.uint8), sampling=spacing)
   )


   return distance.astype(np.float32)




def normalization(values: np.ndarray, valid_mask: Optional[np.ndarray] = None):
   """
   Original CarveMix normalizes each image before mixing.


   Here we normalize using valid brain/patch voxels instead of full zero-padded volume.
   """
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
   """
   Same lambda logic as original script.


   c ~ Beta(1,1)
   c = (c - 0.5) * 2


   if c > 0:
       lam = c * min(distance) / 2
   else:
       lam = c * min(distance)
   """
   c = float(rng.beta(1.0, 1.0))
   c = (c - 0.5) * 2.0


   min_dis = float(np.min(distance))


   if c > 0:
       lam = c * min_dis / 2.0
   else:
       lam = c * min_dis


   return float(lam)




def make_carvemix_mask(tumor_mask: np.ndarray, spacing: tuple[float, float, float], rng: np.random.Generator):
   """
   Creates M from the signed distance map.


   This is the key original CarveMix step:
       M = distance < lambda
   """
   distance = get_distance(tumor_mask, spacing)
   lam = sample_carvemix_lambda(distance, rng)
   mix_mask = (distance < lam).astype(np.float32)


   return mix_mask, lam, distance




# -----------------------------------------------------------------------------
# Patch loading and placement
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
   """
   Supports both your newer patch library and the earlier bbox-based library.
   """
   lib = load_npz(item_path)


   meta = {
       "library_item_path": str(item_path),
       "keys": list(lib.keys()),
   }


   if "patch_img" in lib and "patch_mask" in lib:
       patch = lib["patch_img"].astype(np.float32)
       mask = lib["patch_mask"].astype(np.uint8)


       meta["library_format"] = "patch_img_patch_mask"


   elif "mask" in lib and "bbox_min" in lib and "bbox_max" in lib and "img_path" in lib:
       mask = lib["mask"].astype(np.uint8)
       bbox_min = lib["bbox_min"]
       bbox_max = lib["bbox_max"]
       img_path = Path(str(lib["img_path"]))


       source_img, _ = load_nifti(img_path)
       patch = crop_from_bbox(source_img, bbox_min, bbox_max).astype(np.float32)


       meta["library_format"] = "bbox_img_path"
       meta["source_img_path"] = str(img_path)


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




def paste_at_center(
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
   small_mask: np.ndarray,
   brain_mask: np.ndarray,
   rng: np.random.Generator,
   max_tries: int,
   min_inside_ratio: float,
) -> Optional[tuple[int, int, int]]:
   coords = np.argwhere(brain_mask > 0)


   if coords.size == 0:
       return None


   for _ in range(max_tries):
       c = coords[int(rng.integers(0, len(coords)))]
       center = (int(c[0]), int(c[1]), int(c[2]))


       placed = paste_at_center(
           arr_small=small_mask,
           out_shape=brain_mask.shape,
           center=center,
           fill_value=0
       ).astype(np.uint8)


       voxels = float(placed.sum())
       if voxels <= 0:
           continue


       inside = float((placed * brain_mask).sum())
       ratio = inside / voxels


       if ratio >= min_inside_ratio:
           return center


   return None




# -----------------------------------------------------------------------------
# Generate one synthetic sample
# -----------------------------------------------------------------------------


def generate_one(
   healthy_path: Path,
   item_path: Path,
   out_dirs: dict[str, Path],
   rng: np.random.Generator,
   args: argparse.Namespace,
) -> bool:
   healthy, healthy_nii = load_nifti(healthy_path)
   brain = make_brain_mask(healthy)


   if int(brain.sum()) == 0:
       logging.warning("Empty brain mask: %s", healthy_path)
       return False


   patch, patch_mask, lib_meta = load_patch_and_mask(item_path)


   center = sample_valid_center(
       small_mask=patch_mask,
       brain_mask=brain,
       rng=rng,
       max_tries=args.max_placement_tries,
       min_inside_ratio=args.min_inside_ratio,
   )


   if center is None:
       logging.warning("Could not place tumor item %s into %s", item_path.name, healthy_path.name)
       return False


   placed_patch = paste_at_center(
       arr_small=patch,
       out_shape=healthy.shape,
       center=center,
       fill_value=0.0,
   ).astype(np.float32)


   placed_tumor_mask = paste_at_center(
       arr_small=patch_mask,
       out_shape=healthy.shape,
       center=center,
       fill_value=0,
   ).astype(np.uint8)


   placed_patch[placed_tumor_mask == 0] = 0.0


   spacing = tuple(float(x) for x in healthy_nii.header.get_zooms()[:3])


   carvemix_mask, lam, _ = make_carvemix_mask(
       tumor_mask=placed_tumor_mask,
       spacing=spacing,
       rng=rng,
   )


   carvemix_mask = carvemix_mask.astype(np.float32)


   if int(carvemix_mask.sum()) == 0:
       logging.warning("CarveMix mask became empty for %s", healthy_path.name)
       return False


   # Original CarveMix-style normalization:
   # normalize target and source separately, mix, restore using target stats.
   healthy_norm, healthy_mean, healthy_std = normalization(healthy, valid_mask=brain)
   patch_norm, patch_mean, patch_std = normalization(placed_patch, valid_mask=placed_tumor_mask)


   synthetic_norm = healthy_norm * (carvemix_mask == 0) + patch_norm * carvemix_mask
   synthetic = synthetic_norm * healthy_std + healthy_mean


   synthetic = synthetic.astype(np.float32)
   synthetic[brain == 0] = healthy[brain == 0]


   diff = np.abs(synthetic - healthy).astype(np.float32)


   if float(diff.sum()) <= 0:
       logging.warning("Synthetic identical to healthy: %s", healthy_path.name)
       return False


   stem = strip_suffixes(healthy_path.name)


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
       "healthy_path": str(healthy_path),
       "library_item_path": str(item_path),
       "center": [int(x) for x in center],
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


   return True




# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
   parser = argparse.ArgumentParser(
       description="One-to-one original-core CarveMix generation for healthy MRIs."
   )


   parser.add_argument("--healthy-dir", type=Path, required=True)
   parser.add_argument("--library-dir", type=Path, required=True)
   parser.add_argument("--output-dir", type=Path, required=True)
   parser.add_argument("--name-filter", type=str, default=None)


   parser.add_argument("--seed", type=int, default=42)
   parser.add_argument("--max-placement-tries", type=int, default=100)
   parser.add_argument("--min-inside-ratio", type=float, default=0.98)


   parser.add_argument(
       "--num-samples",
       type=int,
       default=None,
       help="Optional cap. Still one-to-one: no healthy scan or tumor item is reused."
   )


   return parser.parse_args()




def main() -> None:
   setup_logging()
   args = parse_args()


   rng = np.random.default_rng(args.seed)
   random.seed(args.seed)


   healthy_files = find_nifti_images(args.healthy_dir, args.name_filter)
   library_items = load_library_items(args.library_dir)


   if len(healthy_files) == 0:
       raise FileNotFoundError(f"No healthy NIfTI files found in: {args.healthy_dir}")


   if len(library_items) == 0:
       raise FileNotFoundError(f"No library items found in: {args.library_dir / 'items'}")


   random.shuffle(healthy_files)
   random.shuffle(library_items)


   n = min(len(healthy_files), len(library_items))


   if args.num_samples is not None:
       n = min(n, int(args.num_samples))


   healthy_files = healthy_files[:n]
   library_items = library_items[:n]


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


   logging.info("Healthy files available: %d", len(find_nifti_images(args.healthy_dir, args.name_filter)))
   logging.info("Library items available: %d", len(load_library_items(args.library_dir)))
   logging.info("One-to-one pairs to generate: %d", n)


   success = 0
   failed = 0
   pairs = []


   for i, (healthy_path, item_path) in enumerate(zip(healthy_files, library_items), start=1):
       logging.info("[%d/%d] %s  <--  %s", i, n, healthy_path.name, item_path.name)


       ok = generate_one(
           healthy_path=healthy_path,
           item_path=item_path,
           out_dirs=out_dirs,
           rng=rng,
           args=args,
       )


       pair_info = {
           "index": i,
           "healthy_path": str(healthy_path),
           "library_item_path": str(item_path),
           "success": bool(ok),
       }


       pairs.append(pair_info)


       if ok:
           success += 1
       else:
           failed += 1


   summary = {
       "healthy_dir": str(args.healthy_dir),
       "library_dir": str(args.library_dir),
       "output_dir": str(args.output_dir),
       "seed": int(args.seed),
       "num_pairs_attempted": int(n),
       "num_success": int(success),
       "num_failed": int(failed),
       "one_to_one_no_reuse": True,
       "pairs": pairs,
   }


   save_json(summary, args.output_dir / "summary.json")


   logging.info("Done. Success=%d Failed=%d", success, failed)




if __name__ == "__main__":
   main()
