#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Research-backed CarveMix + Guizard + T1-glioma intensity variant


Design:
- CarveMix provides lesion geometry and source patch
- Guizard-inspired contextual filling provides local continuity
- T1 lesion appearance is modeled as a stochastic deviation from
 local tissue statistics, biased darker than surrounding brain


Why:
- Guizard supports local intensity propagation and smooth anatomical
 continuity in lesion filling
- Glioma MRI literature supports T1 hypointensity / heterogeneity


Example:
py -3.10 exp_0\synth_lesion_generator\CarveMix\carvemix_guizard_gen.py `
 --healthy-dir data\preprocessed\BrainAgeNeXt\IXI `
 --library-dir data\library\Guizard_CarveMix\npz_files\ `
 --output-dir data\library\Guizard_CarveMix\generated\ `
 --seed 42 `
 --enable-center-darkening `
 --t1-dark-alpha-min 3 --t1-dark-alpha-max 3.7 `
 --t1-std-beta-min 0.4 --t1-std-beta-max 0.55 `
 --center-dark-strength-min 0.3 --center-dark-strength-max 0.7 `
 --center-dark-gamma-min 2.0 --center-dark-gamma-max 3.5 `
 --soft-sigma-min 1.2 --soft-sigma-max 2.0 `
 --boundary-sigma-min 0.9 --boundary-sigma-max 1.8
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




def save_json(obj: dict, path: Path):
   path.parent.mkdir(parents=True, exist_ok=True)
   with open(path, "w", encoding="utf-8") as f:
       json.dump(obj, f, indent=2)




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
       raise RuntimeError(f"No library items found under: {lib_dir / 'items'}")
   return items




def load_lib(item_path: Path):
   f = np.load(item_path, allow_pickle=True)
   return {k: f[k] for k in f.files}




# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------




def brain_mask(img: np.ndarray) -> np.ndarray:
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
# Guizard-inspired contextual filling
# -----------------------------------------------------------------------------




def masked_gaussian_average(values: np.ndarray, valid_mask: np.ndarray, sigma: float) -> np.ndarray:
   values = values.astype(np.float32)
   valid_mask = valid_mask.astype(np.float32)


   num = gaussian_filter(values * valid_mask, sigma=sigma)
   den = gaussian_filter(valid_mask, sigma=sigma)


   out = np.zeros_like(values, dtype=np.float32)
   good = den > 1e-6
   out[good] = num[good] / den[good]
   return out




def get_outer_layer(mask: np.ndarray) -> np.ndarray:
   mask = mask.astype(bool)
   if not np.any(mask):
       return mask.astype(np.uint8)


   eroded = binary_erosion(mask, iterations=1, border_value=0)
   outer = np.logical_and(mask, ~eroded)
   return outer.astype(np.uint8)




def guizard_concentric_context_fill(
   healthy_img: np.ndarray,
   lesion_mask: np.ndarray,
   brain: np.ndarray,
   sigma_schedule: list[float],
) -> np.ndarray:
   """
   Concentric local-context fill inspired by Guizard:
   fill from outside inward using weighted local averages.
   """
   lesion_mask = lesion_mask.astype(bool)
   brain = brain.astype(bool)


   filled = healthy_img.astype(np.float32).copy()
   unknown = lesion_mask.copy()
   known = np.logical_and(~unknown, brain)


   if not np.any(unknown):
       return np.zeros_like(healthy_img, dtype=np.float32)


   for sigma in sigma_schedule:
       while np.any(unknown):
           layer = get_outer_layer(unknown)
           layer = np.logical_and(layer, brain)


           if not np.any(layer):
               break


           local_avg = masked_gaussian_average(filled, known, sigma=sigma)
           filled[layer] = local_avg[layer]


           known[layer] = True
           unknown[layer] = False


   out = np.zeros_like(healthy_img, dtype=np.float32)
   out[lesion_mask] = filled[lesion_mask]
   return out




# -----------------------------------------------------------------------------
# T1 tumor appearance model
# -----------------------------------------------------------------------------




def robust_zscore(vals: np.ndarray, eps: float = 1e-6) -> np.ndarray:
   vals = vals.astype(np.float32)
   mu = float(np.mean(vals))
   sd = float(np.std(vals))
   if sd < eps:
       sd = 1.0
   return (vals - mu) / sd




def compute_local_context_stats(
   healthy_img: np.ndarray,
   lesion_mask: np.ndarray,
   brain: np.ndarray,
   ring_iters: int,
) -> tuple[np.ndarray, float, float]:
   ring = make_ring(lesion_mask, dilate_iters=ring_iters).astype(bool)
   ring = np.logical_and(ring, brain.astype(bool))


   vals = healthy_img[ring]
   if vals.size == 0:
       vals = healthy_img[np.logical_and(brain.astype(bool), ~lesion_mask.astype(bool))]
   if vals.size == 0:
       vals = healthy_img[brain.astype(bool)]


   mu = float(np.mean(vals))
   sd = float(np.std(vals))
   if sd < 1e-6:
       sd = 1.0
   return vals.astype(np.float32), mu, sd




def sample_t1_tumor_relative_to_context(
   source_vals: np.ndarray,
   context_field_vals: np.ndarray,
   context_mean: float,
   context_std: float,
   lesion_mask: np.ndarray,
   rng: np.random.Generator,
   args: argparse.Namespace,
) -> np.ndarray:
   """
   Research-backed T1 lesion model:
   - tumors are typically darker than local brain on T1
   - lesion remains heterogeneous
   - source patch texture is retained weakly
   - context field keeps continuity
   """
   n = source_vals.size
   if n == 0:
       return np.array([], dtype=np.float32)


   src_z = robust_zscore(source_vals)


   # T1 hypointense relative shift:
   # target_mean = ctx_mean - alpha * ctx_std
   alpha = float(rng.uniform(args.t1_dark_alpha_min, args.t1_dark_alpha_max))
   beta = float(rng.uniform(args.t1_std_beta_min, args.t1_std_beta_max))


   target_mean = context_mean - alpha * context_std
   target_std = max(1e-6, beta * context_std)


   texture_vals = target_mean + src_z * target_std


   # Mild additional stochastic heterogeneity
   noise_sd = float(rng.uniform(args.t1_noise_frac_min, args.t1_noise_frac_max)) * context_std
   texture_vals = texture_vals + rng.normal(0.0, noise_sd, size=n).astype(np.float32)


   # Optional very mild center darkening
   if args.enable_center_darkening:
       dist = distance_transform_edt(lesion_mask.astype(bool)).astype(np.float32)
       if dist.max() > 0:
           dist = dist / dist.max()
           center_gamma = float(rng.uniform(args.center_dark_gamma_min, args.center_dark_gamma_max))
           center_strength = float(rng.uniform(args.center_dark_strength_min, args.center_dark_strength_max))
           texture_vals = texture_vals - center_strength * context_std * (dist[lesion_mask.astype(bool)] ** center_gamma)


   # Mix with Guizard-style context field for continuity
   context_mix = float(rng.uniform(args.context_mix_min, args.context_mix_max))
   out_vals = context_mix * texture_vals + (1.0 - context_mix) * context_field_vals


   return out_vals.astype(np.float32)




def adapt_patch_guizard_t1(
   source_patch: np.ndarray,
   placed_mask: np.ndarray,
   healthy_img: np.ndarray,
   brain: np.ndarray,
   rng: np.random.Generator,
   args: argparse.Namespace,
) -> tuple[np.ndarray, dict]:
   out = np.zeros_like(source_patch, dtype=np.float32)
   mask = placed_mask.astype(bool)


   if not np.any(mask):
       return out, {}


   ring_iters = int(rng.integers(args.context_ring_min, args.context_ring_max + 1))
   _, context_mean, context_std = compute_local_context_stats(
       healthy_img=healthy_img,
       lesion_mask=placed_mask,
       brain=brain,
       ring_iters=ring_iters,
   )


   # Source smoothing (retain some source tumor structure)
   source_sigma = float(rng.uniform(args.patch_sigma_min, args.patch_sigma_max))
   smoothed_source = masked_gaussian_average(source_patch, mask.astype(np.float32), sigma=source_sigma)
   source_vals = smoothed_source[mask]


   # Guizard smooth-to-fine context field
   high = float(rng.uniform(args.fill_sigma_high_min, args.fill_sigma_high_max))
   mid = float(rng.uniform(args.fill_sigma_mid_min, args.fill_sigma_mid_max))
   low = float(rng.uniform(args.fill_sigma_low_min, args.fill_sigma_low_max))
   sigma_schedule = [high, mid, low]


   context_field = guizard_concentric_context_fill(
       healthy_img=healthy_img,
       lesion_mask=placed_mask,
       brain=brain,
       sigma_schedule=sigma_schedule,
   )


   context_field_vals = context_field[mask]


   final_vals = sample_t1_tumor_relative_to_context(
       source_vals=source_vals,
       context_field_vals=context_field_vals,
       context_mean=context_mean,
       context_std=context_std,
       lesion_mask=placed_mask,
       rng=rng,
       args=args,
   )


   # Clamp to plausible local range
   brain_vals = healthy_img[brain.astype(bool)]
   if brain_vals.size > 0:
       p1, p99 = np.percentile(brain_vals, [1, 99])
       final_vals = np.clip(final_vals, p1, p99)


   out[mask] = final_vals


   meta = {
       "context_ring_iters": ring_iters,
       "source_sigma": source_sigma,
       "fill_sigma_schedule": [high, mid, low],
       "context_mean": context_mean,
       "context_std": context_std,
   }
   return out.astype(np.float32), meta




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
# Main generator
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


   center = find_valid_center(
       mask,
       brain,
       rng,
       tries=args.max_placement_tries,
       min_inside_ratio=args.min_inside_ratio,
   )
   if center is None:
       logging.warning("Could not find valid center for %s", h_path.name)
       return False


   placed_mask = paste(mask, healthy.shape, center, fill_value=0).astype(np.uint8)
   placed_patch = paste(patch, healthy.shape, center, fill_value=0.0).astype(np.float32)
   placed_patch[placed_mask == 0] = 0.0


   adapted_patch, meta = adapt_patch_guizard_t1(
       source_patch=placed_patch,
       placed_mask=placed_mask,
       healthy_img=healthy,
       brain=brain,
       rng=rng,
       args=args,
   )


   soft_sigma = float(rng.uniform(args.soft_sigma_min, args.soft_sigma_max))
   boundary_sigma = float(rng.uniform(args.boundary_sigma_min, args.boundary_sigma_max))


   synthetic, soft = blend(healthy, placed_mask, adapted_patch, soft_sigma)
   synthetic = smooth_boundary_only(synthetic, placed_mask, sigma=boundary_sigma)


   validate_synthetic_output(healthy, synthetic, placed_mask)


   diff = np.abs(synthetic - healthy).astype(np.float32)


   stem = strip_suffixes(h_path.name)
   save_nifti(healthy, nii, out_dirs["healthy"] / f"{stem}.nii.gz")
   save_nifti(synthetic, nii, out_dirs["synthetic"] / f"{stem}_synthetic.nii.gz")
   save_nifti(placed_mask.astype(np.float32), nii, out_dirs["mask"] / f"{stem}_mask.nii.gz")
   save_nifti(diff, nii, out_dirs["diff"] / f"{stem}_diff.nii.gz")
   save_nifti(soft.astype(np.float32), nii, out_dirs["soft"] / f"{stem}_soft.nii.gz")


   save_json(
       {
           "healthy_file": str(h_path),
           "library_item": str(lib_path),
           "soft_sigma": soft_sigma,
           "boundary_sigma": boundary_sigma,
           **meta,
       },
       out_dirs["meta"] / f"{stem}_meta.json",
   )


   return True




# -----------------------------------------------------------------------------
# Main
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
       "soft": args.output_dir / "soft",
       "meta": args.output_dir / "metadata",
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
# Args
# -----------------------------------------------------------------------------




def parse_args():
   p = argparse.ArgumentParser()
   p.add_argument("--healthy-dir", type=Path, required=True)
   p.add_argument("--library-dir", type=Path, required=True)
   p.add_argument("--output-dir", type=Path, required=True)
   p.add_argument("--name-filter", type=str, default=None)
   p.add_argument("--seed", type=int, default=42)


   # CarveMix geometric randomization
   p.add_argument("--scale-min", type=float, default=0.90)
   p.add_argument("--scale-max", type=float, default=1.10)
   p.add_argument("--rot-min", type=float, default=-10.0)
   p.add_argument("--rot-max", type=float, default=10.0)
   p.add_argument("--allow-flip", action="store_true")
   p.add_argument("--max-placement-tries", type=int, default=100)
   p.add_argument("--min-inside-ratio", type=float, default=0.98)


   # Guizard-style local-context adaptation
   p.add_argument("--context-ring-min", type=int, default=3)
   p.add_argument("--context-ring-max", type=int, default=6)


   p.add_argument("--patch-sigma-min", type=float, default=0.6)
   p.add_argument("--patch-sigma-max", type=float, default=1.8)


   p.add_argument("--fill-sigma-high-min", type=float, default=2.0)
   p.add_argument("--fill-sigma-high-max", type=float, default=4.0)
   p.add_argument("--fill-sigma-mid-min", type=float, default=1.0)
   p.add_argument("--fill-sigma-mid-max", type=float, default=2.0)
   p.add_argument("--fill-sigma-low-min", type=float, default=0.4)
   p.add_argument("--fill-sigma-low-max", type=float, default=1.0)


   # Mix between source-derived lesion and context carrier field
   p.add_argument("--context-mix-min", type=float, default=0.45)
   p.add_argument("--context-mix-max", type=float, default=0.75)


   # T1 tumor appearance model:
   # target_mean = context_mean - alpha * context_std
   # target_std  = beta * context_std
   p.add_argument("--t1-dark-alpha-min", type=float, default=0.60)
   p.add_argument("--t1-dark-alpha-max", type=float, default=1.60)
   p.add_argument("--t1-std-beta-min", type=float, default=0.15)
   p.add_argument("--t1-std-beta-max", type=float, default=0.45)
   p.add_argument("--t1-noise-frac-min", type=float, default=0.01)
   p.add_argument("--t1-noise-frac-max", type=float, default=0.06)


   # Optional mild central darkening
   p.add_argument("--enable-center-darkening", action="store_true")
   p.add_argument("--center-dark-strength-min", type=float, default=0.15)
   p.add_argument("--center-dark-strength-max", type=float, default=0.5)
   p.add_argument("--center-dark-gamma-min", type=float, default=0.8)
   p.add_argument("--center-dark-gamma-max", type=float, default=1.4)


   # Boundary randomization
   p.add_argument("--soft-sigma-min", type=float, default=1.8)
   p.add_argument("--soft-sigma-max", type=float, default=3.0)
   p.add_argument("--boundary-sigma-min", type=float, default=0.8)
   p.add_argument("--boundary-sigma-max", type=float, default=1.4)


   return p.parse_args()




if __name__ == "__main__":
   main()



