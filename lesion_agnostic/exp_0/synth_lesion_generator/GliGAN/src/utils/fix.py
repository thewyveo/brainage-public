#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
from pathlib import Path


import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom




def load_mask(path):
    arr = nib.load(str(path)).get_fdata()
    return (arr > 0).astype(np.uint8)




def crop_bbox(mask):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None


    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1


    return mask[
        mins[0]:maxs[0],
        mins[1]:maxs[1],
        mins[2]:maxs[2],
    ]




def resize_mask(mask, target_shape=(64, 64, 64)):
    factors = [
        target_shape[i] / mask.shape[i]
        for i in range(3)
    ]
    resized = zoom(mask.astype(float), factors, order=0)
    return (resized > 0).astype(np.uint8)




def dice(a, b):
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    if denom == 0:
        return 0.0
    return 2 * inter / denom




def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return inter / union




def find_niftis(folder):
    return sorted([
        p for p in folder.rglob("*")
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz"))
    ])




def main():
    parser = argparse.ArgumentParser()


    parser.add_argument("--used-mask-dir", required=True, type=Path,
                        help="Folder containing GliGAN output subject folders with synthetic_seg.nii.gz")
    parser.add_argument("--library-mask-dir", required=True, type=Path,
                        help="Folder containing original BraTS/input masks")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)


    args = parser.parse_args()


    used_masks = sorted(args.used_mask_dir.rglob("synthetic_seg.nii.gz"))
    library_masks = find_niftis(args.library_mask_dir)


    print(f"Used GliGAN masks found: {len(used_masks)}")
    print(f"Library masks found: {len(library_masks)}")


    library_cache = []


    for lib_path in library_masks:
        try:
            m = load_mask(lib_path)
            c = crop_bbox(m)
            if c is None:
                continue
            r = resize_mask(c)
            library_cache.append((lib_path, r, int(m.sum())))
        except Exception as e:
            print(f"Skipping library mask {lib_path.name}: {e}")


    rows = []


    for used_path in used_masks:
        print(f"Matching: {used_path}")


        try:
            used = load_mask(used_path)
            used_crop = crop_bbox(used)


            if used_crop is None:
                rows.append({
                    "used_mask": str(used_path),
                    "rank": 1,
                    "matched_library_mask": None,
                    "dice": 0,
                    "iou": 0,
                    "used_volume": 0,
                    "library_volume": None,
                    "volume_ratio": None,
                })
                continue


            used_resized = resize_mask(used_crop)
            used_volume = int(used.sum())


            scores = []


            for lib_path, lib_resized, lib_volume in library_cache:
                d = dice(used_resized, lib_resized)
                j = iou(used_resized, lib_resized)


                volume_ratio = used_volume / lib_volume if lib_volume > 0 else np.nan


                scores.append({
                    "used_mask": str(used_path),
                    "used_case": used_path.parent.name,
                    "rank": None,
                    "matched_library_mask": str(lib_path),
                    "matched_library_name": lib_path.name,
                    "dice": d,
                    "iou": j,
                    "used_volume": used_volume,
                    "library_volume": lib_volume,
                    "volume_ratio": volume_ratio,
                })


            scores = sorted(scores, key=lambda x: x["dice"], reverse=True)


            for rank, row in enumerate(scores[:args.top_k], start=1):
                row["rank"] = rank
                rows.append(row)


        except Exception as e:
            print(f"FAILED {used_path}: {e}")


    out = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)


    print(f"\nSaved matches to: {args.output_csv}")




if __name__ == "__main__":
    main()
