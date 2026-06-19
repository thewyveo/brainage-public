#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import nibabel as nib
import numpy as np
import pandas as pd

r"""
py -3.10 exp_0\processing\preprocessing\utils\find_biggest_seg2.py `
    --seg-root data\library\BraTS_Masks `
    --output-dir exp_0\processing\preprocessing\utils\insights\BraTS_Mask_Analysis_2 `
    --topk 20
"""

def is_seg_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith("-seg.nii") or name.endswith("-seg.nii.gz")


def strip_nii(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def get_bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    return mins, maxs


def bbox_size(mins, maxs):
    return (
        int(maxs[0] - mins[0] + 1),
        int(maxs[1] - mins[1] + 1),
        int(maxs[2] - mins[2] + 1),
    )


def analyze_seg(path: Path):
    img = nib.load(str(path))
    data = img.get_fdata()

    mask = data > 0
    voxel_count = int(np.count_nonzero(mask))

    bbox = get_bbox(mask)
    if bbox is None:
        return None

    mins, maxs = bbox
    bx, by, bz = bbox_size(mins, maxs)

    return {
        "case": strip_nii(path.name),
        "path": str(path),
        "voxels": voxel_count,
        "bbox_x": bx,
        "bbox_y": by,
        "bbox_z": bz,
        "fits_96": (bx <= 96 and by <= 96 and bz <= 96),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Find largest BraTS tumor that fits GliGAN 96³ constraint"
    )
    parser.add_argument("--seg-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    seg_files = sorted([p for p in args.seg_root.rglob("*") if is_seg_file(p)])

    rows = []
    for i, path in enumerate(seg_files, 1):
        print(f"[{i}/{len(seg_files)}] {path.name}")
        res = analyze_seg(path)
        if res:
            rows.append(res)

    df = pd.DataFrame(rows)

    # Save full table
    df.to_csv(args.output_dir / "all_tumors.csv", index=False)

    # Filter valid ones
    valid_df = df[df["fits_96"]].copy()

    if len(valid_df) == 0:
        print("\n❌ No tumors fit inside 96³")
        return

    # Sort by voxel count
    valid_df = valid_df.sort_values("voxels", ascending=False).reset_index(drop=True)

    # Save valid ones
    valid_df.to_csv(args.output_dir / "valid_tumors_96.csv", index=False)

    # Best one
    best = valid_df.iloc[0]

    print("\n✅ BEST VALID TUMOR:")
    print(f"Path: {best['path']}")
    print(f"Voxels: {best['voxels']}")
    print(f"BBox: ({best['bbox_x']}, {best['bbox_y']}, {best['bbox_z']})")

    # Save top-K
    topk = valid_df.head(args.topk)
    topk.to_csv(args.output_dir / f"top_{args.topk}_valid_tumors.csv", index=False)

    print(f"\nTop {args.topk} saved.")


if __name__ == "__main__":
    main()
