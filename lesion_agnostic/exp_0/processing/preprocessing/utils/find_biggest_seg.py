#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import argparse
import nibabel as nib
import numpy as np

r"""
py -3.10 exp_0\processing\preprocessing\utils\find_biggest_seg.py `
    --seg-root data\library\BraTS_Masks
"""


def is_seg_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith("-seg.nii") or name.endswith("-seg.nii.gz")




def count_tumor_voxels(seg_path: Path) -> int:
    img = nib.load(str(seg_path))
    data = img.get_fdata()


    # Any label > 0 is considered tumor / abnormal region
    voxel_count = int(np.count_nonzero(data > 0))
    return voxel_count




def find_biggest_mask(seg_root: Path) -> None:
    seg_files = sorted([p for p in seg_root.rglob("*") if p.is_file() and is_seg_file(p)])


    if not seg_files:
        print("No segmentation files found.")
        return


    biggest_path = None
    biggest_voxels = -1


    for i, seg_path in enumerate(seg_files, start=1):
        voxels = count_tumor_voxels(seg_path)
        print(f"[{i}/{len(seg_files)}] {seg_path.name}: {voxels} tumor voxels")


        if voxels > biggest_voxels:
            biggest_voxels = voxels
            biggest_path = seg_path


    print("\nDone.")
    print(f"Biggest mask: {biggest_path}")
    print(f"Tumor voxel count: {biggest_voxels}")




def main():
    parser = argparse.ArgumentParser(
        description="Find the biggest BraTS segmentation mask by tumor voxel count."
    )
    parser.add_argument(
        "--seg-root",
        type=Path,
        required=True,
        help="Folder containing BraTS seg files."
    )
    args = parser.parse_args()


    if not args.seg_root.exists():
        raise FileNotFoundError(f"Seg root does not exist: {args.seg_root}")


    find_biggest_mask(args.seg_root)




if __name__ == "__main__":
    main()

