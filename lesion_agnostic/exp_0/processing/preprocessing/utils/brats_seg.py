#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import shutil
import argparse

r"""
py -3.10 exp_0\processing\preprocessing\utils\brats_seg.py `
    --input-root data\training_data1_v2 `
    --output-dir data\library\BraTS_Masks
"""


def is_seg_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith("-seg.nii") or name.endswith("-seg.nii.gz")




def make_unique_destination(dst_dir: Path, filename: str) -> Path:
    dst = dst_dir / filename
    if not dst.exists():
        return dst


    stem = filename
    suffix = ""


    if filename.endswith(".nii.gz"):
        stem = filename[:-7]
        suffix = ".nii.gz"
    else:
        stem = Path(filename).stem
        suffix = Path(filename).suffix


    counter = 1
    while True:
        candidate = dst_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1




def extract_seg_masks(input_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


    found = 0
    copied = 0


    for path in input_root.rglob("*"):
        if not path.is_file():
            continue


        if is_seg_file(path):
            found += 1
            dst = make_unique_destination(output_dir, path.name)
            shutil.copy2(path, dst)
            copied += 1
            print(f"[{copied}] Copied: {path} -> {dst}")


    print("\nDone.")
    print(f"Segmentation files found: {found}")
    print(f"Segmentation files copied: {copied}")
    print(f"Output folder: {output_dir}")




def main():
    parser = argparse.ArgumentParser(
        description="Extract only BraTS segmentation masks into a single flat folder."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root folder containing BraTS case folders."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder where all seg files will be copied."
    )
    args = parser.parse_args()


    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")


    extract_seg_masks(args.input_root, args.output_dir)




if __name__ == "__main__":
    main()

