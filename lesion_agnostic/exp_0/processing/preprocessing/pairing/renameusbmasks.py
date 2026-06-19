#!/usr/bin/env python3

from pathlib import Path
import argparse
import re
import shutil


def strip_nii(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--usb-mri-dir", required=True, type=Path)
    p.add_argument("--brats-mask-dir", required=True, type=Path)
    p.add_argument("--out-mask-dir", required=True, type=Path)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    args.out_mask_dir.mkdir(parents=True, exist_ok=True)

    mris = sorted(args.usb_mri_dir.glob("*.nii.gz"))

    made = 0
    missing = 0

    for mri in mris:
        stem = strip_nii(mri.name)

        brats = re.search(r"BraTS-GLI-\d{5}-\d{3}", stem)
        if not brats:
            print(f"NO BRATS ID: {mri.name}")
            missing += 1
            continue

        brats_id = brats.group(0)

        src_mask = args.brats_mask_dir / f"{brats_id}_seg_rigid.nii.gz"

        # mask gets exact same basename as USB MRI
        dst_mask = args.out_mask_dir / f"{stem}.nii.gz"

        if not src_mask.exists():
            print(f"MISSING SOURCE MASK: {src_mask}")
            missing += 1
            continue

        if dst_mask.exists() and not args.overwrite:
            print(f"EXISTS, skipping: {dst_mask.name}")
            continue

        shutil.copy2(src_mask, dst_mask)
        print(f"{src_mask.name} -> {dst_mask.name}")
        made += 1

    print("=" * 60)
    print(f"USB MRIs found: {len(mris)}")
    print(f"Masks copied:   {made}")
    print(f"Missing/errors: {missing}")
    print(f"Output folder:  {args.out_mask_dir}")


if __name__ == "__main__":
    main()
