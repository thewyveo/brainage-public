#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BraTS T1 + SEG preprocessing:
1. N4 bias correction on BraTS T1
2. ANTs rigid registration of N4-corrected T1 to MNI
3. Apply same rigid transform to BraTS segmentation with nearest-neighbor

Minimal preprocessing for:
- CarveMix
- synthetic lesion generation
- FastSurfer-LIT aligned masks

Example:
python brats_t1_seg_n4_rigid.py \
  --brats-img-dir /home/kozdemir/BraTS_T1 \
  --brats-seg-dir /home/kozdemir/BraTS_SEG \
  --out-img-dir /home/kozdemir/BraTS_T1_n4_rigid \
  --out-seg-dir /home/kozdemir/BraTS_SEG_rigid \
  --mni /home/kozdemir/brainage/lesion_agnostic/data/MNI152_T1_1mm_Brain.nii \
  --image-filter t1n \
  --seg-filter seg \
  --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import ants


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def strip_nii(name: str) -> str:
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def normalize_case_key(path: Path) -> str:
    stem = strip_nii(path.name)

    suffixes = [
        "_seg", "-seg",
        "_t1n", "-t1n",
        "_t1", "-t1",
        "_t1ce", "-t1ce",
        "_t1c", "-t1c",
        "_flair", "-flair",
        "_t2", "-t2",
    ]

    changed = True
    while changed:
        changed = False
        lower = stem.lower()

        for suffix in suffixes:
            if lower.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
                break

    return stem


def find_niftis(root: Path, name_filter: str | None) -> list[Path]:
    files: list[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        lower = path.name.lower()

        if not (lower.endswith(".nii.gz") or lower.endswith(".nii")):
            continue

        if name_filter is not None and name_filter.lower() not in lower:
            continue

        files.append(path)

    return sorted(files)


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------

def process_case(
    img_path: str,
    seg_path: str,
    out_img: str,
    out_seg: str,
    mni_path: str,
) -> tuple[str, bool, str]:

    case_key = normalize_case_key(Path(img_path))

    try:
        fixed = ants.image_read(mni_path)

        moving_img = ants.image_read(img_path)
        moving_seg = ants.image_read(seg_path)

        # ---------------------------------------------------------------------
        # 1. N4 bias correction
        # ---------------------------------------------------------------------

        moving_img_n4 = ants.n4_bias_field_correction(moving_img)

        # ---------------------------------------------------------------------
        # 2. Rigid registration
        # ---------------------------------------------------------------------

        reg = ants.registration(
            fixed=fixed,
            moving=moving_img_n4,
            type_of_transform="Rigid",
        )

        warped_img = reg["warpedmovout"]

        # ---------------------------------------------------------------------
        # 3. Apply SAME transform to seg
        # ---------------------------------------------------------------------

        warped_seg = ants.apply_transforms(
            fixed=fixed,
            moving=moving_seg,
            transformlist=reg["fwdtransforms"],
            interpolator="nearestNeighbor",
        )

        ants.image_write(warped_img, out_img)
        ants.image_write(warped_seg, out_seg)

        return case_key, True, "OK"

    except Exception as e:
        return case_key, False, str(e)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description="N4 + rigid-register BraTS T1 to MNI and apply same transform to SEG."
    )

    parser.add_argument("--brats-img-dir", type=Path, required=True)
    parser.add_argument("--brats-seg-dir", type=Path, required=True)

    parser.add_argument("--out-img-dir", type=Path, required=True)
    parser.add_argument("--out-seg-dir", type=Path, required=True)

    parser.add_argument("--mni", type=Path, required=True)

    parser.add_argument("--image-filter", type=str, default="t1n")
    parser.add_argument("--seg-filter", type=str, default="seg")

    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    args.out_img_dir.mkdir(parents=True, exist_ok=True)
    args.out_seg_dir.mkdir(parents=True, exist_ok=True)

    img_files = find_niftis(args.brats_img_dir, args.image_filter)
    seg_files = find_niftis(args.brats_seg_dir, args.seg_filter)

    if args.limit is not None:
        img_files = img_files[: args.limit]

    seg_map = {
        normalize_case_key(seg): seg
        for seg in seg_files
    }

    print(f"Found T1 images: {len(img_files)}")
    print(f"Found SEG masks: {len(seg_files)}")

    futures = []

    done = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:

        for img_path in img_files:

            case_key = normalize_case_key(img_path)

            seg_path = seg_map.get(case_key)

            if seg_path is None:
                print(f"SKIP no matching seg: {img_path.name}")
                skipped += 1
                continue

            out_img = args.out_img_dir / f"{case_key}_t1n_n4_rigid.nii.gz"
            out_seg = args.out_seg_dir / f"{case_key}_seg_rigid.nii.gz"

            if out_img.exists() and out_seg.exists():
                print(f"EXISTS {case_key}")
                done += 1
                continue

            futures.append(
                ex.submit(
                    process_case,
                    str(img_path),
                    str(seg_path),
                    str(out_img),
                    str(out_seg),
                    str(args.mni),
                )
            )

        for fut in as_completed(futures):

            case_key, ok, msg = fut.result()

            if ok:
                print(f"DONE   {case_key}")
                done += 1
            else:
                print(f"FAILED {case_key}: {msg}")
                failed += 1

    print()
    print("Finished.")
    print(f"Done:    {done}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {failed}")


if __name__ == "__main__":
    main()
