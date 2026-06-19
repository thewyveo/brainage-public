#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import torch

from utils.demo_utils import prepare_image, get_feature
from utils.misc import viewVolume, make_dir


r"""
Example flat folder:

PYTHONPATH=. python3.10 scripts/infer_folder.py \
  --input-dir "/path/to/images" \
  --checkpoint "/path/to/brain_id_pretrained.pth" \
  --out-dir "/path/to/outputs" \
  --device cpu \
  --image-filter t1 \
  --min-ixi-id 464

Example recursive folder:

PYTHONPATH=. python3.10 scripts/infer_folder.py \
  --input-dir "/path/to/images" \
  --checkpoint "/path/to/brain_id_pretrained.pth" \
  --out-dir "/path/to/outputs" \
  --device cuda:0 \
  --image-filter t1n \
  --recursive \
  --min-ixi-id 464
"""


def strip_nii(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def is_nifti(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith(".nii.gz") or lower.endswith(".nii")


def safe_name(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out).strip("_")


def extract_ixi_number(path: Path) -> int | None:
    """
    Extracts numeric IXI ID from filename or relative path.

    Examples:
    IXI464-HH-2391-T1.nii.gz -> 464
    some/folder/IXI002-Guys-0828-T1/file.nii.gz -> 2
    """
    m = re.search(r"IXI(\d+)", str(path))
    if not m:
        return None
    return int(m.group(1))


def output_stem_for(path: Path, root: Path, recursive: bool) -> str:
    if not recursive:
        return safe_name(strip_nii(path.name))

    rel = path.relative_to(root)
    parts = list(rel.parts)
    parts[-1] = strip_nii(parts[-1])

    return safe_name("__".join(parts))


def find_niftis(
    input_dir: Path,
    image_filter: str | None,
    recursive: bool,
    min_ixi_id: int | None,
) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")

    files = []

    for path in iterator:
        if not path.is_file():
            continue

        if not is_nifti(path):
            continue

        rel_text = str(path.relative_to(input_dir))

        if image_filter is not None and image_filter.strip() != "":
            if image_filter.lower() not in rel_text.lower():
                continue

        if min_ixi_id is not None:
            ixi_num = extract_ixi_number(path)
            if ixi_num is None:
                continue
            if ixi_num < min_ixi_id:
                continue

        files.append(path)

    return sorted(files)


def infer_one(
    input_path: Path,
    output_stem: str,
    checkpoint: str,
    out_dir: Path,
    device: str,
) -> None:
    print("=" * 80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_stem}_brainid_recon")
    print("=" * 80)

    print("Preparing image...")
    im, aff = prepare_image(str(input_path), device=device)
    print("Image prepared")

    print("Extracting features...")
    with torch.inference_mode():
        outputs = get_feature(
            im,
            checkpoint,
            feature_only=False,
            device=device,
        )
    print("Features extracted")

    recon = outputs["image"]
    print("Reconstructed")

    viewVolume(
        recon,
        aff,
        names=[f"{output_stem}_brainid_recon"],
        save_dir=out_dir,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Batch Brain-ID inference over a folder of NIfTI images."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Folder containing input NIfTI images.",
    )

    parser.add_argument(
        "--checkpoint",
        default="assets/brain_id_pretrained.pth",
        help="Path to Brain-ID checkpoint.",
    )

    parser.add_argument(
        "--out-dir",
        default="outs/reconstruction",
        help="Output folder.",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device, e.g. cuda:0, cpu.",
    )

    parser.add_argument(
        "--image-filter",
        default=None,
        help="Optional substring filter. Example: T1, t1n, brain_n4_rigid.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subfolders recursively. Output names include relative subfolder names.",
    )

    parser.add_argument(
        "--min-ixi-id",
        type=int,
        default=None,
        help="Only process files with IXI ID >= this value. Example: --min-ixi-id 464.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to process after filtering.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing reconstructions. Default: skip existing outputs.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    out_dir = Path(make_dir(args.out_dir))

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    img_files = find_niftis(
        input_dir=input_dir,
        image_filter=args.image_filter,
        recursive=args.recursive,
        min_ixi_id=args.min_ixi_id,
    )

    if args.limit is not None:
        img_files = img_files[: args.limit]

    print(f"Found images: {len(img_files)}")
    print(f"Input dir:     {input_dir}")
    print(f"Output dir:    {out_dir}")
    print(f"Recursive:     {args.recursive}")
    print(f"Image filter:  {args.image_filter}")
    print(f"Min IXI ID:    {args.min_ixi_id}")
    print(f"Device:        {args.device}")
    print()

    if len(img_files) == 0:
        print("No images found.")
        return

    done = 0
    skipped = 0
    failed = 0

    for input_path in img_files:
        output_stem = output_stem_for(
            path=input_path,
            root=input_dir,
            recursive=args.recursive,
        )

        expected_out = out_dir / f"{output_stem}_brainid_recon.nii.gz"

        if expected_out.exists() and not args.overwrite:
            print(f"SKIP existing: {expected_out}")
            skipped += 1
            continue

        try:
            infer_one(
                input_path=input_path,
                output_stem=output_stem,
                checkpoint=args.checkpoint,
                out_dir=out_dir,
                device=args.device,
            )
            done += 1

        except Exception as e:
            print(f"FAILED: {input_path}")
            print(f"Reason: {repr(e)}")
            failed += 1

    print()
    print("=" * 80)
    print("Finished Brain-ID batch inference.")
    print(f"Done:    {done}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {failed}")
    print("=" * 80)


if __name__ == "__main__":
    main()
