#!/usr/bin/env python3

from pathlib import Path
import argparse
import tempfile
import shutil

import ants
import nibabel as nib
import numpy as np


def strip_nii(name):
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def make_flipped_image(inp_path, out_path):
    img = nib.load(str(inp_path))
    data = img.get_fdata().astype(np.float32)

    # Left-right flip
    flipped = np.flip(data, axis=0).copy()

    out = nib.Nifti1Image(flipped, img.affine, img.header)
    nib.save(out, str(out_path))


def register_flip_to_original(orig_path, flipped_path, output_path):
    fixed = ants.image_read(str(orig_path))
    moving = ants.image_read(str(flipped_path))

    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="SyN"
    )

    ants.image_write(reg["warpedmovout"], str(output_path))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(args.input_dir.glob("*.nii.gz")))

    print(f"Found {len(files)} files")

    for i, f in enumerate(files, start=1):
        stem = strip_nii(f.name)

        print(f"[{i}/{len(files)}] {stem}")

        case_dir = args.output_dir / stem
        case_dir.mkdir(parents=True, exist_ok=True)

        input_out = case_dir / "input.nii.gz"
        flip_reg_out = case_dir / "input_flip_reg2orig.nii.gz"

        shutil.copy2(f, input_out)

        with tempfile.TemporaryDirectory(prefix="una_flip_") as tmpdir:
            tmpdir = Path(tmpdir)

            flipped_tmp = tmpdir / "flipped.nii.gz"

            print("  flipping")
            make_flipped_image(input_out, flipped_tmp)

            print("  registering flipped back to original")
            register_flip_to_original(
                input_out,
                flipped_tmp,
                flip_reg_out
            )

    print("DONE")


if __name__ == "__main__":
    main()
