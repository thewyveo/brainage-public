#!/usr/bin/env python3

import nibabel as nib
import numpy as np
from pathlib import Path

base_dir = Path("/home/kozdemir/makefig1_synthmorph")

usb_files = [
    "usb_aligned.nii.gz",
    "usbbid_aligned.nii.gz",
    "usblit_aligned.nii.gz",
    "gliusb_aligned.nii.gz",
    "cmusb_aligned.nii.gz",
    "usbusb_aligned.nii.gz",
]

for fname in usb_files:
    inp = base_dir / fname

    out = base_dir / fname.replace(
        "_aligned.nii.gz",
        "_aligned_axis0flip.nii.gz"
    )

    print(f"[PROCESS] {inp}")

    img = nib.load(str(inp))
    data = img.get_fdata().astype(np.float32)

    # fix USB left-right flip
    data = np.flip(data, axis=0)

    fixed = nib.Nifti1Image(
        data,
        img.affine,
        img.header.copy()
    )

    nib.save(fixed, str(out))

    print(f"[SAVED] {out}")

print("DONE")
