#!/usr/bin/env python3
import nibabel as nib
import numpy as np
from pathlib import Path

inp = Path("/home/kozdemir/makefigs_synthmorph_aligned/y_p_IXI013-HH-1212-T1__BraTS-GLI-02608-102_IXI013-HH-1212-T1__BraTS-GLI-02608-102_aligned.nii.gz")
out = Path("/home/kozdemir/makefigs_synthmorph_aligned/y_p_IXI013_USB_aligned_axis0flip.nii.gz")

img = nib.load(str(inp))
data = img.get_fdata().astype(np.float32)

data = np.flip(data, axis=0)

fixed = nib.Nifti1Image(data, img.affine, img.header.copy())
nib.save(fixed, str(out))

print("saved:", out)
