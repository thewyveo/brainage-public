#!/usr/bin/env python3
import argparse
from pathlib import Path


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt




r"""
Linux/Snellius:


py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure_gridcombined.py `
  --base-dir exp_0\processing\preprocessing\utils\make_figure\makefig1_synthmorph `
  --slice-index 100 `
  --output exp_0\processing\preprocessing\utils\make_figure\fig1_all_methods.png
"""




def load_nifti(path):
    img = nib.load(str(path))
    data = img.get_fdata()
    data = np.nan_to_num(data)
    return data




def normalize_image(x, is_mask=False):
    x = np.asarray(x, dtype=np.float32)


    if is_mask:
        return (x > 0).astype(np.float32)


    nonzero = x[x != 0]
    if nonzero.size == 0:
        return x


    lo, hi = np.percentile(nonzero, [1, 99])
    if hi <= lo:
        return np.zeros_like(x)


    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    return x




def extract_slice(volume, axis, slice_index=None):
    if slice_index is None:
        slice_index = volume.shape[axis] // 2


    if axis == 0:
        sl = volume[slice_index, :, :]
    elif axis == 1:
        sl = volume[:, slice_index, :]
    elif axis == 2:
        sl = volume[:, :, slice_index]
    else:
        raise ValueError("axis must be 0, 1, or 2")


    return np.rot90(sl)




def load_slice(path, axis, slice_index, is_mask=False):
    vol = load_nifti(path)
    sl = extract_slice(vol, axis=axis, slice_index=slice_index)
    return normalize_image(sl, is_mask=is_mask)




def plot_img(ax, base_dir, fname, title, axis, slice_index, is_mask=False, title_fontsize=10):
    img = load_slice(base_dir / fname, axis, slice_index, is_mask=is_mask)
    ax.imshow(img, cmap="gray")
    ax.set_title(title, fontsize=title_fontsize, pad=2)
    ax.axis("off")




def make_big_figure(base_dir, output_path, axis=2, slice_index=100, dpi=300):
    base_dir = Path(base_dir)
    output_path = Path(output_path)


    rows = [
        [
            ("healthy_aligned.nii.gz", "Healthy T1", False),
            ("BraTS-GLI-02608-102_seg_rigid.nii.gz", "Tumor Mask", True),
        ],
        [
            ("cm_aligned.nii.gz", "CM T1", False),
            ("cmlit_aligned.nii.gz", "CM + LIT", False),
            ("cmbid_aligned.nii.gz", "CM + BID", False),
            ("cmusb_aligned_axis0flip.nii.gz", "CM + USB", False),
        ],
        [
            ("gli_aligned.nii.gz", "GLI T1", False),
            ("glilit_aligned.nii.gz", "GLI + LIT", False),
            ("glibid_aligned.nii.gz", "GLI + BID", False),
            ("gliusb_aligned_axis0flip.nii.gz", "GLI + USB", False),
        ],
        [
            ("usb_aligned_axis0flip.nii.gz", "USB T1", False),
            ("usblit_aligned_axis0flip.nii.gz", "USB + LIT", False),
            ("usbbid_aligned_axis0flip.nii.gz", "USB + BID", False),
            ("usbusb_aligned_axis0flip.nii.gz", "USB + USB", False),
        ],
    ]


    fig = plt.figure(figsize=(4, 6))


    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[1, 1, 1, 1],
        hspace=0.15,
    )


    # Top row: exactly 2 panels, centered
    top_gs = outer[0].subgridspec(
        1,
        2,
        wspace=0.005,
    )


    for col, (fname, title, is_mask) in enumerate(rows[0]):
        ax = fig.add_subplot(top_gs[0, col])
        plot_img(ax, base_dir, fname, title, axis, slice_index, is_mask, title_fontsize=10)


    # Rows 2-4: exactly 4 panels each
    for row_idx in range(1, 4):
        row_gs = outer[row_idx].subgridspec(
            1,
            4,
            wspace=0.01,
        )


        for col, (fname, title, is_mask) in enumerate(rows[row_idx]):
            ax = fig.add_subplot(row_gs[0, col])
            plot_img(ax, base_dir, fname, title, axis, slice_index, is_mask, title_fontsize=9.5)


    output_path.parent.mkdir(parents=True, exist_ok=True)


    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.015,
    )
    plt.close()


    print(f"Saved figure to: {output_path}")




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--axis", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--slice-index", type=int, default=100)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()




def main():
    args = parse_args()
    make_big_figure(
        base_dir=args.base_dir,
        output_path=args.output,
        axis=args.axis,
        slice_index=args.slice_index,
        dpi=args.dpi,
    )




if __name__ == "__main__":
    main()


