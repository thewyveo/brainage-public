#!/usr/bin/env python3
import argparse
from pathlib import Path


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt




r"""
py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\healthy.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\mask.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\carvemix.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\gli.nii.gz `
  --labels `
    "Ground-Truth Healthy" `
    "BraTS Tumor Mask" `
    "CarveMix Output" `
    "GliGAN output" `
  --mask-indices 1 `
  --axis 2 `
  --slice-index 80 `
  --output exp_0\processing\preprocessing\utils\make_figure\combined_synthetic_generators.png




py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\figure_inpainting_n4_rigid\healthy_n4_rigid.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\correctedinput\gli.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\correctedinput\GLILIT.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\correctedinput\GLIBID.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\correctedinput\GLIUSB.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\correctedinput\gli_sr.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\correctedinput\gli_una.nii.gz `
  --labels `
    "Ground-Truth Healthy" `
    "GliGAN-Generated Tumored T1" `
    "GliGAN T1 + FastSurfer-LIT" `
    "GliGAN T1 + Brain-ID" `
    "GliGAN T1 + USB p2h" `
    "GliGAN T1 + SynthSR" `
    "GliGAN T1 + UNA" `
  --axis 2 `
  --slice-index 80 `
  --output exp_0\processing\preprocessing\utils\make_figure\combined_inpainters_processed.png




py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\healthy_t1.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\synthetic_seg.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\gli_rerun.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\cm_rerun.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\usb_rerun.nii.gz `
  --labels `
    "Healthy T1" `
    "Input Tumor Mask" `
    "GliGAN T1" `
    "CarveMix T1" `
    "USB h2p T1" `
  --output exp_0\processing\preprocessing\utils\make_figure\exp0_figure.png




py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\healthy_t1.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\tumor_mask.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\cm_rerun.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\synthetic_seg.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\gli_rerun.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\usb_mask.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\usb_rerun.nii.gz `
  --labels `
    "Healthy T1" `
    "Input Tumor Mask" `
    "CarveMix T1" `
    "Input Tumor Mask" `
    "GliGAN T1" `
    "Input Tumor Mask" `
    "USB h2p T1" `
  --output exp_0\processing\preprocessing\utils\make_figure\exp0_figure.png




py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\healthy_t1.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\cm_mask.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\IXI012-HH-1211-T1_brain_n4_rigid__BraTS-GLI-02217-102_carvemix.nii.gz `
  --labels `
    "Healthy T1" `
    "Input Tumor Mask" `
    "CarveMix T1" `
  --mask-indices 1 `
  --output exp_0\processing\preprocessing\utils\make_figure\exp0_figure_CM.png




py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\healthy_t1.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\synthetic_seg.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\synthetic_t1.nii.gz `
  --labels `
    "Healthy T1" `
    "Input Tumor Mask" `
    "GliGAN T1" `
  --mask-indices 1 `
  --output exp_0\processing\preprocessing\utils\make_figure\exp0_figure_GLI.png




py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\y_h.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\mask_usb.nii `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\y_p.nii.gz `
  --labels `
    "Healthy T1" `
    "Input Tumor Mask" `
    "USB h2p T1" `
  --mask-indices 1 `
 --output exp_0\processing\preprocessing\utils\make_figure\exp0_figure_USB.png




py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure2.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\makefigs_synthmorph_aligned\IXI013-HH-1212-T1_preprocessed_aligned.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefigs_synthmorph_aligned\BraTS-GLI-02608-102_seg_rigid.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefigs_synthmorph_aligned\IXI013-HH-1212-T1_brain_n4_rigid__BraTS-GLI-02608-102_carvemix_aligned.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefigs_synthmorph_aligned\synthetic_t1_aligned.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefigs_synthmorph_aligned\y_p_IXI013_USB_aligned_axis0flip.nii.gz `
  --labels `
    "Healthy T1" `
    "BraTS Tumor Mask" `
    "CarveMix T1" `
    "GliGAN T1" `
    "USB h2p T1" `
  --mask-indices 1 `
  --output exp_0\processing\preprocessing\utils\make_figure\finalcorrectedexp0_figure.png `
  --slice-index 100

py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure2.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\makefig1_synthmorph\healthy_aligned.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefig1_synthmorph\cm_aligned.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefig1_synthmorph\cmlit_aligned.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefig1_synthmorph\cmbid_aligned.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefig1_synthmorph\cmusb_aligned_axis0flip.nii.gz `
  --labels `
    "Healthy T1" `
    "CarveMix T1" `
    "CarveMix + FastSurfer-LIT" `
    "CarveMix + Brain-ID" `
    "CarveMix + USB p2h" `
  --slice-index 100 `
  --output exp_0\processing\preprocessing\utils\make_figure\fig1_cm.png

py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure2.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\makefig1_synthmorph\healthy_aligned.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefig1_synthmorph\gli_aligned.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefig1_synthmorph\glilit_aligned.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefig1_synthmorph\glibid_aligned.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefig1_synthmorph\gliusb_aligned_axis0flip.nii.gz `
  --labels `
    "Healthy T1" `
    "GliGAN T1" `
    "GliGAN + FastSurfer-LIT" `
    "GliGAN + Brain-ID" `
    "GliGAN + USB p2h" `
  --slice-index 100 `
  --output exp_0\processing\preprocessing\utils\make_figure\fig1_gli.png

py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure2.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\makefig1_synthmorph\healthy_aligned.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefig1_synthmorph\usb_aligned_axis0flip.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefig1_synthmorph\usblit_aligned_axis0flip.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefig1_synthmorph\usbbid_aligned_axis0flip.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefig1_synthmorph\usbusb_aligned_axis0flip.nii.gz `
  --labels `
    "Healthy T1" `
    "USB h2p T1" `
    "USB h2p + FastSurfer-LIT" `
    "USB h2p + Brain-ID" `
    "USB h2p + USB p2h" `
  --slice-index 100 `
  --output exp_0\processing\preprocessing\utils\make_figure\fig1_usb.png
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




def make_figure(
    image_paths,
    labels,
    output_path,
    axis=2,
    slice_index=None,
    mask_indices=None,
    dpi=300,
    title_fontsize=14,
):
    mask_indices = set(mask_indices or [])


    if len(image_paths) != len(labels):
        raise ValueError("Number of images must match number of labels.")


    slices = []


    for i, path in enumerate(image_paths):
        volume = load_nifti(path)
        sl = extract_slice(volume, axis=axis, slice_index=slice_index)
        sl = normalize_image(sl, is_mask=i in mask_indices)
        slices.append(sl)


    n = len(slices)


    fig, axes = plt.subplots(
        1,
        n,
        figsize=(3.2 * n, 4),
        gridspec_kw={"wspace": 0.01},
    )


    if n == 1:
        axes = [axes]


    for ax, sl, label in zip(axes, slices, labels):
        ax.imshow(sl, cmap="gray")
        ax.set_title(label, fontsize=title_fontsize, pad=4)
        ax.axis("off")


    plt.tight_layout()


    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)


    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.01,
    )


    plt.close()


    print(f"Saved figure to: {output_path}")




def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a combined PNG figure from MRI NIfTI files."
    )


    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Input MRI files in .nii or .nii.gz format.",
    )


    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels shown above each image. Must match --images order.",
    )


    parser.add_argument(
        "--output",
        required=True,
        help="Output PNG path.",
    )


    parser.add_argument(
        "--axis",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Slice axis: 0=sagittal, 1=coronal, 2=axial. Default: 2.",
    )


    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Slice index to extract. If omitted, uses middle slice.",
    )


    parser.add_argument(
        "--mask-indices",
        nargs="*",
        type=int,
        default=[],
        help="Indices of images that should be treated as binary masks. 0-based.",
    )


    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output PNG resolution.",
    )


    return parser.parse_args()




def main():
    args = parse_args()


    image_paths = [Path(p) for p in args.images]


    make_figure(
        image_paths=image_paths,
        labels=args.labels,
        output_path=args.output,
        axis=args.axis,
        slice_index=args.slice_index,
        mask_indices=args.mask_indices,
        dpi=args.dpi,
    )




if __name__ == "__main__":
    main()


#!/usr/bin/env python3
import argparse
from pathlib import Path


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt




r"""
py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\healthy.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\mask.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\carvemix.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\gli.nii.gz `
  --labels `
    "Ground-Truth Healthy" `
    "BraTS Tumor Mask" `
    "CarveMix Output" `
    "GliGAN output" `
  --mask-indices 1 `
  --axis 2 `
  --slice-index 80 `
  --output exp_0\processing\preprocessing\utils\make_figure\combined_synthetic_generators.png




py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\figure_inpainting_n4_rigid\healthy_n4_rigid.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\correctedinput\gli.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\correctedinput\GLILIT.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\correctedinput\GLIBID.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\correctedinput\GLIUSB.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\correctedinput\gli_sr.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\correctedinput\gli_una.nii.gz `
  --labels `
    "Ground-Truth Healthy" `
    "GliGAN-Generated Tumored T1" `
    "GliGAN T1 + FastSurfer-LIT" `
    "GliGAN T1 + Brain-ID" `
    "GliGAN T1 + USB p2h" `
    "GliGAN T1 + SynthSR" `
    "GliGAN T1 + UNA" `
  --axis 2 `
  --slice-index 80 `
  --output exp_0\processing\preprocessing\utils\make_figure\combined_inpainters_processed.png




py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\healthy_t1.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\synthetic_seg.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\gli_rerun.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\cm_rerun.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\usb_rerun.nii.gz `
  --labels `
    "Healthy T1" `
    "Input Tumor Mask" `
    "GliGAN T1" `
    "CarveMix T1" `
    "USB h2p T1" `
  --output exp_0\processing\preprocessing\utils\make_figure\exp0_figure.png




py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\healthy_t1.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\tumor_mask.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\cm_rerun.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\synthetic_seg.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\gli_rerun.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\usb_mask.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\usb_rerun.nii.gz `
  --labels `
    "Healthy T1" `
    "Input Tumor Mask" `
    "CarveMix T1" `
    "Input Tumor Mask" `
    "GliGAN T1" `
    "Input Tumor Mask" `
    "USB h2p T1" `
  --output exp_0\processing\preprocessing\utils\make_figure\exp0_figure.png




py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\healthy_t1.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\cm_mask.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\IXI012-HH-1211-T1_brain_n4_rigid__BraTS-GLI-02217-102_carvemix.nii.gz `
  --labels `
    "Healthy T1" `
    "Input Tumor Mask" `
    "CarveMix T1" `
  --mask-indices 1 `
  --output exp_0\processing\preprocessing\utils\make_figure\exp0_figure_CM.png




py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\healthy_t1.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\synthetic_seg.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\synthetic_t1.nii.gz `
  --labels `
    "Healthy T1" `
    "Input Tumor Mask" `
    "GliGAN T1" `
  --mask-indices 1 `
  --output exp_0\processing\preprocessing\utils\make_figure\exp0_figure_GLI.png




py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\y_h.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\mask_usb.nii `
    exp_0\processing\preprocessing\utils\make_figure\exp0fig\y_p.nii.gz `
  --labels `
    "Healthy T1" `
    "Input Tumor Mask" `
    "USB h2p T1" `
  --mask-indices 1 `
 --output exp_0\processing\preprocessing\utils\make_figure\exp0_figure_USB.png




py -3.10 exp_0\processing\preprocessing\utils\make_figure\make_figure2.py `
  --images `
    exp_0\processing\preprocessing\utils\make_figure\makefigs_synthmorph_aligned\IXI013-HH-1212-T1_preprocessed_aligned.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefigs_synthmorph_aligned\BraTS-GLI-02608-102_seg_rigid.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefigs_synthmorph_aligned\IXI013-HH-1212-T1_brain_n4_rigid__BraTS-GLI-02608-102_carvemix_aligned.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefigs_synthmorph_aligned\synthetic_t1_aligned.nii.gz `
    exp_0\processing\preprocessing\utils\make_figure\makefigs_synthmorph_aligned\y_p_IXI013_USB_aligned_axis0flip.nii.gz `
  --labels `
    "Healthy T1" `
    "BraTS Tumor Mask" `
    "CarveMix T1" `
    "GliGAN T1" `
    "USB h2p T1" `
  --mask-indices 1 `
  --output exp_0\processing\preprocessing\utils\make_figure\finalcorrectedexp0_figure.png `
  --slice-index 100
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




def make_figure(
    image_paths,
    labels,
    output_path,
    axis=2,
    slice_index=None,
    mask_indices=None,
    dpi=300,
    title_fontsize=14,
):
    mask_indices = set(mask_indices or [])


    if len(image_paths) != len(labels):
        raise ValueError("Number of images must match number of labels.")


    slices = []


    for i, path in enumerate(image_paths):
        volume = load_nifti(path)
        sl = extract_slice(volume, axis=axis, slice_index=slice_index)
        sl = normalize_image(sl, is_mask=i in mask_indices)
        slices.append(sl)


    n = len(slices)


    fig, axes = plt.subplots(
        1,
        n,
        figsize=(3.2 * n, 4),
        gridspec_kw={"wspace": 0.01},
    )


    if n == 1:
        axes = [axes]


    for ax, sl, label in zip(axes, slices, labels):
        ax.imshow(sl, cmap="gray")
        ax.set_title(label, fontsize=title_fontsize, pad=4)
        ax.axis("off")


    plt.tight_layout()


    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)


    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.01,
    )


    plt.close()


    print(f"Saved figure to: {output_path}")




def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a combined PNG figure from MRI NIfTI files."
    )


    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Input MRI files in .nii or .nii.gz format.",
    )


    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Labels shown above each image. Must match --images order.",
    )


    parser.add_argument(
        "--output",
        required=True,
        help="Output PNG path.",
    )


    parser.add_argument(
        "--axis",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Slice axis: 0=sagittal, 1=coronal, 2=axial. Default: 2.",
    )


    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help="Slice index to extract. If omitted, uses middle slice.",
    )


    parser.add_argument(
        "--mask-indices",
        nargs="*",
        type=int,
        default=[],
        help="Indices of images that should be treated as binary masks. 0-based.",
    )


    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output PNG resolution.",
    )


    return parser.parse_args()




def main():
    args = parse_args()


    image_paths = [Path(p) for p in args.images]


    make_figure(
        image_paths=image_paths,
        labels=args.labels,
        output_path=args.output,
        axis=args.axis,
        slice_index=args.slice_index,
        mask_indices=args.mask_indices,
        dpi=args.dpi,
    )




if __name__ == "__main__":
    main()






