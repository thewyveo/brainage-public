# extract_mri_slices.py


import argparse
from pathlib import Path


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


r"""
py -3.10 test.py `
  --input_dir data\IXI_GLI_RERUN `
  --output_dir data\IXI_GLI_RERUN_imgs `
  --recursive `
  --name_filter synthetic_t1

py -3.10 test.py `
  --input_dir C:\Users\P102179\Downloads\IXI_CM_rerun\IXI_CM_rerun\synthetic `
  --output_dir data\IXI_CM_rerun_imgs `

py -3.10 test.py `
  --input_dir "\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\rerun\gen\IXI_CM\synthetic" `
  --output_dir data\qc_imgs\IXI_CM_synthetic_imgs `

py -3.10 test.py `
  --input_dir "\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\rerun\gen\IXI_GLI" `
  --output_dir data\qc_imgs\IXI_GLI_synthetic_imgs `
  --recursive `
  --name_filter synthetic_t1

py -3.10 qc.py `
  --input_dir "\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\rerun\gen\CM_BID" `
  --output_dir data\qc_imgs\CM_BID_imgs `

py -3.10 qc.py `
  --input_dir "\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\rerun\gen\CM_LIT" `
  --output_dir data\qc_imgs\CM_LIT_imgs `
  --recursive `
  --name_filter inpainting_result `
  --enable-dir-name

py -3.10 qc.py `
  --input_dir "\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\rerun\gen\GLI_BID" `
  --output_dir data\qc_imgs\GLI_BID_imgs `

py -3.10 qc.py `
  --input_dir "\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\rerun\gen\GLI_LIT" `
  --output_dir data\qc_imgs\GLI_LIT_imgs `
  --recursive `
  --name_filter inpainting_result `
  --enable-dir-name

py -3.10 qc.py `
  --input_dir "\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\rerun\gen\USB_BID" `
  --output_dir data\qc_imgs\USB_BID_imgs `

py -3.10 qc.py `
  --input_dir "\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\rerun\gen\USB_LIT" `
  --output_dir data\qc_imgs\USB_LIT_imgs `
  --recursive `
  --name_filter inpainting_result `
  --enable-dir-name
"""

def normalize_slice(slice_2d):
    slice_2d = np.asarray(slice_2d, dtype=np.float32)


    finite = np.isfinite(slice_2d)
    if not finite.any():
        return np.zeros_like(slice_2d)


    vals = slice_2d[finite]
    lo, hi = np.percentile(vals, [1, 99])


    if hi <= lo:
        return np.zeros_like(slice_2d)


    slice_2d = np.clip(slice_2d, lo, hi)
    slice_2d = (slice_2d - lo) / (hi - lo)


    return slice_2d



def get_output_name_with_subdirs(path, root_dir):
    """
    Example:


    root_dir:
        CM_LIT/


    path:
        CM_LIT/IXI013/inpainting_volume/inpainting_result.nii.gz


    output:
        IXI013__inpainting_volume__inpainting_result.png
    """


    rel = path.relative_to(root_dir)


    parts = list(rel.parts)


    filename = parts[-1]


    if filename.endswith(".nii.gz"):
        filename = filename[:-7]
    else:
        filename = Path(filename).stem


    parts[-1] = filename


    return "__".join(parts) + ".png"





def get_slice(volume, axis="axial", index=None):
    if volume.ndim == 4:
        volume = volume[..., 0]


    if axis == "axial":
        if index is None:
            index = volume.shape[2] // 2
        return volume[:, :, index]


    elif axis == "coronal":
        if index is None:
            index = volume.shape[1] // 2
        return volume[:, index, :]


    elif axis == "sagittal":
        if index is None:
            index = volume.shape[0] // 2
        return volume[index, :, :]


    else:
        raise ValueError("axis must be one of: axial, coronal, sagittal")




def clean_name(path):
    name = path.name


    if name.endswith(".nii.gz"):
        return name[:-7]


    return path.stem




def get_output_name_from_parent(path):
    """
    Example:


    input:
        subject_001/T1.nii.gz


    output:
        subject_001.png
    """


    return path.parent.name + ".png"




def extract_slices(
    input_dir,
    output_dir,
    axis="axial",
    index=None,
    recursive=False,
    name_filter=None,
    enable_dir_name=False,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)


    output_dir.mkdir(parents=True, exist_ok=True)


    if recursive:
        mri_paths = sorted(
            list(input_dir.rglob("*.nii")) +
            list(input_dir.rglob("*.nii.gz"))
        )


        if name_filter is not None:
            mri_paths = [
                p for p in mri_paths
                if name_filter in p.name
            ]


    else:
        mri_paths = sorted(
            list(input_dir.glob("*.nii")) +
            list(input_dir.glob("*.nii.gz"))
        )


    print(f"Found {len(mri_paths)} MRI files")


    for mri_path in mri_paths:
        try:
            img = nib.load(str(mri_path))
            data = img.get_fdata()


            slice_2d = get_slice(
                data,
                axis=axis,
                index=index
            )


            slice_2d = normalize_slice(slice_2d)


            # Rotate for nicer visualization
            slice_2d = np.rot90(slice_2d)


            if enable_dir_name:
                out_name = get_output_name_with_subdirs(
                    mri_path,
                    input_dir
                )


            elif recursive:
                out_name = get_output_name_from_parent(mri_path)


            else:
                out_name = clean_name(mri_path) + ".png"





            out_path = output_dir / out_name


            plt.imsave(out_path, slice_2d, cmap="gray")


            print(f"Saved: {out_path}")


        except Exception as e:
            print(f"FAILED: {mri_path} | {e}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input_dir",
        required=True,
        help="Folder containing MRI files"
    )


    parser.add_argument(
        "--output_dir",
        required=True,
        help="Folder where slice images will be saved"
    )


    parser.add_argument(
        "--axis",
        default="axial",
        choices=["axial", "coronal", "sagittal"],
        help="Slice orientation"
    )


    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Slice index. If omitted, uses the middle slice."
    )


    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subfolders recursively"
    )


    parser.add_argument(
        "--name_filter",
        type=str,
        default=None,
        help="Only use MRI files whose filename contains this string"
    )


    parser.add_argument(
    "--enable-dir-name",
    action="store_true",
    help="Include relative subdirectory path in output filename"
    )

    args = parser.parse_args()


    extract_slices(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        axis=args.axis,
        index=args.index,
        recursive=args.recursive,
        name_filter=args.name_filter,
        enable_dir_name=args.enable_dir_name,
    )
