from pathlib import Path
import argparse
import nibabel as nib
import numpy as np




def make_binary_masks(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


    files = sorted(input_dir.glob("*.nii.gz"))


    if not files:
        raise RuntimeError(f"No .nii.gz files found in {input_dir}")


    for path in files:
        nii = nib.load(str(path))
        data = nii.get_fdata()


        binary = (data > 0).astype(np.uint8)


        header = nii.header.copy()
        header.set_data_dtype(np.uint8)


        out_path = output_dir / path.name
        nib.save(nib.Nifti1Image(binary, nii.affine, header), str(out_path))


        print(f"Saved binary mask: {out_path}")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder containing BraTS segmentation masks")
    parser.add_argument("--output_dir", required=True, help="Output folder for binary pathology_probability masks")
    args = parser.parse_args()


    make_binary_masks(Path(args.input_dir), Path(args.output_dir))




if __name__ == "__main__":
    main()


