#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import argparse
import shutil
import re

r"""
py -3.10 exp_0\processing\preprocessing\utils\gligan_scaled_flatten.py `
    --input-root exp_0\synth_lesion_generator\GliGAN\data\SCALED_generated_t1_gligan `
    --output-dir exp_0\synth_lesion_generator\GliGAN\data\SCALEDFLAT_generated_t1_gligan `
    --move

py -3.10 exp_0\processing\preprocessing\utils\gligan_scaled_flatten.py `
    --input-root exp_0\synth_lesion_generator\GliGAN\data\SCALED2_generated_t1_gligan `
    --output-dir exp_0\synth_lesion_generator\GliGAN\data\SCALED2FLAT_generated_t1_gligan `
    --move

py -3.10 exp_0\processing\preprocessing\utils\gligan_scaled_flatten.py `
    --input-root exp_0\synth_lesion_generator\GliGAN\data\generated_t1_gligan_FAITHFUL `
    --output-dir exp_0\synth_lesion_generator\GliGAN\data\FAITHFULFLAT_generated_t1_gligan `
    --move
"""


def strip_nii(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name




def parse_case_folder(folder_name: str):
    """
    Example input:
    IXI002-Guys-0828-T1__BraTS-GLI-00005-100__scale_0p33


    Output:
    IXI002, B00005-100, s0p33
    """
    parts = folder_name.split("__")
    if len(parts) < 2:
        raise ValueError(f"Unexpected folder name format: {folder_name}")


    healthy_part = parts[0]
    brats_part = parts[1]
    # NOT actually brats part for now, currently sample number saved in metadata folder
    #scale_part = parts[2]


    # IXI code
    m_ixi = re.search(r"(IXI\d{3})", healthy_part)
    if not m_ixi:
        raise ValueError(f"Could not parse IXI code from: {folder_name}")
    ixi_code = m_ixi.group(1)


    # BraTS code
    # BraTS-GLI-00005-100 -> B00005-100
    m_brats = re.search(r"BraTS-GLI-(\d{5}-\d{3})", brats_part)
    if not m_brats:
        raise ValueError(f"Could not parse BraTS code from: {folder_name}")
    brats_code = f"B{m_brats.group(1)}"


    # scale_0p33 -> s0p33
    '''
    m_scale = re.search(r"scale_(.+)", scale_part)
    if not m_scale:
        raise ValueError(f"Could not parse scale from: {folder_name}")
    scale_code = f"s{m_scale.group(1)}"
    '''


    return ixi_code, brats_code #, scale_code




def make_output_name(folder_name: str) -> str:
    #ixi_code, brats_code, scale_code = parse_case_folder(folder_name)
    ixi_code, brats_code = parse_case_folder(folder_name)
    #return f"{ixi_code}__{brats_code}__{scale_code}.nii.gz"
    return f"{ixi_code}__{brats_code}_t1n.nii.gz"




def flatten_synthetic_folder(input_root: Path, output_dir: Path, move: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)


    case_dirs = [p for p in input_root.iterdir() if p.is_dir()]
    if not case_dirs:
        print(f"No case folders found in: {input_root}")
        return


    copied = 0
    skipped = 0
    failed = 0


    for case_dir in sorted(case_dirs):
        src_file = case_dir / "synthetic_t1.nii.gz"


        if not src_file.exists():
            print(f"[SKIP] Missing file in {case_dir.name}")
            skipped += 1
            continue


        try:
            out_name = make_output_name(case_dir.name)
            dst_file = output_dir / out_name


            if dst_file.exists():
                print(f"[SKIP] Output already exists: {dst_file.name}")
                skipped += 1
                continue


            if move:
                shutil.move(str(src_file), str(dst_file))
            else:
                shutil.copy2(src_file, dst_file)


            copied += 1
            print(f"[{copied}] {case_dir.name} -> {dst_file.name}")


        except Exception as e:
            failed += 1
            print(f"[FAIL] {case_dir.name}: {e}")


    print("\nDone.")
    print(f"Copied/moved: {copied}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Output: {output_dir}")




def main():
    parser = argparse.ArgumentParser(
        description="Flatten nested GliGAN synthetic folders into one folder with compact filenames."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Folder containing case subfolders."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Flat output folder."
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them."
    )


    args = parser.parse_args()


    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root not found: {args.input_root}")


    flatten_synthetic_folder(args.input_root, args.output_dir, move=args.move)




if __name__ == "__main__":
    main()

