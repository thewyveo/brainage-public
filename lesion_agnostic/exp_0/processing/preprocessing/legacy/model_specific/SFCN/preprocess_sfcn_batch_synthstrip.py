r"""
Modified SFCN-style preprocessing for multiple T1-weighted MRIs in a folder.

Pipeline:
1. Brain extraction with SynthStrip
2. Bias field correction with FSL FAST
3. Affine registration to MNI152 standard space with FSL FLIRT

This version:
- runs from normal Windows Python
- uses WSL for FSL commands
- uses local temp copies to avoid OneDrive / mounted-drive issues

Example
-------
first run:
subst X: "C:\Users\P102179\OneDrive - Amsterdam UMC\Bureaublad\repo\brainage\lesion_agnostic\exp_0\SFCN"

then:
py -3.10 preprocessing/preprocess_sfcn_batch_synthstrip.py --input-dir X:\data\raw\BraTS\ --output-dir X:\data\preprocessed\synthstrip\BraTS\ --mni X:\data\MNI152_T1_1mm_Brain.nii --name-filter t1n
"""

#TODO
# make batch processing more efficient via docker
#TODO


import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def windows_to_wsl_path(path: Path) -> str:
    path_str = str(path)
    if len(path_str) >= 2 and path_str[1] == ":":
        drive = path_str[0].lower()
        rest = path_str[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return path_str.replace("\\", "/")


def run_wsl_cmd(cmd_str: str):
    logging.info("Running in WSL: %s", cmd_str)
    subprocess.run(["wsl", "bash", "-lc", cmd_str], check=True)


def ensure_wsl_and_fsl():
    if shutil.which("wsl") is None:
        raise RuntimeError("Could not find 'wsl'. WSL must be installed and callable from this Python environment.")

    test_cmd = "which fast && which flirt"
    try:
        subprocess.run(["wsl", "bash", "-lc", test_cmd], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Could not find FSL tools inside WSL. Make sure fast/flirt work in your WSL shell first."
        ) from e


def find_synthstrip_command():
    for candidate in ["mri_synthstrip", "synthstrip"]:
        if shutil.which(candidate) is not None:
            return candidate
    return None


def run_synthstrip(inp: Path, out: Path):
    """
    Run SynthStrip locally from Windows Python.
    Priority:
    1. local executable on PATH
    2. Docker fallback
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    synthstrip_cmd = find_synthstrip_command()

    if synthstrip_cmd is not None:
        logging.info("Using local SynthStrip executable: %s", synthstrip_cmd)
        subprocess.run(
            [synthstrip_cmd, "-i", str(inp), "-o", str(out)],
            check=True,
        )
        return

    if shutil.which("docker") is None:
        raise RuntimeError(
            "SynthStrip not found locally and Docker is not installed. "
            "Install SynthStrip locally or install Docker."
        )

    logging.info("Using Docker fallback for SynthStrip (CPU mode)")
    subprocess.run([
        "docker", "run", "--rm",
        "-v", f"{inp.parent.resolve()}:/input",
        "-v", f"{out.parent.resolve()}:/output",
        "freesurfer/synthstrip:1.7",
        "-i", f"/input/{inp.name}",
        "-o", f"/output/{out.name}",
    ], check=True)


def run_fast_bias_correction(inp: Path, out: Path):
    """
    Bias field correction with FSL FAST via WSL.

    FAST writes outputs based on a prefix.
    The bias-corrected image is <prefix>_restore.nii.gz
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    prefix = out.parent / strip_nii_suffix(out)

    inp_wsl = windows_to_wsl_path(inp)
    prefix_wsl = windows_to_wsl_path(prefix)

    cmd = f'fast -B -o "{prefix_wsl}" "{inp_wsl}"'
    run_wsl_cmd(cmd)

    restore_img = Path(str(prefix) + "_restore.nii.gz")
    if not restore_img.exists():
        raise FileNotFoundError(f"FAST restore image not found: {restore_img}")

    shutil.move(str(restore_img), str(out))

    for extra in out.parent.glob(prefix.name + "*"):
        if extra == out:
            continue
        try:
            extra.unlink()
        except (IsADirectoryError, FileNotFoundError):
            pass


def run_affine_flirt_to_mni(inp: Path, mni: Path, out: Path, mat_out: Path | None = None):
    """
    Affine registration to MNI with FSL FLIRT via WSL.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    inp_wsl = windows_to_wsl_path(inp)
    mni_wsl = windows_to_wsl_path(mni)
    out_wsl = windows_to_wsl_path(out)

    if mat_out is None:
        tmp_mat = out.parent / "tmp.mat"
        mat_wsl = windows_to_wsl_path(tmp_mat)
    else:
        mat_out.parent.mkdir(parents=True, exist_ok=True)
        mat_wsl = windows_to_wsl_path(mat_out)

    cmd = (
        f'flirt -in "{inp_wsl}" '
        f'-ref "{mni_wsl}" '
        f'-out "{out_wsl}" '
        f'-omat "{mat_wsl}" '
        f'-dof 12 '
        f'-interp trilinear'
    )

    run_wsl_cmd(cmd)

    if not out.exists():
        raise FileNotFoundError(f"FLIRT did not create output file: {out}")


def preprocess_single_image(
    input_path: Path,
    output_path: Path,
    mni_path: Path,
    keep_intermediate: bool = False,
):
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if not mni_path.exists():
        raise FileNotFoundError(f"MNI template not found: {mni_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stem = strip_nii_suffix(input_path)

    with tempfile.TemporaryDirectory(prefix="sfcn_preproc_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        local_input = tmpdir / "input.nii.gz"
        local_mni = tmpdir / "mni.nii.gz"
        shutil.copy2(input_path, local_input)
        shutil.copy2(mni_path, local_mni)

        brain_out = tmpdir / f"{stem}_brain.nii.gz"
        biascorr_out = tmpdir / f"{stem}_brain_restore.nii.gz"
        flirt_out = tmpdir / f"{stem}_preprocessed.nii.gz"
        affine_mat = tmpdir / f"{stem}_to_mni.mat"

        logging.info("Input image: %s", input_path)
        logging.info("Local input copy: %s", local_input)
        logging.info("Output image: %s", output_path)
        logging.info("Local MNI copy: %s", local_mni)

        # 1. SynthStrip
        run_synthstrip(local_input, brain_out)

        # 2. FAST
        run_fast_bias_correction(brain_out, biascorr_out)

        # 3. FLIRT to temp
        run_affine_flirt_to_mni(
            inp=biascorr_out,
            mni=local_mni,
            out=flirt_out,
            mat_out=affine_mat if keep_intermediate else None,
        )

        if not flirt_out.exists():
            raise FileNotFoundError(f"Expected FLIRT output not found: {flirt_out}")

        shutil.copy2(flirt_out, output_path)

        if not output_path.exists():
            raise FileNotFoundError(f"Final output was not copied successfully: {output_path}")

        if keep_intermediate:
            keep_dir = output_path.parent / f"{stem}_intermediates"
            keep_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(brain_out, keep_dir / brain_out.name)
            shutil.copy2(biascorr_out, keep_dir / biascorr_out.name)
            if affine_mat.exists():
                shutil.copy2(affine_mat, keep_dir / affine_mat.name)

    logging.info("Preprocessing completed successfully for: %s", input_path.name)
    logging.info("Final preprocessed image saved to: %s", output_path)


def find_input_images(input_dir: Path, name_filter: str | None = None):
    images = []
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue

        lower_name = path.name.lower()
        if not (lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")):
            continue

        if name_filter is not None and name_filter.lower() not in lower_name:
            continue

        images.append(path)

    return sorted(images)


def make_output_path(input_path: Path, input_dir: Path, output_dir: Path) -> Path:
    rel = input_path.relative_to(input_dir)
    stem = strip_nii_suffix(rel)
    return output_dir / rel.parent / f"{stem}_preprocessed.nii.gz"


def preprocess_folder(
    input_dir: Path,
    output_dir: Path,
    mni_path: Path,
    keep_intermediate: bool = False,
    name_filter: str | None = None,
    limit: int | None = None,
):
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not mni_path.exists():
        raise FileNotFoundError(f"MNI template not found: {mni_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    images = find_input_images(input_dir, name_filter=name_filter)

    if limit is not None:
        images = images[:limit]

    if not images:
        logging.warning("No matching MRI files found in: %s", input_dir)
        return

    total = len(images)
    success_count = 0
    skipped_count = 0
    failed_count = 0

    logging.info("Found %d matching image(s) to process.", total)

    for idx, input_path in enumerate(images, start=1):
        output_path = make_output_path(input_path, input_dir, output_dir)

        logging.info("=== [%d/%d] Processing %s ===", idx, total, input_path.name)

        if output_path.exists():
            logging.info("Skipping already processed file: %s", output_path)
            skipped_count += 1
            continue

        try:
            preprocess_single_image(
                input_path=input_path,
                output_path=output_path,
                mni_path=mni_path,
                keep_intermediate=keep_intermediate,
            )
            success_count += 1
        except Exception as e:
            failed_count += 1
            logging.exception("Failed on %s: %s", input_path, e)

    logging.info("=== Done ===")
    logging.info("Successful: %d", success_count)
    logging.info("Skipped: %d", skipped_count)
    logging.info("Failed: %d", failed_count)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Modified SFCN-style batch preprocessing with SynthStrip + FAST + FLIRT."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing raw MRI files (.nii or .nii.gz). Search is recursive."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder where final preprocessed outputs will be saved."
    )
    parser.add_argument(
        "--mni",
        type=Path,
        required=True,
        help="Path to the MNI152 standard-space reference image."
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep SynthStrip/FAST/FLIRT intermediate files."
    )
    parser.add_argument(
        "--name-filter",
        type=str,
        default=None,
        help="Optional case-insensitive substring filter, e.g. 't1' or 't1n'."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of files to process."
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    ensure_wsl_and_fsl()

    args = parse_args()

    try:
        preprocess_folder(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            mni_path=args.mni,
            keep_intermediate=args.keep_intermediate,
            name_filter=args.name_filter,
            limit=args.limit,
        )
    except Exception as e:
        logging.exception("Batch preprocessing failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

