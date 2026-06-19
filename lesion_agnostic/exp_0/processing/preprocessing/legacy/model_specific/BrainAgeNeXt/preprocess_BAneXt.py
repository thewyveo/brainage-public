"""
Minimal BrainAgeNeXt-style preprocessing for a single T1-weighted MRI.

Pipeline:
1. Skull stripping with SynthStrip
2. N4 bias field correction with ANTs (default parameters)
3. 6-DOF linear registration (rigid) to MNI152 1mm isotropic space with ANTs

This script is intentionally minimal to match the paper description as closely
as possible for an initial single-image test.

Requirements:
- Python package: antspyx (imported as ants)
- Either:
    A) SynthStrip installed locally and available on PATH as `mri_synthstrip`
       or `synthstrip`
    B) Docker installed, so the FreeSurfer SynthStrip container can be used

Example:
python3.10 preprocess/preprocess_BAneXt.py \
    --input preprocess/BraTS-GLI-02093-103-t1n.nii.gz \
    --output preprocess/preprocessed_BraTS-GLI-02093-103-t1n.nii.gz \
    --mni preprocess/MNI152_T1_1mm_Brain.nii
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import ants


def run_cmd(cmd):
    logging.info("Running command: %s", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def find_synthstrip_command():
    """
    Return the local SynthStrip executable if found, otherwise None.
    """
    for candidate in ["mri_synthstrip", "synthstrip"]:
        if shutil.which(candidate) is not None:
            return candidate
    return None


def run_synthstrip(inp: Path, out: Path):
    """
    Run SynthStrip on CPU.

    Priority:
    1. Local executable if installed
    2. Docker CPU fallback

    Output is a skull-stripped image.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    synthstrip_cmd = find_synthstrip_command()

    if synthstrip_cmd is not None:
        logging.info("Using local SynthStrip executable: %s", synthstrip_cmd)
        run_cmd([
            synthstrip_cmd,
            "-i", str(inp),
            "-o", str(out),
        ])
        return

    if shutil.which("docker") is None:
        raise RuntimeError(
            "SynthStrip not found locally and Docker is not installed. "
            "Install SynthStrip locally or install Docker."
        )

    logging.info("Using Docker fallback for SynthStrip (CPU mode)")
    run_cmd([
        "docker", "run", "--rm",
        "-v", f"{inp.parent.resolve()}:/input",
        "-v", f"{out.parent.resolve()}:/output",
        "freesurfer/synthstrip:1.7",
        "-i", f"/input/{inp.name}",
        "-o", f"/output/{out.name}",
    ])


def run_n4(inp: Path, out: Path):
    """
    Run ANTs N4 bias field correction with default parameters.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Running N4 bias field correction")
    img = ants.image_read(str(inp))
    corrected = ants.n4_bias_field_correction(img)
    ants.image_write(corrected, str(out))


def run_rigid_registration_to_mni(inp: Path, mni: Path, out: Path, transform_prefix: Path | None = None):
    """
    Register input image to MNI space using ANTs rigid registration.

    Rigid registration corresponds to 6 DOF:
    - 3 translations
    - 3 rotations

    The output saved is the moving image warped into MNI space.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Running 6-DOF rigid registration to MNI with ANTs")
    fixed = ants.image_read(str(mni))
    moving = ants.image_read(str(inp))

    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Rigid"
    )

    warped = reg["warpedmovout"]
    ants.image_write(warped, str(out))

    if transform_prefix is not None:
        transform_prefix.parent.mkdir(parents=True, exist_ok=True)
        fwd = reg.get("fwdtransforms", [])
        for i, tfm in enumerate(fwd):
            src = Path(tfm)
            if src.exists():
                dst = transform_prefix.parent / f"{transform_prefix.name}_fwd_{i}{src.suffix}"
                shutil.copy2(src, dst)


def preprocess_single_image(input_path: Path, output_path: Path, mni_path: Path, keep_intermediate: bool = False):
    """
    Full minimal preprocessing pipeline for one image.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if not mni_path.exists():
        raise FileNotFoundError(f"MNI template not found: {mni_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stem = input_path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    elif stem.endswith(".nii"):
        stem = stem[:-4]

    with tempfile.TemporaryDirectory(prefix="brainagenext_preproc_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        skullstrip_out = tmpdir / f"{stem}_brain.nii.gz"
        n4_out = tmpdir / f"{stem}_brain_n4.nii.gz"
        transform_prefix = tmpdir / f"{stem}_rigid"

        logging.info("Input image: %s", input_path)
        logging.info("Output image: %s", output_path)
        logging.info("MNI template: %s", mni_path)

        # Step 1: SynthStrip
        run_synthstrip(input_path, skullstrip_out)

        # Step 2: N4 bias correction
        run_n4(skullstrip_out, n4_out)

        # Step 3: 6-DOF linear registration to MNI152 1mm
        run_rigid_registration_to_mni(
            inp=n4_out,
            mni=mni_path,
            out=output_path,
            transform_prefix=transform_prefix if keep_intermediate else None,
        )

        if keep_intermediate:
            keep_dir = output_path.parent / f"{stem}_intermediates"
            keep_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(skullstrip_out, keep_dir / skullstrip_out.name)
            shutil.copy2(n4_out, keep_dir / n4_out.name)
            logging.info("Saved intermediate files to: %s", keep_dir)

    logging.info("Preprocessing completed successfully.")
    logging.info("Final preprocessed image saved to: %s", output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal single-image preprocessing for BrainAgeNeXt-style inference."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input T1-weighted MRI (.nii or .nii.gz)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save the final preprocessed output (.nii.gz recommended)"
    )
    parser.add_argument(
        "--mni",
        type=Path,
        required=True,
        help="Path to the MNI152 1mm brain template used as fixed image"
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep skull-stripped and N4 intermediate files"
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    args = parse_args()

    try:
        preprocess_single_image(
            input_path=args.input,
            output_path=args.output,
            mni_path=args.mni,
            keep_intermediate=args.keep_intermediate
        )
    except Exception as e:
        logging.exception("Preprocessing failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()