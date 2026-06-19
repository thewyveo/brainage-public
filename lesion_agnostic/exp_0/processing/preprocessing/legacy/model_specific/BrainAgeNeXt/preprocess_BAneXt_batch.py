"""
Minimal BrainAgeNeXt-style preprocessing for multiple T1-weighted MRIs in a folder.

Pipeline:
1. Skull stripping with SynthStrip
2. N4 bias field correction with ANTs (default parameters)
3. 6-DOF linear registration (rigid) to MNI152 1mm isotropic space with ANTs

This version processes all matching MRI files in an input folder recursively
and saves the final preprocessed images to an output folder.

Requirements:
- Python package: antspyx (imported as ants)
- Either:
    A) SynthStrip installed locally and available on PATH as `mri_synthstrip`
       or `synthstrip`
    B) Docker installed, so the FreeSurfer SynthStrip container can be used

Example:
python3.10 preprocess_BAneXt_batch.py \
    --input-dir data/raw/BraTS/ \
    --output-dir data/preprocessed/BraTS/ \
    --mni data/MNI152_T1_1mm_Brain.nii \
    --name-filter t1n
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


def strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def find_synthstrip_command():
    for candidate in ["mri_synthstrip", "synthstrip"]:
        if shutil.which(candidate) is not None:
            return candidate
    return None


def run_synthstrip(inp: Path, out: Path):
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
    out.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Running N4 bias field correction")
    img = ants.image_read(str(inp))
    corrected = ants.n4_bias_field_correction(img)
    ants.image_write(corrected, str(out))


def run_rigid_registration_to_mni(
    inp: Path,
    mni: Path,
    out: Path,
    transform_prefix: Path | None = None
):
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


def preprocess_single_image(
    input_path: Path,
    output_path: Path,
    mni_path: Path,
    keep_intermediate: bool = False
):
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    if not mni_path.exists():
        raise FileNotFoundError(f"MNI template not found: {mni_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stem = strip_nii_suffix(input_path)

    with tempfile.TemporaryDirectory(prefix="brainagenext_preproc_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        skullstrip_out = tmpdir / f"{stem}_brain.nii.gz"
        n4_out = tmpdir / f"{stem}_brain_n4.nii.gz"
        transform_prefix = tmpdir / f"{stem}_rigid"

        logging.info("Input image: %s", input_path)
        logging.info("Output image: %s", output_path)
        logging.info("MNI template: %s", mni_path)

        run_synthstrip(input_path, skullstrip_out)
        run_n4(skullstrip_out, n4_out)
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

    logging.info("Preprocessing completed successfully for: %s", input_path.name)
    logging.info("Final preprocessed image saved to: %s", output_path)


def find_input_images(input_dir: Path, name_filter: str | None = None):
    """
    Recursively find .nii and .nii.gz files.
    If name_filter is given, only include files whose names contain that string
    (case-insensitive).
    """
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
    """
    Preserve relative folder structure under output_dir and append _preprocessed.
    Example:
    input:  /data/BraTS/sub1/file-t1n.nii.gz
    output: /out/BraTS/sub1/file-t1n_preprocessed.nii.gz
    """
    rel = input_path.relative_to(input_dir)
    stem = strip_nii_suffix(rel)
    return output_dir / rel.parent / f"{stem}_preprocessed.nii.gz"


def preprocess_folder(
    input_dir: Path,
    output_dir: Path,
    mni_path: Path,
    keep_intermediate: bool = False,
    name_filter: str | None = None,
    limit: int | None = None
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
        description="Minimal batch preprocessing for BrainAgeNeXt-style inference."
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
        help="Path to the MNI152 1mm brain template used as fixed image."
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep skull-stripped and N4 intermediate files."
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