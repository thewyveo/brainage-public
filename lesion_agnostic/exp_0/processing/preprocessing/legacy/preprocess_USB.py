#!/usr/bin/env python3
# -*- coding: utf-8 -*-


r"""
USB preprocessing script


Purpose
-------
Prepare:
1. healthy IXI T1 images
2. BraTS T1 images + BraTS segmentation masks


into a common USB-ready space so that:
- healthy images are ready for USB
- masks are binary
- masks and images live in the same template space
- final geometry matches USB input geometry


Pipeline
--------
Healthy IXI:
    SynthStrip
    -> resample to 1 mm isotropic
    -> affine register to template
    -> center crop/pad to 160 x 160 x 160


BraTS T1 + mask:
    BraTS T1: SynthStrip
    -> resample to 1 mm isotropic
    -> affine register to template


    BraTS mask:
    -> apply same affine transform as BraTS T1 using nearest-neighbor
    -> binarize
    -> center crop/pad to 160 x 160 x 160


Important
---------
This script performs a practical common-space alignment strategy.
It is appropriate if you want a pool of:
- USB-ready healthy IXI images
- USB-ready BraTS-derived binary masks
that can be paired later.


Example
-------
py -3.10 exp_0\processing\preprocessing\overall\preprocess_USB.py `
    --healthy-dir data\raw\IXI-T1 `
    --brats-t1-dir data\raw\CM_BraTS_Masks `
    --brats-mask-dir data\raw\CM_BraTS_Masks `
    --template data\MNI152_T1_1mm_Brain.nii `
    --output-dir data\preprocessed\USB `
    --healthy-filter T1 `
    --brats-t1-filter t1n `
    --brats-mask-filter seg


Outputs
-------
output-dir/
├── healthy/
│   └── ..._usb.nii.gz
├── brats_t1/
│   └── ..._usb.nii.gz
├── masks/
│   └── ..._usbmask.nii.gz
└── metadata/
    ├── healthy_manifest.csv
    └── brats_manifest.csv
"""


from __future__ import annotations


import argparse
import csv
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


import ants
import nibabel as nib
import numpy as np




# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------




def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")




def strip_suffixes(name: str) -> str:
    for s in (".nii.gz", ".nii", ".gz", ".npy", ".npz", ".json", ".csv"):
        if name.endswith(s):
            return name[:-len(s)]
    return name




def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)




def safe_copy(src: Path, dst: Path) -> Path:
    ensure_parent(dst)
    shutil.copy2(src, dst)
    return dst




def find_nifti(root: Path, name_filter: Optional[str] = None) -> list[Path]:
    out = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        lower = p.name.lower()
        if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
            continue
        if name_filter is not None and name_filter.lower() not in lower:
            continue
        out.append(p)
    return sorted(out)




def load_nifti(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    data = img.get_fdata()
    return data, img




def save_nifti_float(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    ensure_parent(out_path)
    header = ref_img.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data.astype(np.float32), ref_img.affine, header), str(out_path))




def save_nifti_uint8(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    ensure_parent(out_path)
    header = ref_img.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(data.astype(np.uint8), ref_img.affine, header), str(out_path))




def find_synthstrip_command() -> Optional[str]:
    for candidate in ["mri_synthstrip", "synthstrip"]:
        if shutil.which(candidate) is not None:
            return candidate
    return None




def run_cmd(cmd: list[str]) -> None:
    logging.info("CMD: %s", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)




def normalize_case_key(path: Path) -> str:
    """
    Build a robust pairing key from filename by stripping common modality suffixes.
    """
    stem = strip_suffixes(path.name)


    suffixes = [
        "_seg", "-seg",
        "_label", "-label",
        "_labels", "-labels",
        "_t1", "-t1",
        "_t1n", "-t1n",
        "_t1ce", "-t1ce",
        "_t1c", "-t1c",
        "_flair", "-flair",
        "_t2", "-t2",
    ]


    changed = True
    while changed:
        changed = False
        lower = stem.lower()
        for suf in suffixes:
            if lower.endswith(suf):
                stem = stem[:-len(suf)]
                changed = True
                break


    return stem




# -----------------------------------------------------------------------------
# Core processing helpers
# -----------------------------------------------------------------------------




def run_synthstrip(inp: Path, out: Path, use_gpu: bool = False) -> Path:
    ensure_parent(out)


    synthstrip_cmd = find_synthstrip_command()
    if synthstrip_cmd is not None:
        cmd = [synthstrip_cmd, "-i", str(inp), "-o", str(out)]
        if use_gpu:
            cmd.append("-g")
        run_cmd(cmd)
        return out


    if shutil.which("docker") is None:
        raise RuntimeError(
            "SynthStrip not found locally and Docker is not installed. "
            "Install SynthStrip locally or install Docker."
        )


    image_name = "freesurfer/synthstrip:1.7-gpu" if use_gpu else "freesurfer/synthstrip:1.7"
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{inp.parent.resolve()}:/input",
        "-v", f"{out.parent.resolve()}:/output",
        image_name,
        "-i", f"/input/{inp.name}",
        "-o", f"/output/{out.name}",
    ]
    if use_gpu:
        docker_cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{inp.parent.resolve()}:/input",
            "-v", f"{out.parent.resolve()}:/output",
            image_name,
            "-i", f"/input/{inp.name}",
            "-o", f"/output/{out.name}",
            "-g",
        ]
    run_cmd(docker_cmd)
    return out




def run_resample_spacing(inp: Path, out: Path, spacing: tuple[float, float, float], is_mask: bool) -> Path:
    ensure_parent(out)
    img = ants.image_read(str(inp))
    interp_type = 1 if is_mask else 0
    resampled = ants.resample_image(
        img,
        resample_params=spacing,
        use_voxels=False,
        interp_type=interp_type,
    )
    ants.image_write(resampled, str(out))
    return out




def run_affine_registration_to_template(
    inp: Path,
    template: Path,
    out: Path,
    tx_prefix: Path,
) -> tuple[Path, list[str]]:
    """
    Register an image to template, return output image and forward transforms.
    """
    ensure_parent(out)
    ensure_parent(tx_prefix)


    moving = ants.image_read(str(inp))
    fixed = ants.image_read(str(template))


    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Affine",
    )


    ants.image_write(reg["warpedmovout"], str(out))


    tx_log = tx_prefix.with_suffix(".txt")
    with open(tx_log, "w", encoding="utf-8") as f:
        f.write("forward_transforms:\n")
        for t in reg.get("fwdtransforms", []):
            f.write(f"{t}\n")
        f.write("inverse_transforms:\n")
        for t in reg.get("invtransforms", []):
            f.write(f"{t}\n")


    return out, list(reg["fwdtransforms"])




def run_apply_transforms_to_mask(
    mask_path: Path,
    template_path: Path,
    out_path: Path,
    transform_list: list[str],
) -> Path:
    ensure_parent(out_path)


    moving = ants.image_read(str(mask_path))
    fixed = ants.image_read(str(template_path))


    warped = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=transform_list,
        interpolator="nearestNeighbor",
    )
    ants.image_write(warped, str(out_path))
    return out_path




def run_binarize_mask(inp: Path, out: Path) -> Path:
    ensure_parent(out)
    data, img = load_nifti(inp)
    data = (data > 0).astype(np.uint8)
    save_nifti_uint8(data, img, out)
    return out




def run_center_crop_or_pad(
    inp: Path,
    out: Path,
    target_shape: tuple[int, int, int],
    is_mask: bool,
    pad_value: float = 0.0,
) -> Path:
    ensure_parent(out)
    img = nib.load(str(inp))
    data = img.get_fdata()


    if data.ndim != 3:
        raise ValueError(f"Expected 3D image for crop/pad, got shape {data.shape} in {inp}")


    current = data


    pad_width = []
    for cur, tgt in zip(current.shape, target_shape):
        if cur < tgt:
            total = tgt - cur
            before = total // 2
            after = total - before
            pad_width.append((before, after))
        else:
            pad_width.append((0, 0))


    if any(b != (0, 0) for b in pad_width):
        current = np.pad(current, pad_width, mode="constant", constant_values=pad_value)


    slices = []
    for cur, tgt in zip(current.shape, target_shape):
        if cur > tgt:
            start = (cur - tgt) // 2
            end = start + tgt
            slices.append(slice(start, end))
        else:
            slices.append(slice(0, cur))


    current = current[tuple(slices)]


    if current.shape != target_shape:
        raise RuntimeError(f"Crop/pad failed: got {current.shape}, expected {target_shape}")


    if is_mask:
        current = (current > 0).astype(np.uint8)
        save_nifti_uint8(current, img, out)
    else:
        save_nifti_float(current.astype(np.float32), img, out)


    return out




# -----------------------------------------------------------------------------
# Main per-file pipelines
# -----------------------------------------------------------------------------




def preprocess_healthy_image(
    img_path: Path,
    template_path: Path,
    out_dir: Path,
    use_gpu: bool,
    target_spacing: tuple[float, float, float],
    target_shape: tuple[int, int, int],
) -> Path:
    stem = strip_suffixes(img_path.name)
    final_out = out_dir / f"{stem}_usb.nii.gz"


    if final_out.exists():
        logging.info("Skipping existing healthy output: %s", final_out)
        return final_out


    with tempfile.TemporaryDirectory(prefix="usb_preproc_healthy_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)


        p0 = tmpdir / "input.nii.gz"
        p1 = tmpdir / "brain.nii.gz"
        p2 = tmpdir / "resampled.nii.gz"
        p3 = tmpdir / "registered.nii.gz"
        tx_prefix = tmpdir / "affine_tx"
        p4 = tmpdir / "cropped.nii.gz"


        safe_copy(img_path, p0)
        run_synthstrip(p0, p1, use_gpu=use_gpu)
        run_resample_spacing(p1, p2, spacing=target_spacing, is_mask=False)
        run_affine_registration_to_template(p2, template_path, p3, tx_prefix)
        run_center_crop_or_pad(p3, p4, target_shape=target_shape, is_mask=False, pad_value=0.0)


        ensure_parent(final_out)
        shutil.copy2(p4, final_out)


    logging.info("Saved healthy USB image: %s", final_out)
    return final_out




def preprocess_brats_t1_and_mask(
    brats_t1_path: Path,
    brats_mask_path: Path,
    template_path: Path,
    out_t1_dir: Path,
    out_mask_dir: Path,
    use_gpu: bool,
    target_spacing: tuple[float, float, float],
    target_shape: tuple[int, int, int],
) -> tuple[Path, Path]:
    stem = strip_suffixes(brats_t1_path.name)
    case_key = normalize_case_key(brats_t1_path)


    final_t1 = out_t1_dir / f"{stem}_usb.nii.gz"
    final_mask = out_mask_dir / f"{case_key}_usbmask.nii.gz"


    if final_t1.exists() and final_mask.exists():
        logging.info("Skipping existing BraTS USB outputs: %s / %s", final_t1, final_mask)
        return final_t1, final_mask


    with tempfile.TemporaryDirectory(prefix="usb_preproc_brats_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)


        # BraTS T1 pipeline
        p_t1_0 = tmpdir / "brats_t1_input.nii.gz"
        p_t1_1 = tmpdir / "brats_t1_brain.nii.gz"
        p_t1_2 = tmpdir / "brats_t1_resampled.nii.gz"
        p_t1_3 = tmpdir / "brats_t1_registered.nii.gz"
        tx_prefix = tmpdir / "brats_affine_tx"
        p_t1_4 = tmpdir / "brats_t1_cropped.nii.gz"


        safe_copy(brats_t1_path, p_t1_0)
        run_synthstrip(p_t1_0, p_t1_1, use_gpu=use_gpu)
        run_resample_spacing(p_t1_1, p_t1_2, spacing=target_spacing, is_mask=False)
        _, fwd_txs = run_affine_registration_to_template(p_t1_2, template_path, p_t1_3, tx_prefix)
        run_center_crop_or_pad(p_t1_3, p_t1_4, target_shape=target_shape, is_mask=False, pad_value=0.0)


        ensure_parent(final_t1)
        shutil.copy2(p_t1_4, final_t1)


        # BraTS mask pipeline
        p_m_0 = tmpdir / "mask_input.nii.gz"
        p_m_1 = tmpdir / "mask_registered.nii.gz"
        p_m_2 = tmpdir / "mask_binary.nii.gz"
        p_m_3 = tmpdir / "mask_cropped.nii.gz"


        safe_copy(brats_mask_path, p_m_0)
        run_apply_transforms_to_mask(p_m_0, template_path, p_m_1, fwd_txs)
        run_binarize_mask(p_m_1, p_m_2)
        run_center_crop_or_pad(p_m_2, p_m_3, target_shape=target_shape, is_mask=True, pad_value=0)


        ensure_parent(final_mask)
        shutil.copy2(p_m_3, final_mask)


    logging.info("Saved BraTS USB T1: %s", final_t1)
    logging.info("Saved BraTS USB mask: %s", final_mask)
    return final_t1, final_mask




# -----------------------------------------------------------------------------
# Manifests
# -----------------------------------------------------------------------------




def write_manifest_csv(rows: list[dict], out_path: Path) -> None:
    ensure_parent(out_path)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)




# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="USB-only preprocessing for IXI healthy images and BraTS masks.")


    parser.add_argument("--healthy-dir", type=Path, required=True,
                        help="Directory containing healthy IXI T1 images.")
    parser.add_argument("--brats-t1-dir", type=Path, required=True,
                        help="Directory containing BraTS T1/T1n images used to align masks.")
    parser.add_argument("--brats-mask-dir", type=Path, required=True,
                        help="Directory containing BraTS segmentation masks.")
    parser.add_argument("--template", type=Path, required=True,
                        help="Common-space template image, e.g. MNI152_T1_1mm_Brain.nii.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Root output directory.")


    parser.add_argument("--healthy-filter", type=str, default="T1",
                        help="Substring filter for healthy IXI images.")
    parser.add_argument("--brats-t1-filter", type=str, default="t1n",
                        help="Substring filter for BraTS T1 images.")
    parser.add_argument("--brats-mask-filter", type=str, default="seg",
                        help="Substring filter for BraTS segmentation masks.")


    parser.add_argument("--target-spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0),
                        metavar=("SX", "SY", "SZ"),
                        help="Target spacing. Default: 1 1 1")
    parser.add_argument("--target-shape", type=int, nargs=3, default=(160, 160, 160),
                        metavar=("NX", "NY", "NZ"),
                        help="Target USB shape. Default: 160 160 160")


    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU SynthStrip docker variant if available.")
    parser.add_argument("--limit-healthy", type=int, default=None,
                        help="Optional limit on number of healthy IXI images.")
    parser.add_argument("--limit-brats", type=int, default=None,
                        help="Optional limit on number of BraTS cases.")
    return parser.parse_args()




# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------




def main() -> None:
    setup_logging()
    args = parse_args()


    if not args.healthy_dir.exists():
        raise FileNotFoundError(f"Healthy directory not found: {args.healthy_dir}")
    if not args.brats_t1_dir.exists():
        raise FileNotFoundError(f"BraTS T1 directory not found: {args.brats_t1_dir}")
    if not args.brats_mask_dir.exists():
        raise FileNotFoundError(f"BraTS mask directory not found: {args.brats_mask_dir}")
    if not args.template.exists():
        raise FileNotFoundError(f"Template not found: {args.template}")


    healthy_out_dir = args.output_dir / "healthy"
    brats_t1_out_dir = args.output_dir / "brats_t1"
    masks_out_dir = args.output_dir / "masks"
    meta_out_dir = args.output_dir / "metadata"


    healthy_out_dir.mkdir(parents=True, exist_ok=True)
    brats_t1_out_dir.mkdir(parents=True, exist_ok=True)
    masks_out_dir.mkdir(parents=True, exist_ok=True)
    meta_out_dir.mkdir(parents=True, exist_ok=True)


    # Find healthy IXI images
    healthy_files = find_nifti(args.healthy_dir, args.healthy_filter)
    if args.limit_healthy is not None:
        healthy_files = healthy_files[:args.limit_healthy]


    if len(healthy_files) == 0:
        raise FileNotFoundError(
            f"No healthy IXI images found in {args.healthy_dir} with filter={args.healthy_filter}"
        )


    # Find BraTS T1 and masks
    brats_t1_files = find_nifti(args.brats_t1_dir, args.brats_t1_filter)
    brats_mask_files = find_nifti(args.brats_mask_dir, args.brats_mask_filter)


    if args.limit_brats is not None:
        brats_t1_files = brats_t1_files[:args.limit_brats]


    if len(brats_t1_files) == 0:
        raise FileNotFoundError(
            f"No BraTS T1 files found in {args.brats_t1_dir} with filter={args.brats_t1_filter}"
        )
    if len(brats_mask_files) == 0:
        raise FileNotFoundError(
            f"No BraTS mask files found in {args.brats_mask_dir} with filter={args.brats_mask_filter}"
        )


    logging.info("Healthy IXI images found: %d", len(healthy_files))
    logging.info("BraTS T1 files found: %d", len(brats_t1_files))
    logging.info("BraTS mask files found: %d", len(brats_mask_files))


    # Preprocess healthy IXI images
    healthy_manifest = []
    for i, img_path in enumerate(healthy_files, start=1):
        logging.info("[Healthy %d/%d] %s", i, len(healthy_files), img_path.name)
        out_path = preprocess_healthy_image(
            img_path=img_path,
            template_path=args.template,
            out_dir=healthy_out_dir,
            use_gpu=args.use_gpu,
            target_spacing=tuple(args.target_spacing),
            target_shape=tuple(args.target_shape),
        )
        healthy_manifest.append({
            "input_path": str(img_path),
            "output_path": str(out_path),
        })


    # Pair BraTS T1 with BraTS mask
    mask_map = {normalize_case_key(p): p for p in brats_mask_files}


    brats_manifest = []
    skipped_no_mask = 0


    for i, t1_path in enumerate(brats_t1_files, start=1):
        case_key = normalize_case_key(t1_path)
        mask_path = mask_map.get(case_key)


        logging.info("[BraTS %d/%d] case=%s", i, len(brats_t1_files), case_key)


        if mask_path is None:
            logging.warning("No matching BraTS mask found for T1: %s", t1_path)
            skipped_no_mask += 1
            continue


        out_t1, out_mask = preprocess_brats_t1_and_mask(
            brats_t1_path=t1_path,
            brats_mask_path=mask_path,
            template_path=args.template,
            out_t1_dir=brats_t1_out_dir,
            out_mask_dir=masks_out_dir,
            use_gpu=args.use_gpu,
            target_spacing=tuple(args.target_spacing),
            target_shape=tuple(args.target_shape),
        )


        brats_manifest.append({
            "case_key": case_key,
            "brats_t1_input": str(t1_path),
            "brats_mask_input": str(mask_path),
            "brats_t1_output": str(out_t1),
            "mask_output": str(out_mask),
        })


    write_manifest_csv(healthy_manifest, meta_out_dir / "healthy_manifest.csv")
    write_manifest_csv(brats_manifest, meta_out_dir / "brats_manifest.csv")


    logging.info("Done.")
    logging.info("Healthy processed: %d", len(healthy_manifest))
    logging.info("BraTS processed: %d", len(brats_manifest))
    logging.info("BraTS skipped (no matching mask): %d", skipped_no_mask)




if __name__ == "__main__":
    main()

