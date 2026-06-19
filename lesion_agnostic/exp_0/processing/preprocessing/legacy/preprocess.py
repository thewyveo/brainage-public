#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Universal modular MRI preprocessing script.

py -3.10 exp_0\processing\preprocessing\overall\preprocess.py `
    --profile joos `
    --input-dir data\raw\IXI-T1 `
    --output-dir data\preprocessed\Andras\IXI `
    --mni data\MNI152_T1_1mm_Brain.nii `
    --name-filter T1 `
    --workers 1 `


py -3.10 exp_0\processing\preprocessing\overall\preprocess.py `
    --profile brainagenext `
    --input-dir data\raw\IXI-T1 `
    --output-dir data\preprocessed\BrainAgeNeXt\IXI `
    --mni data\MNI152_T1_1mm_Brain.nii `
    --name-filter T1 `
    --workers 1 `

Goal
----
One script that can reproduce or approximate the preprocessing families you showed:
- Joós two-step / multitask model
- BrainAgeNeXt
- SFCN faithful
- SFCN modified

Every preprocessing method is implemented as its own function and can be
enabled/disabled independently. You can also use a preset profile and override
individual steps if needed.

Implemented methods
-------------------
FSL / FSL-through-WSL:
- fslreorient2std
- robustfov
- BET
- FAST
- FLIRT

ANTs / ANTsPy:
- Rician denoising
- N4 bias correction
- Rigid registration
- Affine registration

SynthStrip:
- local executable
- Docker fallback

SynthMorph:
- Docker affine registration

Intensity normalization:
- 1st–99th percentile clipping + scaling to [0,1]

Saving:
- .nii.gz output
- optional .npy output

Profiles
--------
--profile joos
    1. fslreorient2std
    2. robustfov
    3. ANTs Rician denoising
    4. ANTs N4
    5. SynthStrip
    6. SynthMorph affine registration
    7. 1-99 clipping + [0,1]
    8. optional .npy save

--profile brainagenext
    1. SynthStrip
    2. ANTs N4
    3. ANTs rigid registration

--profile sfcn_faithful
    1. BET
    2. FAST
    3. FLIRT affine registration

--profile sfcn_modified
    1. SynthStrip
    2. FAST
    3. FLIRT affine registration

--profile custom
    Use explicit flags only.

Example
-------
Joós:
python universal_preprocess.py \
    --profile joos \
    --input-dir /data/raw/BraTS \
    --output-dir /data/preprocessed/joos/BraTS \
    --mni /data/MNI152_T1_1mm_Brain.nii.gz \
    --name-filter t1n \
    --workers 2 \
    --save-npy

BrainAgeNeXt:
python universal_preprocess.py \
    --profile brainagenext \
    --input-dir /data/raw/BraTS \
    --output-dir /data/preprocessed/brainagenext/BraTS \
    --mni /data/MNI152_T1_1mm_Brain.nii.gz \
    --name-filter t1n

SFCN faithful (Windows + WSL):
py -3.10 universal_preprocess.py ^
    --profile sfcn_faithful ^
    --input-dir X:\\data\\raw\\BraTS ^
    --output-dir X:\\data\\preprocessed\\sfcn_faithful\\BraTS ^
    --mni X:\\data\\MNI152_T1_1mm_Brain.nii.gz ^
    --name-filter t1n

Notes
-----
- FSL steps can run directly or through WSL. Use --fsl-mode wsl for Windows/FSL-in-WSL.
- SynthStrip tries local executable first, then Docker fallback.
- SynthMorph uses Docker.
- Registration outputs are saved as processed images; transform files are optionally saved when available.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

import ants
import nibabel as nib
import numpy as np


CACHE_FILE_NAME = "image_paths.txt"


# =============================================================================
# Utilities
# =============================================================================

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )


def run_cmd(cmd: list[str], **kwargs) -> None:
    logging.info("CMD: %s", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kwargs)


def run_shell_cmd(cmd: str) -> None:
    logging.info("SHELL CMD: %s", cmd)
    subprocess.run(cmd, shell=True, check=True)


def windows_to_wsl_path(path: Path) -> str:
    path_str = str(path)
    if len(path_str) >= 2 and path_str[1] == ":":
        drive = path_str[0].lower()
        rest = path_str[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return path_str.replace("\\", "/")


def run_wsl_cmd(cmd_str: str) -> None:
    logging.info("Running in WSL: %s", cmd_str)
    subprocess.run(["wsl", "bash", "-lc", cmd_str], check=True)


def safe_copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def strip_suffixes(name: str) -> str:
    for s in (".nii.gz", ".nii", ".gz", ".npy"):
        if name.endswith(s):
            return name[:-len(s)]
    return name


def strip_nii_suffix(path: Path) -> str:
    return strip_suffixes(path.name)


def find_images(root: Path, name_filter: str | None = None) -> Iterable[Path]:
    log = logging.getLogger()
    log.info("Scanning directory for images: %s", root)

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        lower_name = p.name.lower()
        if not (lower_name.endswith(".nii.gz") or lower_name.endswith(".nii")):
            continue

        if name_filter is not None and name_filter.lower() not in lower_name:
            continue

        log.info("Found image: %s", p)
        yield p


def make_output_path(input_path: Path, input_dir: Path, output_dir: Path, suffix: str = "_preprocessed") -> Path:
    rel = input_path.relative_to(input_dir)
    stem = strip_suffixes(rel.name)
    return output_dir / rel.parent / f"{stem}{suffix}.nii.gz"


def maybe_copy_disabled(inp: Path, out: Path, enabled: bool, step_name: str) -> Optional[Path]:
    if enabled:
        return None
    logging.info("%s disabled; copying input forward.", step_name)
    return safe_copy(inp, out)


def ensure_parent(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)


def find_synthstrip_command() -> Optional[str]:
    for candidate in ["mri_synthstrip", "synthstrip"]:
        if shutil.which(candidate) is not None:
            return candidate
    return None


# =============================================================================
# FSL availability helpers
# =============================================================================

def ensure_wsl_and_fsl(required_tools: list[str]) -> None:
    if shutil.which("wsl") is None:
        raise RuntimeError("Could not find 'wsl'. WSL must be installed and callable from this environment.")

    test_cmd = " && ".join([f"which {tool}" for tool in required_tools])
    try:
        subprocess.run(["wsl", "bash", "-lc", test_cmd], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Could not find required FSL tools inside WSL: {required_tools}. "
            "Make sure they work in your WSL shell first."
        ) from e


def ensure_local_tools(required_tools: list[str]) -> None:
    missing = [tool for tool in required_tools if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"Required tools not found on PATH: {missing}")


# =============================================================================
# Step functions
# =============================================================================

def run_fslreorient2std(inp: Path, out: Path, enabled: bool, fsl_mode: str = "local") -> Path:
    copied = maybe_copy_disabled(inp, out, enabled, "fslreorient2std")
    if copied is not None:
        return copied

    ensure_parent(out)
    if fsl_mode == "wsl":
        inp_wsl = windows_to_wsl_path(inp)
        out_wsl = windows_to_wsl_path(out)
        run_wsl_cmd(f'fslreorient2std "{inp_wsl}" "{out_wsl}"')
    else:
        run_cmd(["fslreorient2std", str(inp), str(out)])
    return out


def run_robustfov(inp: Path, out: Path, enabled: bool, fsl_mode: str = "local") -> Path:
    copied = maybe_copy_disabled(inp, out, enabled, "robustfov")
    if copied is not None:
        return copied

    ensure_parent(out)
    if fsl_mode == "wsl":
        inp_wsl = windows_to_wsl_path(inp)
        out_wsl = windows_to_wsl_path(out)
        run_wsl_cmd(f'robustfov -i "{inp_wsl}" -r "{out_wsl}"')
    else:
        run_cmd(["robustfov", "-i", str(inp), "-r", str(out)])
    return out


def run_denoise_rician(inp: Path, out: Path, enabled: bool) -> Path:
    copied = maybe_copy_disabled(inp, out, enabled, "ANTs Rician denoising")
    if copied is not None:
        return copied

    ensure_parent(out)
    img = ants.image_read(str(inp))
    den = ants.denoise_image(img, noise_model="Rician")
    ants.image_write(den, str(out))
    return out


def run_n4(inp: Path, out: Path, enabled: bool) -> Path:
    copied = maybe_copy_disabled(inp, out, enabled, "ANTs N4 bias correction")
    if copied is not None:
        return copied

    ensure_parent(out)
    img = ants.image_read(str(inp))
    corr = ants.n4_bias_field_correction(img)
    ants.image_write(corr, str(out))
    return out


def run_synthstrip(inp: Path, out: Path, enabled: bool, use_gpu: bool = False) -> Path:
    copied = maybe_copy_disabled(inp, out, enabled, "SynthStrip")
    if copied is not None:
        return copied

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


def run_bet(inp: Path, out: Path, enabled: bool, fsl_mode: str = "local", frac: float = 0.5, robust: bool = True) -> Path:
    copied = maybe_copy_disabled(inp, out, enabled, "FSL BET")
    if copied is not None:
        return copied

    ensure_parent(out)

    if fsl_mode == "wsl":
        inp_wsl = windows_to_wsl_path(inp)
        out_wsl = windows_to_wsl_path(out)
        cmd = f'bet "{inp_wsl}" "{out_wsl}" -f {frac}'
        if robust:
            cmd += " -R"
        run_wsl_cmd(cmd)
    else:
        cmd = ["bet", str(inp), str(out), "-f", str(frac)]
        if robust:
            cmd.append("-R")
        run_cmd(cmd)
    return out


def run_fast_bias_correction(inp: Path, out: Path, enabled: bool, fsl_mode: str = "local") -> Path:
    copied = maybe_copy_disabled(inp, out, enabled, "FSL FAST bias correction")
    if copied is not None:
        return copied

    ensure_parent(out)
    prefix = out.parent / strip_suffixes(out.name)

    if fsl_mode == "wsl":
        inp_wsl = windows_to_wsl_path(inp)
        prefix_wsl = windows_to_wsl_path(prefix)
        run_wsl_cmd(f'fast -B -o "{prefix_wsl}" "{inp_wsl}"')
    else:
        run_cmd(["fast", "-B", "-o", str(prefix), str(inp)])

    restore_img = Path(str(prefix) + "_restore.nii.gz")
    restore_img_nii = Path(str(prefix) + "_restore.nii")
    actual_restore = restore_img if restore_img.exists() else restore_img_nii

    if not actual_restore.exists():
        raise FileNotFoundError(f"FAST restore image not found: {restore_img} or {restore_img_nii}")

    shutil.move(str(actual_restore), str(out))

    for extra in out.parent.glob(prefix.name + "*"):
        if extra == out:
            continue
        try:
            if extra.is_file():
                extra.unlink()
        except FileNotFoundError:
            pass

    return out


def run_flirt_affine(inp: Path, ref: Path, out: Path, enabled: bool, fsl_mode: str = "local", mat_out: Path | None = None, dof: int = 12) -> Path:
    copied = maybe_copy_disabled(inp, out, enabled, "FSL FLIRT affine registration")
    if copied is not None:
        return copied

    ensure_parent(out)
    if mat_out is None:
        mat_out = out.parent / f"{strip_suffixes(out.name)}.mat"
    mat_out.parent.mkdir(parents=True, exist_ok=True)

    if fsl_mode == "wsl":
        inp_wsl = windows_to_wsl_path(inp)
        ref_wsl = windows_to_wsl_path(ref)
        out_wsl = windows_to_wsl_path(out)
        mat_wsl = windows_to_wsl_path(mat_out)
        cmd = (
            f'flirt -in "{inp_wsl}" '
            f'-ref "{ref_wsl}" '
            f'-out "{out_wsl}" '
            f'-omat "{mat_wsl}" '
            f'-dof {dof} '
            f'-interp trilinear'
        )
        run_wsl_cmd(cmd)
    else:
        run_cmd([
            "flirt",
            "-in", str(inp),
            "-ref", str(ref),
            "-out", str(out),
            "-omat", str(mat_out),
            "-dof", str(dof),
            "-interp", "trilinear",
        ])

    if not out.exists():
        raise FileNotFoundError(f"FLIRT did not create output file: {out}")

    return out


def run_ants_registration(inp: Path, ref: Path, out: Path, enabled: bool, transform_type: str = "Rigid", transform_prefix: Path | None = None) -> Path:
    copied = maybe_copy_disabled(inp, out, enabled, f"ANTs {transform_type} registration")
    if copied is not None:
        return copied

    ensure_parent(out)

    moving = ants.image_read(str(inp))
    fixed = ants.image_read(str(ref))

    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=transform_type
    )

    ants.image_write(reg["warpedmovout"], str(out))

    if transform_prefix is not None:
        transform_prefix.parent.mkdir(parents=True, exist_ok=True)
        tx_log = transform_prefix.with_suffix(".txt")
        with open(tx_log, "w", encoding="utf-8") as f:
            f.write("forward_transforms:\n")
            for t in reg.get("fwdtransforms", []):
                f.write(f"{t}\n")
            f.write("inverse_transforms:\n")
            for t in reg.get("invtransforms", []):
                f.write(f"{t}\n")

    return out


def run_synthmorph_affine(inp: Path, ref: Path, out: Path, xfm: Path, enabled: bool, use_gpu: bool = False) -> Path:
    copied = maybe_copy_disabled(inp, out, enabled, "SynthMorph affine registration")
    if copied is not None:
        return copied

    ensure_parent(out)
    xfm.parent.mkdir(parents=True, exist_ok=True)

    image_name = "freesurfer/synthmorph"
    docker_cmd = [
        "docker", "run", "--rm",
        "-e", "TF_CPP_MIN_LOG_LEVEL=2",
        "-v", f"{inp.parent.resolve()}:/moving",
        "-v", f"{ref.parent.resolve()}:/fixed",
        image_name, "register",
        "-m", "affine",
        "-o", f"/moving/{out.name}",
        "-t", f"/moving/{xfm.name}",
        f"/moving/{inp.name}",
        f"/fixed/{ref.name}",
    ]

    if use_gpu:
        docker_cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-e", "TF_CPP_MIN_LOG_LEVEL=2",
            "-v", f"{inp.parent.resolve()}:/moving",
            "-v", f"{ref.parent.resolve()}:/fixed",
            image_name, "register",
            "-g",
            "-m", "affine",
            "-o", f"/moving/{out.name}",
            "-t", f"/moving/{xfm.name}",
            f"/moving/{inp.name}",
            f"/fixed/{ref.name}",
        ]

    run_cmd(docker_cmd)
    return out


def normalize_1_99_to_unit_interval(inp_nii: Path, out_nii: Path, enabled: bool, nonzero_only: bool = False) -> Path:
    copied = maybe_copy_disabled(inp_nii, out_nii, enabled, "1-99 percentile normalization")
    if copied is not None:
        return copied

    ensure_parent(out_nii)
    img = nib.load(str(inp_nii))
    data = img.get_fdata().astype(np.float32)

    vals = data[data != 0] if nonzero_only else data.reshape(-1)
    if vals.size == 0:
        raise ValueError(f"No voxels available for normalization in {inp_nii}")

    p1 = np.percentile(vals, 1.0)
    p99 = np.percentile(vals, 99.0)

    if p99 <= p1:
        norm = np.zeros_like(data, dtype=np.float32)
    else:
        clipped = np.clip(data, p1, p99)
        norm = (clipped - p1) / (p99 - p1)
        norm = norm.astype(np.float32)
        if nonzero_only:
            norm[data == 0] = 0.0

    out_img = nib.Nifti1Image(norm, img.affine, img.header)
    nib.save(out_img, str(out_nii))
    return out_nii


def save_npy_from_nifti(inp_nii: Path, out_npy: Path, enabled: bool) -> Path | None:
    if not enabled:
        logging.info("NPY saving disabled.")
        return None

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(inp_nii))
    data = img.get_fdata().astype(np.float32)
    np.save(str(out_npy), data)
    return out_npy


# =============================================================================
# Profile handling
# =============================================================================

def apply_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """
    Fill unset flags from the chosen profile.
    Explicit user flags should override profile defaults.
    """

    profile_defaults = {
        "joos": {
            "enable_reorient": True,
            "enable_robustfov": True,
            "enable_denoise": True,
            "enable_n4": True,
            "enable_synthstrip": True,
            "enable_bet": False,
            "enable_fast": False,
            "registration_backend": "synthmorph_affine",
            "enable_normalize_1_99": True,
            "save_npy": True,
            "output_suffix": "",
        },
        "brainagenext": {
            "enable_reorient": False,
            "enable_robustfov": False,
            "enable_denoise": False,
            "enable_n4": True,
            "enable_synthstrip": True,
            "enable_bet": False,
            "enable_fast": False,
            "registration_backend": "ants_rigid",
            "enable_normalize_1_99": False,
            "save_npy": False,
            "output_suffix": "_preprocessed",
        },
        "sfcn_faithful": {
            "enable_reorient": False,
            "enable_robustfov": False,
            "enable_denoise": False,
            "enable_n4": False,
            "enable_synthstrip": False,
            "enable_bet": True,
            "enable_fast": True,
            "registration_backend": "flirt_affine",
            "enable_normalize_1_99": False,
            "save_npy": False,
            "output_suffix": "_preprocessed",
        },
        "sfcn_modified": {
            "enable_reorient": False,
            "enable_robustfov": False,
            "enable_denoise": False,
            "enable_n4": False,
            "enable_synthstrip": True,
            "enable_bet": False,
            "enable_fast": True,
            "registration_backend": "flirt_affine",
            "enable_normalize_1_99": False,
            "save_npy": False,
            "output_suffix": "_preprocessed",
        },
        "custom": {},
    }

    defaults = profile_defaults[args.profile]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    # Final fallbacks for custom or unspecified overrides
    final_defaults = {
        "enable_reorient": False,
        "enable_robustfov": False,
        "enable_denoise": False,
        "enable_n4": False,
        "enable_synthstrip": False,
        "enable_bet": False,
        "enable_fast": False,
        "registration_backend": "none",
        "enable_normalize_1_99": False,
        "save_npy": False,
        "output_suffix": "_preprocessed",
    }
    for key, value in final_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    # Consistency checks
    if args.enable_synthstrip and args.enable_bet:
        raise ValueError("Choose only one brain extraction backend per run: SynthStrip or BET.")

    if args.enable_n4 and args.enable_fast:
        raise ValueError("Choose only one bias correction backend per run: N4 or FAST.")

    return args


# =============================================================================
# Per-image pipeline
# =============================================================================

def process_image(
    img: Path,
    out_root: Path,
    input_root: Path,
    mni: Path,
    args: argparse.Namespace,
) -> Path:
    log = logging.getLogger()
    final = make_output_path(img, input_root, out_root, suffix=args.output_suffix)

    if final.exists():
        log.info("Skipping (final exists): %s", final)
        return final

    final.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="universal_preproc_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        local_input = tmpdir / "input.nii.gz"
        local_mni = tmpdir / mni.name
        shutil.copy2(img, local_input)
        shutil.copy2(mni, local_mni)

        stem = strip_suffixes(img.name)

        p_reorient = tmpdir / f"{stem}_reoriented.nii.gz"
        p_fov = tmpdir / f"{stem}_fov.nii.gz"
        p_denoise = tmpdir / f"{stem}_denoise.nii.gz"
        p_brain = tmpdir / f"{stem}_brain.nii.gz"
        p_bias = tmpdir / f"{stem}_biascorr.nii.gz"
        p_reg = tmpdir / f"{stem}_registered.nii.gz"
        p_norm = tmpdir / f"{stem}_normalized.nii.gz"
        p_xfm = tmpdir / f"{stem}.lta"
        p_mat = tmpdir / f"{stem}_to_mni.mat"

        log.info("Processing: %s", img.name)

        current = local_input

        # 1. Reorientation
        current = run_fslreorient2std(current, p_reorient, args.enable_reorient, fsl_mode=args.fsl_mode)

        # 2. FOV reduction
        current = run_robustfov(current, p_fov, args.enable_robustfov, fsl_mode=args.fsl_mode)

        # 3. Denoising
        current = run_denoise_rician(current, p_denoise, args.enable_denoise)

        # 4. Brain extraction
        if args.enable_synthstrip:
            current = run_synthstrip(current, p_brain, True, use_gpu=args.use_gpu)
        elif args.enable_bet:
            current = run_bet(current, p_brain, True, fsl_mode=args.fsl_mode, frac=args.bet_frac, robust=args.bet_robust)
        else:
            current = safe_copy(current, p_brain)

        # 5. Bias correction
        if args.enable_n4:
            current = run_n4(current, p_bias, True)
        elif args.enable_fast:
            current = run_fast_bias_correction(current, p_bias, True, fsl_mode=args.fsl_mode)
        else:
            current = safe_copy(current, p_bias)

        # 6. Registration
        reg_backend = args.registration_backend
        if reg_backend == "none":
            current = safe_copy(current, p_reg)
        elif reg_backend == "ants_rigid":
            current = run_ants_registration(current, local_mni, p_reg, True, transform_type="Rigid", transform_prefix=tmpdir / f"{stem}_ants_rigid")
        elif reg_backend == "ants_affine":
            current = run_ants_registration(current, local_mni, p_reg, True, transform_type="Affine", transform_prefix=tmpdir / f"{stem}_ants_affine")
        elif reg_backend == "flirt_affine":
            current = run_flirt_affine(current, local_mni, p_reg, True, fsl_mode=args.fsl_mode, mat_out=p_mat, dof=12)
        elif reg_backend == "synthmorph_affine":
            current = run_synthmorph_affine(current, local_mni, p_reg, p_xfm, True, use_gpu=args.use_gpu)
        else:
            raise ValueError(f"Unknown registration backend: {reg_backend}")

        # 7. Intensity normalization
        current = normalize_1_99_to_unit_interval(
            current,
            p_norm,
            args.enable_normalize_1_99,
            nonzero_only=args.normalize_nonzero_only,
        )

        shutil.copy2(current, final)

        # 8. Optional npy save
        if args.save_npy:
            out_npy = final.with_suffix("").with_suffix(".npy")
            save_npy_from_nifti(final, out_npy, True)

        # 9. Keep intermediates if requested
        if args.keep_intermediate:
            keep_dir = final.parent / f"{strip_suffixes(final.name)}_intermediates"
            keep_dir.mkdir(parents=True, exist_ok=True)
            for f in [p_reorient, p_fov, p_denoise, p_brain, p_bias, p_reg, p_norm]:
                if f.exists():
                    shutil.copy2(f, keep_dir / f.name)
            for f in [p_xfm, p_mat, (tmpdir / f"{stem}_ants_rigid.txt"), (tmpdir / f"{stem}_ants_affine.txt")]:
                if Path(f).exists():
                    shutil.copy2(f, keep_dir / Path(f).name)

        log.info("Saved final output: %s", final)
        return final


# =============================================================================
# Argument parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal modular MRI preprocessing.")

    parser.add_argument("--profile", type=str, required=True,
                        choices=["joos", "brainagenext", "sfcn_faithful", "sfcn_modified", "custom"],
                        help="Preprocessing profile preset.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Root directory containing raw MRI files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save final outputs.")
    parser.add_argument("--mni", type=Path, required=True, help="Path to MNI152_T1_1mm_Brain.nii.gz")
    parser.add_argument("--name-filter", type=str, default=None, help="Optional substring filter, e.g. 't1'")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU variants when available.")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of files.")
    parser.add_argument("--refresh-cache", action="store_true", help="Rescan input directory even if cache exists.")
    parser.add_argument("--fsl-mode", type=str, default="local", choices=["local", "wsl"],
                        help="How to run FSL tools.")

    # Optional explicit overrides; None means 'use profile default'
    parser.add_argument("--enable-reorient", dest="enable_reorient", action="store_true", default=None)
    parser.add_argument("--disable-reorient", dest="enable_reorient", action="store_false")

    parser.add_argument("--enable-robustfov", dest="enable_robustfov", action="store_true", default=None)
    parser.add_argument("--disable-robustfov", dest="enable_robustfov", action="store_false")

    parser.add_argument("--enable-denoise", dest="enable_denoise", action="store_true", default=None)
    parser.add_argument("--disable-denoise", dest="enable_denoise", action="store_false")

    parser.add_argument("--enable-n4", dest="enable_n4", action="store_true", default=None)
    parser.add_argument("--disable-n4", dest="enable_n4", action="store_false")

    parser.add_argument("--enable-synthstrip", dest="enable_synthstrip", action="store_true", default=None)
    parser.add_argument("--disable-synthstrip", dest="enable_synthstrip", action="store_false")

    parser.add_argument("--enable-bet", dest="enable_bet", action="store_true", default=None)
    parser.add_argument("--disable-bet", dest="enable_bet", action="store_false")

    parser.add_argument("--enable-fast", dest="enable_fast", action="store_true", default=None)
    parser.add_argument("--disable-fast", dest="enable_fast", action="store_false")

    parser.add_argument("--registration-backend", type=str, default=None,
                        choices=["none", "ants_rigid", "ants_affine", "flirt_affine", "synthmorph_affine"],
                        help="Registration backend override.")

    parser.add_argument("--enable-normalize-1-99", dest="enable_normalize_1_99", action="store_true", default=None)
    parser.add_argument("--disable-normalize-1-99", dest="enable_normalize_1_99", action="store_false")
    parser.add_argument("--normalize-nonzero-only", action="store_true",
                        help="Compute percentile normalization on nonzero voxels only.")

    parser.add_argument("--save-npy", dest="save_npy", action="store_true", default=None)
    parser.add_argument("--disable-save-npy", dest="save_npy", action="store_false")

    parser.add_argument("--output-suffix", type=str, default=None,
                        help="Suffix before .nii.gz, e.g. '_preprocessed'. Empty string is allowed.")

    parser.add_argument("--bet-frac", type=float, default=0.5, help="BET fractional intensity threshold.")
    parser.add_argument("--bet-robust", action="store_true", default=True, help="Use BET -R.")
    parser.add_argument("--no-bet-robust", dest="bet_robust", action="store_false")

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    setup_logging()
    args = parse_args()
    args = apply_profile_defaults(args)

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.mni.exists():
        raise FileNotFoundError(f"MNI template not found: {args.mni}")

    # Tool checks based on effective config
    if args.fsl_mode == "wsl":
        required_fsl = []
        if args.enable_reorient:
            required_fsl.append("fslreorient2std")
        if args.enable_robustfov:
            required_fsl.append("robustfov")
        if args.enable_bet:
            required_fsl.append("bet")
        if args.enable_fast:
            required_fsl.append("fast")
        if args.registration_backend == "flirt_affine":
            required_fsl.append("flirt")
        if required_fsl:
            ensure_wsl_and_fsl(required_fsl)
    else:
        required_fsl = []
        if args.enable_reorient:
            required_fsl.append("fslreorient2std")
        if args.enable_robustfov:
            required_fsl.append("robustfov")
        if args.enable_bet:
            required_fsl.append("bet")
        if args.enable_fast:
            required_fsl.append("fast")
        if args.registration_backend == "flirt_affine":
            required_fsl.append("flirt")
        if required_fsl:
            ensure_local_tools(required_fsl)

    if args.registration_backend == "synthmorph_affine" and shutil.which("docker") is None:
        raise RuntimeError("SynthMorph affine requires Docker, but Docker was not found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = args.output_dir / CACHE_FILE_NAME

    if cache_file.exists() and not args.refresh_cache:
        logging.info("Loading cached image paths from %s", cache_file)
        with cache_file.open("r", encoding="utf-8") as f:
            all_imgs = [Path(line.strip()) for line in f if line.strip()]
    else:
        all_imgs = list(find_images(args.input_dir, name_filter=args.name_filter))
        with cache_file.open("w", encoding="utf-8") as f:
            for img in all_imgs:
                f.write(str(img) + "\n")

    if args.limit is not None:
        all_imgs = all_imgs[:args.limit]

    if len(all_imgs) == 0:
        logging.warning("No matching images found in %s", args.input_dir)
        return

    pending = [
        img for img in all_imgs
        if not make_output_path(img, args.input_dir, args.output_dir, suffix=args.output_suffix).exists()
    ]
    done0 = len(all_imgs) - len(pending)

    logging.info("Profile: %s", args.profile)
    logging.info("Images found: %d (%d already done, %d pending)", len(all_imgs), done0, len(pending))
    logging.info("Workers: %d", args.workers)

    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futures = {
            exe.submit(
                process_image,
                img,
                args.output_dir,
                args.input_dir,
                args.mni,
                args,
            ): img
            for img in pending
        }

        done = done0
        for fut in as_completed(futures):
            img = futures[fut]
            done += 1
            try:
                result = fut.result()
                logging.info("[%d/%d] ✔ %s -> %s", done, len(all_imgs), img.name, result.name)
            except Exception as e:
                logging.exception("[%d/%d] ✖ %s: %s", done, len(all_imgs), img.name, e)

    logging.info("All done.")


if __name__ == "__main__":
    main()