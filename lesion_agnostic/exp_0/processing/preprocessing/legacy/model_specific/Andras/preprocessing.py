#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
Fully thesis-faithful preprocessing for the Joós two-step / multitask MRI-input model.


Pipeline:
1. Reorientation with FSL fslreorient2std
2. Field-of-view reduction with FSL robustfov
3. Rician denoising with ANTs
4. N4 bias field correction with ANTs
5. Brain extraction with SynthStrip
6. Affine registration to MNI152_T1_1mm_Brain with SynthMorph
7. 1st-99th percentile clipping + linear scaling to [0,1]
8. Save final output as .npy


Example:
first run:
subst Y: "C:\Users\P102179\OneDrive - Amsterdam UMC\Bureaublad\repo\brainage\lesion_agnostic\exp_0\Andras"

then:
py -3.10 preprocessing.py --input-dir Y:\data\raw\BraTS\ --output-dir Y:\data\preprocessed\BraTS\ --mni Y:\data\MNI152_T1_1mm_Brain.nii --name-filter t1n

python preprocessing.py --input-dir /mnt/c/Projects/thesis_project/Data/CBS \
    --output-dir /mnt/c/Projects/thesis_project/Data/brain_age_preprocessed_faithful/CBS \
    --mni /mnt/c/Projects/thesis_project/Data/MNI152_T1_1mm_Brain.nii.gz \
    --name-filter t1 \
    --workers 2
"""


import argparse
import logging
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


import ants
import nibabel as nib
import numpy as np




CACHE_FILE_NAME = "image_paths.txt"




def run_cmd(cmd, **kwargs):
    logging.info("CMD: %s", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kwargs)




def strip_suffixes(name: str) -> str:
    for s in (".nii.gz", ".nii", ".gz", ".npy"):
        if name.endswith(s):
            return name[:-len(s)]
    return name




def find_images(root: Path, name_filter: str | None = None):
    log = logging.getLogger()
    log.info("Scanning directory for images: %s", root)


    for p in root.rglob("*"):
        if not p.is_file():
            continue


        lower_name = p.name.lower()
        if not lower_name.endswith(".nii.gz"):
            continue


        if ("uni" not in lower_name) and ("t1" not in lower_name):
            continue


        if "map" in lower_name:
            continue


        if name_filter is not None and name_filter.lower() not in lower_name:
            continue


        log.info("Found image: %s", p)
        yield p


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


def run_fslreorient2std(inp: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    inp_wsl = windows_to_wsl_path(inp)
    out_wsl = windows_to_wsl_path(out)
    run_wsl_cmd(f'fslreorient2std "{inp_wsl}" "{out_wsl}"')
    return out


def run_robust_fov(inp: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    inp_wsl = windows_to_wsl_path(inp)
    out_wsl = windows_to_wsl_path(out)
    run_wsl_cmd(f'robustfov -i "{inp_wsl}" -r "{out_wsl}"')
    return out


def run_denoise(inp: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    img = ants.image_read(str(inp))
    den = ants.denoise_image(img, noise_model="Rician")
    ants.image_write(den, str(out))
    return out




def run_n4(inp: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    img = ants.image_read(str(inp))
    corr = ants.n4_bias_field_correction(img)
    ants.image_write(corr, str(out))
    return out




def run_synthstrip(inp: Path, out: Path, use_gpu: bool = False):
    """
    Uses Docker SynthStrip.
    """
    out.parent.mkdir(parents=True, exist_ok=True)


    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{inp.parent.resolve()}:/data",
        "freesurfer/synthstrip:1.7",
        "-i", f"/data/{inp.name}",
        "-o", f"/data/{out.name}",
    ]


    if use_gpu:
        docker_cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{inp.parent.resolve()}:/data",
            "freesurfer/synthstrip:1.7-gpu",
            "-i", f"/data/{inp.name}",
            "-o", f"/data/{out.name}",
            "-g",
        ]


    run_cmd(docker_cmd)
    return out




def run_synthmorph_affine(inp: Path, mni: Path, out: Path, xfm: Path, use_gpu: bool = False):
    """
    Uses Docker SynthMorph affine registration.
    Assumes inp and mni are real files with correct extensions.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    xfm.parent.mkdir(parents=True, exist_ok=True)


    docker_cmd = [
        "docker", "run", "--rm",
        "-e", "TF_CPP_MIN_LOG_LEVEL=2",
        "-v", f"{inp.parent.resolve()}:/moving",
        "-v", f"{mni.parent.resolve()}:/fixed",
        "freesurfer/synthmorph", "register",
        "-m", "affine",
        "-o", f"/moving/{out.name}",
        "-t", f"/moving/{xfm.name}",
        f"/moving/{inp.name}",
        f"/fixed/{mni.name}",
    ]


    if use_gpu:
        docker_cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-e", "TF_CPP_MIN_LOG_LEVEL=2",
            "-v", f"{inp.parent.resolve()}:/moving",
            "-v", f"{mni.parent.resolve()}:/fixed",
            "freesurfer/synthmorph", "register",
            "-g",
            "-m", "affine",
            "-o", f"/moving/{out.name}",
            "-t", f"/moving/{xfm.name}",
            f"/moving/{inp.name}",
            f"/fixed/{mni.name}",
        ]


    run_cmd(docker_cmd)
    return out



def normalize_1_99_to_unit_interval(inp_nii: Path, out_nii: Path):
    """
    Exact thesis-faithful final step:
    - clip to 1st and 99th percentiles
    - linearly scale to [0,1]
    - save as NIfTI (.nii.gz)
    """


    out_nii.parent.mkdir(parents=True, exist_ok=True)


    img = nib.load(str(inp_nii))
    data = img.get_fdata().astype(np.float32)


    p1 = np.percentile(data, 1.0)
    p99 = np.percentile(data, 99.0)


    if p99 <= p1:
        norm = np.zeros_like(data, dtype=np.float32)
    else:
        clipped = np.clip(data, p1, p99)
        norm = (clipped - p1) / (p99 - p1)
        norm = norm.astype(np.float32)


    out_img = nib.Nifti1Image(norm, img.affine, img.header)
    nib.save(out_img, str(out_nii))


    return out_nii



def process_image(img: Path, out_root: Path, mni: Path, use_gpu: bool = False, keep_intermediate: bool = False):
    log = logging.getLogger()
    stem = strip_suffixes(img.name)


    final = out_root / f"{stem}.nii.gz"
    if final.exists():
        log.info("Skipping (final exists): %s", final.name)
        return


    out_root.mkdir(parents=True, exist_ok=True)


    with tempfile.TemporaryDirectory(prefix="two_step_preproc_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)


        local_input = tmpdir / "input.nii.gz"
        local_mni = tmpdir / mni.name


        shutil.copy2(img, local_input)
        shutil.copy2(mni, local_mni)


        oriented = tmpdir / f"{stem}_reoriented.nii.gz"
        fov = tmpdir / f"{stem}_fov.nii.gz"
        den = tmpdir / f"{stem}_den.nii.gz"
        n4 = tmpdir / f"{stem}_n4.nii.gz"
        brain = tmpdir / f"{stem}_brain.nii.gz"
        reg = tmpdir / f"{stem}_registered.nii.gz"
        xfm = tmpdir / f"{stem}.lta"


        log.info("Processing: %s", img.name)
        log.info("Local input copy: %s", local_input)
        log.info("Local MNI copy: %s", local_mni)


        run_fslreorient2std(local_input, oriented)
        run_robust_fov(oriented, fov)
        run_denoise(fov, den)
        run_n4(den, n4)
        run_synthstrip(n4, brain, use_gpu=use_gpu)
        run_synthmorph_affine(brain, local_mni, reg, xfm, use_gpu=use_gpu)


        normalize_1_99_to_unit_interval(reg, final)


        if keep_intermediate:
            keep_dir = out_root / f"{stem}_intermediates"
            keep_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(oriented, keep_dir / oriented.name)
            shutil.copy2(fov, keep_dir / fov.name)
            shutil.copy2(den, keep_dir / den.name)
            shutil.copy2(n4, keep_dir / n4.name)
            shutil.copy2(brain, keep_dir / brain.name)
            shutil.copy2(reg, keep_dir / reg.name)
            shutil.copy2(xfm, keep_dir / xfm.name)


        log.info("Successfully processed: %s", img.name)


def main(input_dir: Path, output_dir: Path, mni: Path, name_filter: str | None, workers: int, use_gpu: bool, keep_intermediate: bool):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )
    log = logging.getLogger()


    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not mni.exists():
        raise FileNotFoundError(f"MNI template not found: {mni}")


    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / CACHE_FILE_NAME


    if cache_file.exists():
        log.info("Loading cached image paths from %s", cache_file)
        with cache_file.open("r") as f:
            all_imgs = [Path(line.strip()) for line in f if line.strip()]
    else:
        all_imgs = list(find_images(input_dir, name_filter=name_filter))
        with cache_file.open("w") as f:
            for img in all_imgs:
                f.write(str(img) + "\n")


    if len(all_imgs) == 0:
        log.warning("No matching images found in %s", input_dir)
        return


    pending = [img for img in all_imgs if not (output_dir / f"{strip_suffixes(img.name)}.nii.gz").exists()]
    done0 = len(all_imgs) - len(pending)


    log.info("%d images (%d done); pending: %d", len(all_imgs), done0, len(pending))
    log.info("Workers: %d", workers)


    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {
            exe.submit(
                process_image,
                img,
                output_dir,
                mni,
                use_gpu,
                keep_intermediate,
            ): img
            for img in pending
        }


        done = done0
        for fut in as_completed(futures):
            img = futures[fut]
            done += 1
            try:
                fut.result()
                log.info("[%d/%d] ✔ %s", done, len(all_imgs), img.name)
            except Exception as e:
                log.exception("[%d/%d] ✖ %s: %s", done, len(all_imgs), img.name, e)


    log.info("✅ All done.")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully thesis-faithful preprocessing for the two-step multitask model.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Root directory containing raw MRI files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save final .npy outputs.")
    parser.add_argument("--mni", type=Path, required=True, help="Path to MNI152_T1_1mm_Brain.nii.gz")
    parser.add_argument("--name-filter", type=str, default=None, help="Optional substring filter, e.g. 't1'")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers.")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU variants of SynthStrip/SynthMorph Docker containers.")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate .nii.gz files.")
    args = parser.parse_args()


    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mni=args.mni,
        name_filter=args.name_filter,
        workers=args.workers,
        use_gpu=args.use_gpu,
        keep_intermediate=args.keep_intermediate,
    )

