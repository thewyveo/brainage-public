#!/usr/bin/env python3
# -*- coding: utf-8 -*-


r"""
Corrected GliGAN T1 inference script with strict one-to-one IXI-BraTS pairing.


Key behavior:
- No IXI is used more than once.
- No BraTS mask is used more than once.
- Pairings are decided BEFORE generation and saved immediately.
- Pairings remain correct even if a case fails/skips during generation.
- Full console output is mirrored to a log file.
- Every attempted case receives a metadata.json, including failed/skipped cases.
- Output case folder names encode both IXI and BraTS IDs.


Example Windows:


py -3.10 exp_0\synth_lesion_generator\GliGAN\src\infer\infer.py `
    --healthy-dir C:\Users\P102179\OneDrive - Amsterdam UMC\Bureaublad\storage\preprocessed\preprocessedGliGAN\IXI `
    --label-dir C:\Users\P102179\OneDrive - Amsterdam UMC\Bureaublad\storage\raw\CM_BraTS_Masks `
    --output-dir data\IXI_GLI`
    --generator-path exp_0\synth_lesion_generator\GliGAN\Checkpoint\brats2024\t1\weights\generator_457870.pt `
    --dataset BRATS_2024 `
    --device cpu `
    --seed 42 `
    --overwrite


Example Linux:


python3.10 exp_0/synth_lesion_generator/GliGAN/src/infer/t1_only_from_brats_maps_unique_pairs.py \
    --healthy-dir data/preprocessed/GliGAN/IXI \
    --label-dir data/raw/CM_BraTS_Masks \
    --output-dir exp_0/synth_lesion_generator/GliGAN/data/generated_t1_gligan_FAITHFUL_RERUN \
    --generator-path exp_0/synth_lesion_generator/GliGAN/Checkpoint/brats2024/t1/weights/generator_457870.pt \
    --dataset BRATS_2024 \
    --device cpu \
    --seed 42 \
    --overwrite
"""


from __future__ import annotations


import argparse
import csv
import json
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional


import nibabel as nib
import numpy as np
import torch
from monai.networks.nets.swin_unetr import SwinUNETR
from scipy.ndimage import label as cc_label


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SRC_DIR))


from utils.convert_to_multi_channel_based_on_brats_classes import (  # noqa: E402
    ConvertToMultiChannelBasedOnBratsGliomaClasses2023,
    ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024,
)




# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


_LOG_FILE: Optional[Path] = None




def setup_logger(output_dir: Path) -> Path:
    global _LOG_FILE
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_FILE = output_dir / f"gligan_inference_log_{timestamp}.txt"
    _LOG_FILE.write_text("", encoding="utf-8")
    return _LOG_FILE




def log(msg: str = "") -> None:
    text = str(msg)
    print(text, flush=True)
    if _LOG_FILE is not None:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(text + "\n")




def log_exception(prefix: str, e: BaseException) -> None:
    log(f"{prefix}: {type(e).__name__}: {e}")
    if _LOG_FILE is not None:
        with open(_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(traceback.format_exc())
            f.write("\n")




# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------


def find_nifti_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            files.append(p)
    return sorted(files)




def strip_nii(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name




def clean_stem_from_path(path: Path) -> str:
    stem = strip_nii(path.name)
    stem = stem.replace("_preprocessed", "")
    stem = stem.replace("-seg", "")
    stem = stem.replace("_seg", "")
    return stem




def load_nifti(path: Path):
    img = nib.load(str(path))
    data = img.get_fdata().astype(np.float32)
    return data, img




def save_nifti(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ref_img.header.copy()
    header.set_data_dtype(np.float32)
    out = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, header)
    nib.save(out, str(out_path))




def save_label_nifti(data: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = ref_img.header.copy()
    header.set_data_dtype(np.int16)
    out = nib.Nifti1Image(data.astype(np.int16), ref_img.affine, header)
    nib.save(out, str(out_path))




def save_json(obj: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)




def rescale_array(arr: np.ndarray, minv: float = 0.0, maxv: float = 1.0):
    mina = np.amin(arr)
    maxa = np.amax(arr)
    if mina == maxa:
        return arr * minv
    norm = (arr - mina) / (maxa - mina)
    return (norm * (maxv - minv)) + minv




def rescale_gaussian_noise(arr: np.ndarray, minv: float, maxv: float):
    uniq = np.unique(arr)
    if len(uniq) < 2:
        return arr * minv
    maxa = uniq[-2]
    mina = np.amin(arr)
    if mina == maxa:
        return arr * minv
    norm = (arr - mina) / (maxa - mina)
    return (norm * (maxv - minv)) + minv




def pad_to_shape(arr: np.ndarray, target_shape=(256, 256, 256)):
    current_shape = arr.shape
    pad_width = []
    for current_size, target_size in zip(current_shape, target_shape):
        total_padding = target_size - current_size
        if total_padding < 0:
            raise ValueError(f"Array shape {arr.shape} is larger than target shape {target_shape}.")
        half_padding = total_padding // 2
        pad_width.append((half_padding, total_padding - half_padding))
    return np.pad(arr, pad_width, mode="constant", constant_values=0), pad_width




def crop_from_pad(arr: np.ndarray, pad_width):
    slices = []
    for axis, (left, right) in enumerate(pad_width):
        start = left
        end = arr.shape[axis] - right
        slices.append(slice(start, end))
    return arr[tuple(slices)]




def largest_cc(mask: np.ndarray) -> np.ndarray:
    lbl, n = cc_label(mask > 0)
    if n == 0:
        return mask.astype(np.uint8)
    counts = np.bincount(lbl.ravel())
    counts[0] = 0
    return (lbl == counts.argmax()).astype(np.uint8)




def get_generator(model_path: Path, in_channels: int, out_channels: int, feature_size: int, use_checkpoint: bool, device: torch.device):
    generator = SwinUNETR(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
    )
    state = torch.load(str(model_path), map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    generator.load_state_dict(state)
    generator.to(device)
    generator.eval()
    return generator




def case_dir_for_pair(healthy_path: Path, label_path: Path, output_dir: Path) -> Path:
    stem_h = clean_stem_from_path(healthy_path)
    stem_l = clean_stem_from_path(label_path)
    case_name = f"{stem_h}__{stem_l}"
    return output_dir / case_name




def case_is_success_complete(case_dir: Path) -> bool:
    required = [
        case_dir / "healthy_t1.nii.gz",
        case_dir / "synthetic_t1.nii.gz",
        case_dir / "synthetic_seg.nii.gz",
        case_dir / "metadata.json",
    ]
    return all(p.exists() for p in required)




# -----------------------------------------------------------------------------
# Pairing
# -----------------------------------------------------------------------------


def make_unique_pairs(healthy_files: list[Path], label_files: list[Path], seed: int, pairing_mode: str) -> list[tuple[int, Path, Path]]:
    if len(label_files) < len(healthy_files):
        raise RuntimeError(
            f"Not enough BraTS masks for unique one-to-one pairing: "
            f"healthy={len(healthy_files)}, masks={len(label_files)}."
        )


    healthy_sorted = sorted(healthy_files)
    label_sorted = sorted(label_files)


    rng = random.Random(seed)


    if pairing_mode == "shuffle-labels":
        labels = label_sorted[:]
        rng.shuffle(labels)
        pairs = list(zip(healthy_sorted, labels[:len(healthy_sorted)]))
    elif pairing_mode == "shuffle-both":
        healthy = healthy_sorted[:]
        labels = label_sorted[:]
        rng.shuffle(healthy)
        rng.shuffle(labels)
        pairs = list(zip(healthy, labels[:len(healthy)]))
    elif pairing_mode == "sorted":
        pairs = list(zip(healthy_sorted, label_sorted[:len(healthy_sorted)]))
    else:
        raise ValueError(f"Unknown pairing_mode: {pairing_mode}")


    out = []
    for idx, (healthy_path, label_path) in enumerate(pairs, start=1):
        out.append((idx, healthy_path, label_path))


    # Safety checks.
    healthy_stems = [clean_stem_from_path(h) for _, h, _ in out]
    label_stems = [clean_stem_from_path(l) for _, _, l in out]


    if len(healthy_stems) != len(set(healthy_stems)):
        raise RuntimeError("Duplicate healthy IXI stem detected after pairing. Refusing to continue.")
    if len(label_stems) != len(set(label_stems)):
        raise RuntimeError("Duplicate BraTS label stem detected after pairing. Refusing to continue.")


    return out




def write_pairings_csv(pairs: list[tuple[int, Path, Path]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_index",
                "ixi_id",
                "gli_id",
                "healthy_path",
                "label_path",
                "case_dir_name",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        for idx, healthy_path, label_path in pairs:
            stem_h = clean_stem_from_path(healthy_path)
            stem_l = clean_stem_from_path(label_path)
            writer.writerow({
                "pair_index": idx,
                "ixi_id": stem_h,
                "gli_id": stem_l,
                "healthy_path": str(healthy_path),
                "label_path": str(label_path),
                "case_dir_name": f"{stem_h}__{stem_l}",
                "status": "PLANNED",
                "error": "",
            })




def update_pairing_status(out_csv: Path, pair_index: int, status: str, error: str = "") -> None:
    # Small CSV, safe simple rewrite. Keeps pairings visible during/after run.
    rows = []
    with open(out_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if int(row["pair_index"]) == int(pair_index):
                row["status"] = status
                row["error"] = error
            rows.append(row)


    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)




def write_metadata(
    case_dir: Path,
    status: str,
    pair_index: int,
    healthy_path: Path,
    label_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    extra: Optional[dict] = None,
    error: str = "",
) -> None:
    stem_h = clean_stem_from_path(healthy_path)
    stem_l = clean_stem_from_path(label_path)


    obj = {
        "status": status,
        "error": error,
        "pair_index": int(pair_index),
        "healthy_path": str(healthy_path),
        "healthy_stem": stem_h,
        "label_path": str(label_path),
        "label_stem": stem_l,
        "ixi_id": stem_h,
        "gli_id": stem_l,
        "case_dir": str(case_dir),
        "generator_path": str(args.generator_path),
        "dataset": args.dataset,
        "device": str(device),
        "seed": int(args.seed),
        "pairing_mode": args.pairing_mode,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


    if extra:
        obj.update(extra)


    save_json(obj, case_dir / "metadata.json")




# -----------------------------------------------------------------------------
# Tumor generation helpers
# -----------------------------------------------------------------------------


def add_gaussian_noise_tumour(scan: np.ndarray, label_arr: np.ndarray):
    scan_noisy = np.copy(scan)
    noise = np.full((96, 96, 96), 1000.0, dtype=np.float32)


    coords = np.where(label_arr != 0)
    if len(coords[0]) > 0:
        noise[coords] = np.random.randn(len(coords[0])).astype(np.float32)


    np.copyto(scan_noisy, noise, where=np.logical_and(noise < 100, scan_noisy != -1))
    scan_noisy = rescale_gaussian_noise(scan_noisy, -1, 1)
    noise[noise > 100] = 0
    return scan_noisy, noise




def correct_label(label_arr: np.ndarray, healthy_scan: np.ndarray, original_label: np.ndarray):
    out = np.copy(label_arr)
    out[(healthy_scan == 0) | (original_label != 0)] = 0
    return out




def correct_background(healthy_crop_pad: np.ndarray, imgs_recon: np.ndarray):
    out = np.copy(imgs_recon)
    out[(healthy_crop_pad == -1) | (out < -1)] = -1
    out[out > 1] = 1
    return out




def get_inten_coord(healthy_scan_crop: np.ndarray, original_label_crop: np.ndarray, noise: np.ndarray):
    untouch_x_axis = []
    untouch_y_axis = []
    untouch_z_axis = []
    constant = 5


    def verify(x_axis, y_axis, z_axis):
        if (
            healthy_scan_crop[x_axis, y_axis, z_axis] != 0
            and original_label_crop[x_axis, y_axis, z_axis] == 0
            and noise[x_axis, y_axis, z_axis] == 0
        ):
            mini_x = max(x_axis - constant, 0)
            maxi_x = min(x_axis + constant, 95)
            mini_y = max(y_axis - constant, 0)
            maxi_y = min(y_axis + constant, 95)
            mini_z = max(z_axis - constant, 0)
            maxi_z = min(z_axis + constant, 95)
            zeros = np.sum(noise[mini_x:maxi_x, mini_y:maxi_y, mini_z:maxi_z])
            if zeros == 0:
                untouch_x_axis.append(x_axis)
                untouch_y_axis.append(y_axis)
                untouch_z_axis.append(z_axis)


    y_axis = 0
    for x_axis in range(96):
        for z_axis in range(96):
            verify(x_axis, y_axis, z_axis)


    y_axis = 95
    for x_axis in range(96):
        for z_axis in range(96):
            verify(x_axis, y_axis, z_axis)


    z_axis = 0
    for x_axis in range(96):
        for y_axis in range(96):
            verify(x_axis, y_axis, z_axis)


    z_axis = 95
    for x_axis in range(96):
        for y_axis in range(96):
            verify(x_axis, y_axis, z_axis)


    x_axis = 0
    for y_axis in range(96):
        for z_axis in range(96):
            verify(x_axis, y_axis, z_axis)


    x_axis = 95
    for y_axis in range(96):
        for z_axis in range(96):
            verify(x_axis, y_axis, z_axis)


    return untouch_x_axis, untouch_y_axis, untouch_z_axis




def linear_interpolation(final_recons: np.ndarray, healthy_scan_crop: np.ndarray, untouch_x_axis, untouch_y_axis, untouch_z_axis):
    x_list = [-1]
    y_list = [0]


    for i in range(len(untouch_x_axis)):
        y_1 = healthy_scan_crop[untouch_x_axis[i], untouch_y_axis[i], untouch_z_axis[i]]
        x_1 = final_recons[untouch_x_axis[i], untouch_y_axis[i], untouch_z_axis[i]]
        x_list.append(x_1)
        y_list.append(y_1)


    if len(x_list) < 2:
        inten_scan = np.copy(final_recons)
    else:
        coeffs = np.polyfit(x_list, y_list, 1)
        m, b = coeffs
        inten_scan = m * final_recons + b


    inten_scan[healthy_scan_crop == 0] = 0
    inten_scan[inten_scan < 0] = 0
    return inten_scan




def random_center_tumour_voxel(healthy_scan: np.ndarray, original_label: np.ndarray):
    for _ in range(1000):
        x_axis, y_axis = np.random.randint(low=0, high=256, size=2, dtype=int)
        z_axis = int(np.random.randint(low=0, high=256, size=1, dtype=int)[0])


        if healthy_scan[x_axis, y_axis, z_axis] != 0:
            x_min = x_axis - 48
            x_max = x_axis + 48
            y_min = y_axis - 48
            y_max = y_axis + 48
            z_min = z_axis - 48
            z_max = z_axis + 48


            if x_min < 0 or y_min < 0 or z_min < 0 or x_max > 256 or y_max > 256 or z_max > 256:
                continue


            if np.sum(original_label[x_min:x_max, y_min:y_max, z_min:z_max]) == 0:
                return x_axis, y_axis, z_axis


    return None, None, None




def compute_label_bbox(label_arr: np.ndarray):
    coords = np.argwhere(label_arr > 0)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return {
        "x_extreme_min": int(mins[0]),
        "x_extreme_max": int(maxs[0]),
        "y_extreme_min": int(mins[1]),
        "y_extreme_max": int(maxs[1]),
        "z_extreme_min": int(mins[2]),
        "z_extreme_max": int(maxs[2]),
    }




def prepar_healthy_scan_to_gen(x, y, z, healthy_scan: np.ndarray, original_label: np.ndarray):
    x_min = x - 48
    x_max = x + 48
    y_min = y - 48
    y_max = y + 48
    z_min = z - 48
    z_max = z + 48


    if x_min < 0 or y_min < 0 or z_min < 0 or x_max > 256 or y_max > 256 or z_max > 256:
        raise ValueError("Chosen center gives out-of-bounds crop.")


    healthy_scan_crop = np.copy(healthy_scan[x_min:x_max, y_min:y_max, z_min:z_max])
    original_label_crop = np.copy(original_label[x_min:x_max, y_min:y_max, z_min:z_max])


    return healthy_scan_crop, original_label_crop, x_min, x_max, y_min, y_max, z_min, z_max




def prepar_real_label_to_gen(label_arr: np.ndarray, healthy_scan_crop: np.ndarray, original_label_crop: np.ndarray, label_transform, device: torch.device):
    bbox = compute_label_bbox(label_arr)
    if bbox is None:
        raise ValueError("Provided label is empty.")


    label_x_min = bbox["x_extreme_min"] - 1
    label_x_max = bbox["x_extreme_max"]
    label_y_min = bbox["y_extreme_min"] - 1
    label_y_max = bbox["y_extreme_max"]
    label_z_min = bbox["z_extreme_min"] - 1
    label_z_max = bbox["z_extreme_max"]


    if label_x_max - label_x_min > 96:
        label_x_min = bbox["x_extreme_min"]
    if label_y_max - label_y_min > 96:
        label_y_min = bbox["y_extreme_min"]
    if label_z_max - label_z_min > 96:
        label_z_min = bbox["z_extreme_min"]


    label_crop = np.copy(label_arr[label_x_min:label_x_max, label_y_min:label_y_max, label_z_min:label_z_max])


    def padding_need(min_value, max_value):
        diff = max_value - min_value
        base = int((96 - diff) / 2)
        top = int((96 - diff) / 2 + 0.5)
        return base, top


    x_base_pad, x_top_pad = padding_need(label_x_min, label_x_max)
    y_base_pad, y_top_pad = padding_need(label_y_min, label_y_max)
    z_base_pad, z_top_pad = padding_need(label_z_min, label_z_max)


    label_crop = np.pad(
        np.copy(label_crop),
        pad_width=((x_base_pad, x_top_pad), (y_base_pad, y_top_pad), (z_base_pad, z_top_pad)),
        mode="constant",
        constant_values=(0, 0),
    )


    label_crop = correct_label(np.copy(label_crop), healthy_scan_crop, original_label_crop)


    if np.sum(label_crop > 0) == 0:
        raise ValueError("Label became empty after correction against healthy crop/background.")


    label_crop_t = torch.from_numpy(np.copy(label_crop))
    label_crop_t = label_transform()(label_crop_t)
    label_crop_t = torch.reshape(label_crop_t, (1, int(label_crop_t.shape[0]), 96, 96, 96)).float().to(device)


    return label_crop_t, label_crop




def insert_tumour_t1(
    generator_t1,
    healthy_scan: np.ndarray,
    original_label: np.ndarray,
    new_label: np.ndarray,
    x: int,
    y: int,
    z: int,
    label_transform,
    device: torch.device,
):
    healthy_scan_crop, original_label_crop, x_min, x_max, y_min, y_max, z_min, z_max = prepar_healthy_scan_to_gen(
        x, y, z, healthy_scan, original_label
    )


    label_crop_cuda, label_crop = prepar_real_label_to_gen(
        label_arr=new_label,
        healthy_scan_crop=healthy_scan_crop,
        original_label_crop=original_label_crop,
        label_transform=label_transform,
        device=device,
    )


    healthy_scan_crop_norm = rescale_array(np.copy(healthy_scan_crop), minv=-1, maxv=1)
    healthy_noisy, noise = add_gaussian_noise_tumour(scan=healthy_scan_crop_norm, label_arr=label_crop)


    healthy_noisy_cuda = torch.from_numpy(healthy_noisy).reshape(1, 1, 96, 96, 96).float().to(device)
    input_cat = torch.cat([healthy_noisy_cuda, label_crop_cuda], dim=1).float()


    with torch.no_grad():
        imgs_recon = generator_t1(input_cat)


    imgs_recon = imgs_recon.detach().cpu().numpy().reshape(96, 96, 96)
    imgs_recon_corrected = correct_background(healthy_crop_pad=healthy_scan_crop_norm, imgs_recon=imgs_recon)


    untouch_x_axis, untouch_y_axis, untouch_z_axis = get_inten_coord(
        healthy_scan_crop=healthy_scan_crop,
        original_label_crop=original_label_crop,
        noise=noise,
    )


    final_recons = linear_interpolation(
        final_recons=imgs_recon_corrected,
        healthy_scan_crop=healthy_scan_crop,
        untouch_x_axis=untouch_x_axis,
        untouch_y_axis=untouch_y_axis,
        untouch_z_axis=untouch_z_axis,
    )


    scan_final = np.copy(healthy_scan)
    scan_final[x_min:x_max, y_min:y_max, z_min:z_max] = final_recons


    complete_label_crop = np.copy(label_crop) + np.copy(original_label_crop)
    label_final = np.copy(original_label)
    label_final[x_min:x_max, y_min:y_max, z_min:z_max] = complete_label_crop


    return scan_final, label_final, {
        "center": [int(x), int(y), int(z)],
        "crop_bounds": {
            "x_min": int(x_min), "x_max": int(x_max),
            "y_min": int(y_min), "y_max": int(y_max),
            "z_min": int(z_min), "z_max": int(z_max),
        },
        "label_crop_nonzero_voxels": int(np.sum(label_crop > 0)),
        "synthetic_label_nonzero_voxels_padded": int(np.sum(label_final > 0)),
    }




# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T1-only GliGAN inference with unique IXI-BraTS pairings.")
    parser.add_argument("--healthy-dir", type=Path, required=True, help="Folder with healthy T1 NIfTI files.")
    parser.add_argument("--label-dir", type=Path, required=True, help="Folder with BraTS seg NIfTI files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder to save generated outputs.")
    parser.add_argument("--generator-path", type=Path, required=True, help="Path to T1 generator checkpoint (.pt).")
    parser.add_argument("--dataset", type=str, default="BRATS_2024", choices=["BRATS_2023", "BRATS_2024"], help="BraTS label transform version.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-size", type=int, default=48)
    parser.add_argument("--in-channels-tumour", type=int, default=5, help="1 scan channel + 3 label channels.")
    parser.add_argument("--out-channels-tumour", type=int, default=1)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"], help="Execution device.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite already completed generated cases.")


    parser.add_argument(
        "--pairing-mode",
        type=str,
        default="shuffle-labels",
        choices=["shuffle-labels", "shuffle-both", "sorted"],
        help=(
            "shuffle-labels: healthy files sorted, labels shuffled by seed. "
            "shuffle-both: both healthy and labels shuffled. "
            "sorted: one-to-one sorted order."
        ),
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap for debugging. Uses only the first N planned pairs.",
    )
    return parser.parse_args()




def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = setup_logger(args.output_dir)


    log("============================================================")
    log("GliGAN UNIQUE-PAIR INFERENCE STARTED")
    log("============================================================")
    log(f"Log file: {log_file}")
    log(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    log("")
    log("Arguments:")
    for k, v in vars(args).items():
        log(f"  {k}: {v}")
    log("")


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device("cuda")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


    if args.dataset == "BRATS_2024":
        label_transform = ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024
    else:
        label_transform = ConvertToMultiChannelBasedOnBratsGliomaClasses2023


    healthy_files = find_nifti_files(args.healthy_dir)
    label_files = find_nifti_files(args.label_dir)


    if len(healthy_files) == 0:
        raise FileNotFoundError(f"No healthy T1 files found in: {args.healthy_dir}")
    if len(label_files) == 0:
        raise FileNotFoundError(f"No BraTS label files found in: {args.label_dir}")


    log(f"Healthy T1 files found: {len(healthy_files)}")
    log(f"BraTS label files found: {len(label_files)}")
    log(f"Using device: {device}")
    log(f"Pairing mode: {args.pairing_mode}")
    log(f"Seed: {args.seed}")
    log("")


    pairs = make_unique_pairs(
        healthy_files=healthy_files,
        label_files=label_files,
        seed=args.seed,
        pairing_mode=args.pairing_mode,
    )


    if args.max_cases is not None:
        pairs = pairs[:args.max_cases]
        log(f"DEBUG: max_cases active -> using first {len(pairs)} planned pairs.")


    pairings_csv = args.output_dir / "ixi_brats_pairings.csv"
    write_pairings_csv(pairs, pairings_csv)
    log(f"Saved planned one-to-one pairings BEFORE generation: {pairings_csv}")
    log("First 10 planned pairs:")
    for idx, healthy_path, label_path in pairs[:10]:
        log(f"  [{idx}] {clean_stem_from_path(healthy_path)}  <->  {clean_stem_from_path(label_path)}")
    log("")


    generator_t1 = get_generator(
        model_path=args.generator_path,
        in_channels=args.in_channels_tumour,
        out_channels=args.out_channels_tumour,
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
        device=device,
    )
    log(f"Loaded T1 generator from: {args.generator_path}")
    log("")


    skipped = 0
    generated = 0
    failed = 0


    total = len(pairs)


    for pair_idx, healthy_path, label_path in pairs:
        stem_h = clean_stem_from_path(healthy_path)
        stem_l = clean_stem_from_path(label_path)
        case_dir = case_dir_for_pair(healthy_path, label_path, args.output_dir)


        log("------------------------------------------------------------")
        log(f"[{pair_idx}/{total}] Pair:")
        log(f"  IXI:   {stem_h}")
        log(f"  BraTS: {stem_l}")
        log(f"  Case:  {case_dir.name}")


        case_dir.mkdir(parents=True, exist_ok=True)


        # Metadata is written immediately, before any possible failure.
        write_metadata(
            case_dir=case_dir,
            status="STARTED",
            pair_index=pair_idx,
            healthy_path=healthy_path,
            label_path=label_path,
            args=args,
            device=device,
        )
        update_pairing_status(pairings_csv, pair_idx, "STARTED", "")


        if case_is_success_complete(case_dir) and not args.overwrite:
            msg = "Already complete and --overwrite not set."
            log(f"  SKIPPED: {msg}")
            write_metadata(
                case_dir=case_dir,
                status="SKIPPED_ALREADY_COMPLETE",
                pair_index=pair_idx,
                healthy_path=healthy_path,
                label_path=label_path,
                args=args,
                device=device,
                error=msg,
            )
            update_pairing_status(pairings_csv, pair_idx, "SKIPPED_ALREADY_COMPLETE", msg)
            skipped += 1
            continue


        try:
            log("  Loading healthy image...")
            healthy_scan_raw, healthy_nii = load_nifti(healthy_path)
            healthy_scan, healthy_pad = pad_to_shape(healthy_scan_raw, target_shape=(256, 256, 256))
            original_label = np.zeros_like(healthy_scan, dtype=np.float32)


            log("  Loading assigned BraTS mask...")
            new_label_raw, _ = load_nifti(label_path)
            new_label = pad_to_shape(new_label_raw, target_shape=(256, 256, 256))[0]


            if np.sum(new_label > 0) == 0:
                raise ValueError("Assigned BraTS mask is empty after loading/padding.")


            log("  Sampling tumor placement center...")
            x, y, z = random_center_tumour_voxel(healthy_scan=healthy_scan, original_label=original_label)


            if x is None:
                raise RuntimeError("No valid tumor placement center found after 1000 attempts.")


            log(f"  Center: x={x}, y={y}, z={z}")
            log("  Running GliGAN generator...")


            synthetic_scan_padded, synthetic_label_padded, meta = insert_tumour_t1(
                generator_t1=generator_t1,
                healthy_scan=healthy_scan,
                original_label=original_label,
                new_label=new_label,
                x=x, y=y, z=z,
                label_transform=label_transform,
                device=device,
            )


            synthetic_scan = crop_from_pad(synthetic_scan_padded, healthy_pad)
            synthetic_label = crop_from_pad(synthetic_label_padded, healthy_pad)


            synthetic_label_voxels = int(np.sum(synthetic_label > 0))
            if synthetic_label_voxels == 0:
                raise RuntimeError("Generated synthetic_seg is empty after crop_from_pad.")


            log("  Saving outputs...")
            save_nifti(healthy_scan_raw, healthy_nii, case_dir / "healthy_t1.nii.gz")
            save_nifti(synthetic_scan, healthy_nii, case_dir / "synthetic_t1.nii.gz")
            save_label_nifti(synthetic_label, healthy_nii, case_dir / "synthetic_seg.nii.gz")


            final_meta = {
                **meta,
                "healthy_shape_raw": list(map(int, healthy_scan_raw.shape)),
                "label_shape_raw": list(map(int, new_label_raw.shape)),
                "synthetic_shape": list(map(int, synthetic_scan.shape)),
                "synthetic_label_nonzero_voxels": synthetic_label_voxels,
                "outputs": {
                    "healthy_t1": str(case_dir / "healthy_t1.nii.gz"),
                    "synthetic_t1": str(case_dir / "synthetic_t1.nii.gz"),
                    "synthetic_seg": str(case_dir / "synthetic_seg.nii.gz"),
                },
            }


            write_metadata(
                case_dir=case_dir,
                status="SUCCESS",
                pair_index=pair_idx,
                healthy_path=healthy_path,
                label_path=label_path,
                args=args,
                device=device,
                extra=final_meta,
            )
            update_pairing_status(pairings_csv, pair_idx, "SUCCESS", "")


            generated += 1
            log(f"  SUCCESS: saved to {case_dir}")


        except Exception as e:
            failed += 1
            err = f"{type(e).__name__}: {e}"
            log_exception("  FAILED", e)


            write_metadata(
                case_dir=case_dir,
                status="FAILED",
                pair_index=pair_idx,
                healthy_path=healthy_path,
                label_path=label_path,
                args=args,
                device=device,
                error=err,
            )
            update_pairing_status(pairings_csv, pair_idx, "FAILED", err)
            continue


    log("")
    log("============================================================")
    log("DONE")
    log("============================================================")
    log(f"generated={generated}")
    log(f"skipped={skipped}")
    log(f"failed={failed}")
    log(f"planned_pairs={len(pairs)}")
    log(f"pairings_csv={pairings_csv}")
    log(f"log_file={log_file}")




if __name__ == "__main__":
    main()






