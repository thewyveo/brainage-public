#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
py -3.10 exp_0\synth_lesion_generator\GliGAN\src\infer\t1_only_brats_scale.py `
    --healthy-dir data\preprocessed\GliGAN\IXI `
    --label-dir data\library\BraTS_Masks `
    --output-dir exp_0\synth_lesion_generator\GliGAN\data\SCALED2_generated_t1_gligan `
    --generator-path exp_0\synth_lesion_generator\GliGAN\Checkpoint\brats2024\t1\weights\generator_457870.pt `
    --dataset BRATS_2024 `
    --num-healthy 30 `
    --num-labels 30 `
    --healthy-select first `
    --label-select first `
    --scales 0.25 0.5 0.75 1 1.25 1.5 1.75 2 `
    --device cpu
"""

from __future__ import annotations


import argparse
import csv
import json
import random
import sys
from pathlib import Path


import nibabel as nib
import numpy as np
import torch
from monai.networks.nets.swin_unetr import SwinUNETR
from scipy.ndimage import label as cc_label
from scipy.ndimage import zoom as ndi_zoom


SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SRC_DIR))


from utils.convert_to_multi_channel_based_on_brats_classes import (
    ConvertToMultiChannelBasedOnBratsGliomaClasses2023,
    ConvertToMultiChannelBasedOnBratsGliomaPosTreatClasses2024,
)




# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------


def log(msg: str):
    print(msg, flush=True)




def find_nifti_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and (p.name.endswith(".nii") or p.name.endswith(".nii.gz")):
            files.append(p)
    return sorted(files)




def find_seg_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.endswith("-seg.nii") or p.name.endswith("-seg.nii.gz"):
            files.append(p)
    return sorted(files)




def strip_nii(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name




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




def sanitize_scale(scale: float) -> str:
    s = f"{scale:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")




def select_subset(files: list[Path], num: int | None, mode: str, rng: random.Random) -> list[Path]:
    if num is None or num >= len(files):
        return list(files)
    if mode == "first":
        return files[:num]
    if mode == "random":
        return sorted(rng.sample(files, num))
    raise ValueError(f"Unknown selection mode: {mode}")




# -----------------------------------------------------------------------------
# Label scaling helpers
# -----------------------------------------------------------------------------


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




def scale_label_mask_full_volume(label_arr: np.ndarray, scale: float) -> np.ndarray:
    """
    Scale only the nonzero tumor region around its own bbox center,
    then paste it back into an otherwise empty volume of the same shape.
    Uses nearest-neighbor interpolation to preserve discrete labels.
    """
    if scale <= 0:
        raise ValueError(f"Scale must be > 0, got {scale}")


    bbox = compute_label_bbox(label_arr)
    if bbox is None:
        raise ValueError("Provided label is empty.")


    x0, x1 = bbox["x_extreme_min"], bbox["x_extreme_max"]
    y0, y1 = bbox["y_extreme_min"], bbox["y_extreme_max"]
    z0, z1 = bbox["z_extreme_min"], bbox["z_extreme_max"]


    crop = label_arr[x0:x1, y0:y1, z0:z1]
    if crop.size == 0:
        raise ValueError("Empty bbox crop for label.")


    scaled = ndi_zoom(crop, zoom=(scale, scale, scale), order=0)
    if scaled.size == 0 or np.count_nonzero(scaled) == 0:
        raise ValueError(f"Scaled label became empty at scale={scale}")


    out = np.zeros_like(label_arr, dtype=np.float32)


    # original bbox center
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    cz = (z0 + z1) // 2


    sx, sy, sz = scaled.shape
    nx0 = cx - sx // 2
    ny0 = cy - sy // 2
    nz0 = cz - sz // 2
    nx1 = nx0 + sx
    ny1 = ny0 + sy
    nz1 = nz0 + sz


    # clip to volume bounds
    vx0 = max(nx0, 0)
    vy0 = max(ny0, 0)
    vz0 = max(nz0, 0)
    vx1 = min(nx1, out.shape[0])
    vy1 = min(ny1, out.shape[1])
    vz1 = min(nz1, out.shape[2])


    if vx0 >= vx1 or vy0 >= vy1 or vz0 >= vz1:
        raise ValueError(f"Scaled label is out of bounds at scale={scale}")


    sx0 = vx0 - nx0
    sy0 = vy0 - ny0
    sz0 = vz0 - nz0
    sx1 = sx0 + (vx1 - vx0)
    sy1 = sy0 + (vy1 - vy0)
    sz1 = sz0 + (vz1 - vz0)


    out[vx0:vx1, vy0:vy1, vz0:vz1] = scaled[sx0:sx1, sy0:sy1, sz0:sz1]
    return out.astype(np.float32)




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
        if healthy_scan_crop[x_axis, y_axis, z_axis] != 0 and original_label_crop[x_axis, y_axis, z_axis] == 0 and noise[x_axis, y_axis, z_axis] == 0:
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




def random_center_tumour_voxel(healthy_scan: np.ndarray, original_label: np.ndarray, rng: np.random.Generator):
    for _ in range(1000):
        x_axis, y_axis = rng.integers(low=0, high=256, size=2, dtype=int)
        z_axis = int(rng.integers(low=0, high=256, size=1, dtype=int)[0])


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


    label_x_min = max(bbox["x_extreme_min"] - 1, 0)
    label_x_max = bbox["x_extreme_max"]
    label_y_min = max(bbox["y_extreme_min"] - 1, 0)
    label_y_max = bbox["y_extreme_max"]
    label_z_min = max(bbox["z_extreme_min"] - 1, 0)
    label_z_max = bbox["z_extreme_max"]


    x_size = label_x_max - label_x_min
    y_size = label_y_max - label_y_min
    z_size = label_z_max - label_z_min


    if x_size > 96 or y_size > 96 or z_size > 96:
        raise ValueError(f"Label bbox too large for 96^3 crop: x={x_size}, y={y_size}, z={z_size}")


    label_crop = np.copy(label_arr[label_x_min:label_x_max, label_y_min:label_y_max, label_z_min:label_z_max])


    def padding_need(size):
        total = 96 - size
        base = total // 2
        top = total - base
        return base, top


    x_base_pad, x_top_pad = padding_need(x_size)
    y_base_pad, y_top_pad = padding_need(y_size)
    z_base_pad, z_top_pad = padding_need(z_size)


    label_crop = np.pad(
        np.copy(label_crop),
        pad_width=((x_base_pad, x_top_pad), (y_base_pad, y_top_pad), (z_base_pad, z_top_pad)),
        mode="constant",
        constant_values=0,
    )


    label_crop = correct_label(np.copy(label_crop), healthy_scan_crop, original_label_crop)


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
    }




# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="T1-only GliGAN inference with explicit scale sweep.")
    parser.add_argument("--healthy-dir", type=Path, required=True, help="Folder with healthy T1 NIfTI files.")
    parser.add_argument("--label-dir", type=Path, required=True, help="Folder with BraTS seg NIfTI files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Folder to save generated outputs.")
    parser.add_argument("--generator-path", type=Path, required=True, help="Path to T1 generator checkpoint (.pt).")
    parser.add_argument("--dataset", type=str, default="BRATS_2024", choices=["BRATS_2023", "BRATS_2024"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-size", type=int, default=48)
    parser.add_argument("--in-channels-tumour", type=int, default=5)
    parser.add_argument("--out-channels-tumour", type=int, default=1)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])


    # New experiment controls
    parser.add_argument("--num-healthy", type=int, default=None, help="Number n of healthy MRIs to use.")
    parser.add_argument("--num-labels", type=int, default=None, help="Number k of donor tumor masks to use.")
    parser.add_argument("--healthy-select", type=str, default="first", choices=["first", "random"])
    parser.add_argument("--label-select", type=str, default="first", choices=["first", "random"])
    parser.add_argument("--scales", type=float, nargs="+", required=True, help="Scale sweep, e.g. --scales 0.2 0.5 0.8")
    parser.add_argument("--skip-existing", action="store_true", help="Skip case if output folder already exists.")


    args = parser.parse_args()


    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)


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
    label_files = find_seg_files(args.label_dir)


    if len(healthy_files) == 0:
        raise FileNotFoundError(f"No healthy T1 files found in: {args.healthy_dir}")
    if len(label_files) == 0:
        raise FileNotFoundError(f"No BraTS seg files found in: {args.label_dir}")


    healthy_files = select_subset(healthy_files, args.num_healthy, args.healthy_select, py_rng)
    label_files = select_subset(label_files, args.num_labels, args.label_select, py_rng)


    log(f"Selected healthy T1 files: {len(healthy_files)}")
    log(f"Selected BraTS seg files: {len(label_files)}")
    log(f"Scales: {args.scales}")
    log(f"Total target outputs: {len(healthy_files) * len(label_files) * len(args.scales)}")
    log(f"Using device: {device}")


    generator_t1 = get_generator(
        model_path=args.generator_path,
        in_channels=args.in_channels_tumour,
        out_channels=args.out_channels_tumour,
        feature_size=args.feature_size,
        use_checkpoint=args.use_checkpoint,
        device=device,
    )
    log(f"Loaded T1 generator from: {args.generator_path}")


    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"


    manifest_exists = manifest_path.exists()
    manifest_file = manifest_path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        manifest_file,
        fieldnames=[
            "healthy_path",
            "label_path",
            "healthy_name",
            "label_name",
            "scale",
            "case_dir",
            "synthetic_t1_path",
            "synthetic_seg_path",
            "center_x",
            "center_y",
            "center_z",
            "status",
            "error",
        ],
    )
    if not manifest_exists:
        writer.writeheader()


    try:
        combo_idx = 0
        total_combos = len(healthy_files) * len(label_files) * len(args.scales)


        for healthy_path in healthy_files:
            healthy_scan_raw, healthy_nii = load_nifti(healthy_path)
            healthy_scan, healthy_pad = pad_to_shape(healthy_scan_raw, target_shape=(256, 256, 256))


            stem_h = strip_nii(healthy_path.name).replace("_preprocessed", "")


            for label_path in label_files:
                label_raw, _ = load_nifti(label_path)
                label_padded = pad_to_shape(label_raw, target_shape=(256, 256, 256))[0]
                stem_l = strip_nii(label_path.name).replace("-seg", "")


                # keep same placement for this healthy/label pair across all scales
                placement_label = np.zeros_like(healthy_scan, dtype=np.float32)
                x, y, z = random_center_tumour_voxel(healthy_scan=healthy_scan, original_label=placement_label, rng=np_rng)
                if x is None:
                    log(f"[{stem_h} | {stem_l}] no valid placement found, skipping pair")
                    for scale in args.scales:
                        combo_idx += 1
                        writer.writerow({
                            "healthy_path": str(healthy_path),
                            "label_path": str(label_path),
                            "healthy_name": stem_h,
                            "label_name": stem_l,
                            "scale": scale,
                            "case_dir": "",
                            "synthetic_t1_path": "",
                            "synthetic_seg_path": "",
                            "center_x": "",
                            "center_y": "",
                            "center_z": "",
                            "status": "failed",
                            "error": "no valid placement found",
                        })
                    continue


                for scale in args.scales:
                    combo_idx += 1
                    scale_tag = sanitize_scale(scale)
                    case_name = f"{stem_h}__{stem_l}__scale_{scale_tag}"
                    case_dir = args.output_dir / case_name


                    log(f"[{combo_idx}/{total_combos}] healthy={stem_h} label={stem_l} scale={scale}")


                    if args.skip_existing and (case_dir / "synthetic_t1.nii.gz").exists():
                        writer.writerow({
                            "healthy_path": str(healthy_path),
                            "label_path": str(label_path),
                            "healthy_name": stem_h,
                            "label_name": stem_l,
                            "scale": scale,
                            "case_dir": str(case_dir),
                            "synthetic_t1_path": str(case_dir / "synthetic_t1.nii.gz"),
                            "synthetic_seg_path": str(case_dir / "synthetic_seg.nii.gz"),
                            "center_x": x,
                            "center_y": y,
                            "center_z": z,
                            "status": "skipped_existing",
                            "error": "",
                        })
                        continue


                    try:
                        scaled_label = scale_label_mask_full_volume(label_padded, scale=scale)
                        synthetic_scan_padded, synthetic_label_padded, meta = insert_tumour_t1(
                            generator_t1=generator_t1,
                            healthy_scan=healthy_scan,
                            original_label=np.zeros_like(healthy_scan, dtype=np.float32),
                            new_label=scaled_label,
                            x=x, y=y, z=z,
                            label_transform=label_transform,
                            device=device,
                        )


                        synthetic_scan = crop_from_pad(synthetic_scan_padded, healthy_pad)
                        synthetic_label = crop_from_pad(synthetic_label_padded, healthy_pad)


                        case_dir.mkdir(parents=True, exist_ok=True)
                        healthy_out = case_dir / "healthy_t1.nii.gz"
                        synth_out = case_dir / "synthetic_t1.nii.gz"
                        seg_out = case_dir / "synthetic_seg.nii.gz"


                        save_nifti(healthy_scan_raw, healthy_nii, healthy_out)
                        save_nifti(synthetic_scan, healthy_nii, synth_out)
                        save_label_nifti(synthetic_label, healthy_nii, seg_out)


                        save_json(
                            {
                                "healthy_path": str(healthy_path),
                                "label_path": str(label_path),
                                "generator_path": str(args.generator_path),
                                "dataset": args.dataset,
                                "device": str(device),
                                "scale": scale,
                                **meta,
                            },
                            case_dir / "metadata.json",
                        )


                        writer.writerow({
                            "healthy_path": str(healthy_path),
                            "label_path": str(label_path),
                            "healthy_name": stem_h,
                            "label_name": stem_l,
                            "scale": scale,
                            "case_dir": str(case_dir),
                            "synthetic_t1_path": str(synth_out),
                            "synthetic_seg_path": str(seg_out),
                            "center_x": x,
                            "center_y": y,
                            "center_z": z,
                            "status": "ok",
                            "error": "",
                        })


                    except Exception as e:
                        log(f"  failed: {e}")
                        writer.writerow({
                            "healthy_path": str(healthy_path),
                            "label_path": str(label_path),
                            "healthy_name": stem_h,
                            "label_name": stem_l,
                            "scale": scale,
                            "case_dir": str(case_dir),
                            "synthetic_t1_path": "",
                            "synthetic_seg_path": "",
                            "center_x": x,
                            "center_y": y,
                            "center_z": z,
                            "status": "failed",
                            "error": str(e),
                        })


    finally:
        manifest_file.close()


    log("Done.")




if __name__ == "__main__":
    main()

