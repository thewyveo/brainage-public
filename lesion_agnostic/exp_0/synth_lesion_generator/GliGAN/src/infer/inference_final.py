#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations


import argparse
import csv
import json
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path


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




class Tee:
    def __init__(self, *streams):
        self.streams = streams


    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()


    def flush(self):
        for s in self.streams:
            s.flush()




def log(msg: str):
    print(msg, flush=True)




def strip_nii(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name




def sanitize_name(text: str) -> str:
    allowed = []
    for ch in str(text):
        if ch.isalnum() or ch in "-_.":
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_")




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




def append_csv_row(path: Path, fieldnames: list[str], row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()




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




def get_generator(
    model_path: Path,
    in_channels: int,
    out_channels: int,
    feature_size: int,
    use_checkpoint: bool,
    device: torch.device,
):
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




def case_dir_for(healthy_path: Path, label_path: Path, output_dir: Path) -> Path:
    stem_h = sanitize_name(strip_nii(healthy_path.name).replace("_preprocessed", ""))
    stem_l = sanitize_name(strip_nii(label_path.name).replace("-seg", ""))
    return output_dir / f"{stem_h}__{stem_l}"




def case_is_complete(case_dir: Path) -> bool:
    required = [
        case_dir / "healthy_t1.nii.gz",
        case_dir / "synthetic_t1.nii.gz",
        case_dir / "synthetic_seg.nii.gz",
        case_dir / "metadata.json",
    ]
    return all(p.exists() for p in required)




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


    for x_axis in range(96):
        for z_axis in range(96):
            verify(x_axis, 0, z_axis)
            verify(x_axis, 95, z_axis)


    for x_axis in range(96):
        for y_axis in range(96):
            verify(x_axis, y_axis, 0)
            verify(x_axis, y_axis, 95)


    for y_axis in range(96):
        for z_axis in range(96):
            verify(0, y_axis, z_axis)
            verify(95, y_axis, z_axis)


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




def faithful_center_from_label(label_arr: np.ndarray):
    bbox = compute_label_bbox(label_arr)
    if bbox is None:
        raise ValueError("Provided label is empty.")


    x = int(round((bbox["x_extreme_min"] + bbox["x_extreme_max"]) / 2))
    y = int(round((bbox["y_extreme_min"] + bbox["y_extreme_max"]) / 2))
    z = int(round((bbox["z_extreme_min"] + bbox["z_extreme_max"]) / 2))


    x = min(max(x, 48), 208)
    y = min(max(y, 48), 208)
    z = min(max(z, 48), 208)


    return x, y, z




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




def prepar_real_label_to_gen(
    label_arr: np.ndarray,
    healthy_scan_crop: np.ndarray,
    original_label_crop: np.ndarray,
    label_transform,
    device: torch.device,
):
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
        raise ValueError("Label became empty after fitting/correct_label step.")


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
        "used_label_crop_voxels": int(np.sum(label_crop > 0)),
        "final_label_voxels": int(np.sum(label_final > 0)),
    }




ATTEMPT_FIELDS = [
    "timestamp",
    "pair_index",
    "healthy_path",
    "healthy_stem",
    "label_path",
    "label_stem",
    "case_dir",
    "status",
    "error",
    "center",
    "crop_bounds",
    "used_label_crop_voxels",
    "final_label_voxels",
]


SUCCESS_FIELDS = [
    "pair_index",
    "healthy_path",
    "healthy_stem",
    "label_path",
    "label_stem",
    "case_dir",
    "center",
    "crop_bounds",
    "used_label_crop_voxels",
    "final_label_voxels",
]




def read_pairing_csv(path: Path, healthy_col: str, label_col: str) -> list[tuple[Path, Path]]:
    pairs = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if healthy_col not in reader.fieldnames:
            raise ValueError(f"CSV missing healthy column '{healthy_col}'. Found: {reader.fieldnames}")
        if label_col not in reader.fieldnames:
            raise ValueError(f"CSV missing label column '{label_col}'. Found: {reader.fieldnames}")


        for row in reader:
            healthy = str(row[healthy_col]).strip()
            label = str(row[label_col]).strip()
            if healthy and label:
                pairs.append((Path(healthy), Path(label)))


    if len(pairs) == 0:
        raise ValueError(f"No valid pairs found in CSV: {path}")


    return pairs




def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "T1-only GliGAN inference from an existing IXI/BraTS pairing CSV. "
            "No random pairing. No random tumor placement. "
            "Tumor crop center is derived from the BraTS mask's own bbox."
        )
    )


    parser.add_argument("--pairing-csv", type=Path, required=True)
    parser.add_argument("--healthy-col", type=str, default="healthy_path")
    parser.add_argument("--label-col", type=str, default="label_path")


    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generator-path", type=Path, required=True)
    parser.add_argument("--dataset", type=str, default="BRATS_2024", choices=["BRATS_2023", "BRATS_2024"])


    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-size", type=int, default=48)
    parser.add_argument("--in-channels-tumour", type=int, default=5)
    parser.add_argument("--out-channels-tumour", type=int, default=1)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--overwrite", action="store_true")


    parser.add_argument(
        "--save-source-mask-copy",
        action="store_true",
        help="Also copy the original selected BraTS mask into the successful case folder.",
    )


    return parser.parse_args()




def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)


    log_path = args.output_dir / f"gligan_faithful_rerun_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)


    attempts_csv = args.output_dir / "all_attempts.csv"
    successes_csv = args.output_dir / "successful_ixi_brats_pairings.csv"


    try:
        log("=" * 80)
        log("GliGAN faithful rerun from existing CSV")
        log("=" * 80)
        log(f"Started: {datetime.now().isoformat(timespec='seconds')}")
        log(f"Log file: {log_path}")
        log("")


        for k, v in vars(args).items():
            log(f"  {k}: {v}")
        log("")


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


        pairs = read_pairing_csv(
            path=args.pairing_csv,
            healthy_col=args.healthy_col,
            label_col=args.label_col,
        )


        log(f"Pairs loaded from CSV: {len(pairs)}")
        log(f"Using device: {device}")
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


        generated = 0
        skipped_existing = 0
        failed = 0


        for pair_idx, (healthy_path, label_path) in enumerate(pairs, start=1):
            healthy_stem = strip_nii(healthy_path.name).replace("_preprocessed", "")
            label_stem = strip_nii(label_path.name).replace("-seg", "")
            case_dir = case_dir_for(healthy_path, label_path, args.output_dir)


            log("=" * 80)
            log(f"[PAIR {pair_idx}/{len(pairs)}]")
            log(f"IXI:   {healthy_path}")
            log(f"BraTS: {label_path}")
            log(f"Case:  {case_dir}")
            log("=" * 80)


            base_row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "pair_index": pair_idx,
                "healthy_path": str(healthy_path),
                "healthy_stem": healthy_stem,
                "label_path": str(label_path),
                "label_stem": label_stem,
                "case_dir": str(case_dir),
            }


            if case_is_complete(case_dir) and not args.overwrite:
                log("Existing complete case found; skipping.")
                skipped_existing += 1


                row = {
                    **base_row,
                    "status": "skipped_existing_complete",
                    "error": "",
                    "center": "",
                    "crop_bounds": "",
                    "used_label_crop_voxels": "",
                    "final_label_voxels": "",
                }
                append_csv_row(attempts_csv, ATTEMPT_FIELDS, row)
                append_csv_row(successes_csv, SUCCESS_FIELDS, row)
                continue


            try:
                healthy_scan_raw, healthy_nii = load_nifti(healthy_path)
                healthy_scan, healthy_pad = pad_to_shape(healthy_scan_raw, target_shape=(256, 256, 256))


                original_label = np.zeros_like(healthy_scan, dtype=np.float32)


                new_label_raw, _ = load_nifti(label_path)
                new_label = pad_to_shape(new_label_raw, target_shape=(256, 256, 256))[0]


                x, y, z = faithful_center_from_label(new_label)


                synthetic_scan_padded, synthetic_label_padded, meta = insert_tumour_t1(
                    generator_t1=generator_t1,
                    healthy_scan=healthy_scan,
                    original_label=original_label,
                    new_label=new_label,
                    x=x,
                    y=y,
                    z=z,
                    label_transform=label_transform,
                    device=device,
                )


                synthetic_scan = crop_from_pad(synthetic_scan_padded, healthy_pad)
                synthetic_label = crop_from_pad(synthetic_label_padded, healthy_pad)


                case_dir.mkdir(parents=True, exist_ok=True)


                save_nifti(healthy_scan_raw, healthy_nii, case_dir / "healthy_t1.nii.gz")
                save_nifti(synthetic_scan, healthy_nii, case_dir / "synthetic_t1.nii.gz")
                save_label_nifti(synthetic_label, healthy_nii, case_dir / "synthetic_seg.nii.gz")


                if args.save_source_mask_copy:
                    suffix = ".nii.gz" if label_path.name.endswith(".nii.gz") else label_path.suffix
                    shutil.copy2(label_path, case_dir / f"source_brats_seg{suffix}")


                metadata = {
                    "status": "success",
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "pair_index": int(pair_idx),
                    "healthy_path": str(healthy_path),
                    "healthy_stem": healthy_stem,
                    "label_path": str(label_path),
                    "label_stem": label_stem,
                    "case_dir": str(case_dir),
                    "generator_path": str(args.generator_path),
                    "dataset": args.dataset,
                    "device": str(device),
                    "seed": int(args.seed),
                    "pairing_source": str(args.pairing_csv),
                    "random_pairing": False,
                    "random_tumor_placement": False,
                    "placement_mode": "faithful_center_from_label_bbox",
                    **meta,
                }
                save_json(metadata, case_dir / "metadata.json")


                row = {
                    **base_row,
                    "status": "success",
                    "error": "",
                    "center": json.dumps(meta.get("center", "")),
                    "crop_bounds": json.dumps(meta.get("crop_bounds", "")),
                    "used_label_crop_voxels": meta.get("used_label_crop_voxels", ""),
                    "final_label_voxels": meta.get("final_label_voxels", ""),
                }
                append_csv_row(attempts_csv, ATTEMPT_FIELDS, row)
                append_csv_row(successes_csv, SUCCESS_FIELDS, row)


                generated += 1
                log(f"SUCCESS -> {case_dir}")
                log(f"Center from BraTS mask bbox: {meta.get('center')}")
                log(f"Used label voxels: {meta.get('used_label_crop_voxels')}")
                log(f"Final label voxels: {meta.get('final_label_voxels')}")


            except Exception as e:
                failed += 1
                err = repr(e)
                log(f"FAILED: {err}")
                log(traceback.format_exc())


                row = {
                    **base_row,
                    "status": "failed",
                    "error": err,
                    "center": "",
                    "crop_bounds": "",
                    "used_label_crop_voxels": "",
                    "final_label_voxels": "",
                }
                append_csv_row(attempts_csv, ATTEMPT_FIELDS, row)


        log("")
        log("=" * 80)
        log("DONE")
        log("=" * 80)
        log(f"Generated successful cases: {generated}")
        log(f"Skipped existing complete cases: {skipped_existing}")
        log(f"Failed cases: {failed}")
        log("")
        log(f"Attempts CSV: {attempts_csv}")
        log(f"Success pairings CSV: {successes_csv}")
        log(f"Log file: {log_path}")


    finally:
        try:
            log_file.close()
        except Exception:
            pass




if __name__ == "__main__":
    main()


r"""
py -3.10 exp_0\synth_lesion_generator\GliGAN\src\infer\infer3.py `
  --pairing-csv "C:\Users\P102179\Downloads\successful_ixi_brats_pairings.csv" `
  --output-dir data\IXI_GLI_FAITHFUL_RERUN_FINALALALALALAL `
  --generator-path exp_0\synth_lesion_generator\GliGAN\Checkpoint\brats2024\t1\weights\generator_457870.pt `
  --dataset BRATS_2024 `
  --device cpu
"""