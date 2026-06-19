#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# quick and dirty inference script, no labels, just terminal output

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dp_model.model_files.sfcn import SFCN
from dp_model import dp_utils as dpu


class SFCNInferenceDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = [Path(p) for p in image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        data = nib.load(str(image_path)).get_fdata().astype(np.float32)

        mean_val = data.mean()
        if mean_val == 0:
            raise ValueError(f"Mean intensity is zero for image: {image_path}")

        # SFCN preprocessing from author example
        data = data / mean_val
        data = dpu.crop_center(data, (160, 192, 160))

        # shape: (C, X, Y, Z) = (1, 160, 192, 160)
        data = np.expand_dims(data, axis=0)
        data = torch.tensor(data, dtype=torch.float32)

        return {
            "image": data,
            "path": str(image_path),
        }


def find_images(input_path: Path):
    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    nii_files = sorted(input_path.glob("*.nii.gz"))
    if len(nii_files) == 0:
        raise FileNotFoundError(f"No .nii.gz files found in: {input_path}")

    return nii_files


def create_dataloader(image_paths):
    dataset = SFCNInferenceDataset(image_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )
    return dataloader


def initialize_model(model_path: Path, device: torch.device):
    model = SFCN()

    checkpoint = torch.load(model_path, map_location=device)

    try:
        model.load_state_dict(checkpoint)
    except RuntimeError:
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith("module."):
                new_state_dict[k[len("module."):]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    model = model.to(device)
    model.eval()
    return model


def run_predictions(model_path: Path, dataloader, device: torch.device):
    model = initialize_model(model_path, device)

    bin_range = [42, 82]
    bin_step = 1
    sigma = 1

    _, bc = dpu.num2vect(np.array([50.0]), bin_range, bin_step, sigma)
    bc = np.array(bc).reshape(-1)

    results = []

    with torch.no_grad():
        for batch_data in dataloader:
            images = batch_data["image"].to(device)
            output = model(images)

            if isinstance(output, (list, tuple)):
                output = output[0]

            x = output.cpu().numpy().reshape(-1)
            prob = np.exp(x)
            pred = float(prob @ bc)

            image_path = batch_data["path"][0]
            results.append((image_path, pred))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Quick SFCN inference without labels.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a .nii.gz file or a folder containing .nii.gz files.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to the SFCN checkpoint. If omitted, uses the default repo path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.model_path is None:
        model_path = script_dir / "brain_age" / "run_20190719_00_epoch_best_mae.p"
    else:
        model_path = args.model_path

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths = find_images(args.input)
    dataloader = create_dataloader(image_paths)
    results = run_predictions(model_path, dataloader, device)

    print(f"Using device: {device}")
    print(f"Found {len(results)} image(s)\n")

    for image_path, pred_age in results:
        print(f"{Path(image_path).name}: predicted brain age = {pred_age:.2f}")


if __name__ == "__main__":
    main()