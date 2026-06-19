#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import nibabel as nib

from UKBiobank_deep_pretrain.dp_model.model_files.sfcn import SFCN
from UKBiobank_deep_pretrain.dp_model import dp_utils as dpu


class SFCNDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image_path = record["image"]
        age = record["label"]

        data = nib.load(str(image_path)).get_fdata().astype(np.float32)

        mean_val = data.mean()
        if mean_val == 0:
            raise ValueError(f"Mean intensity is zero for image: {image_path}")

        # SFCN preprocessing from author example
        data = data / mean_val
        data = dpu.crop_center(data, (160, 192, 160))

        # shape: (C, X, Y, Z)
        data = np.expand_dims(data, axis=0)
        data = torch.tensor(data, dtype=torch.float32)

        return {
            "image": data,
            "label": torch.tensor(age, dtype=torch.float32),
            "path": str(image_path),
        }


def normalize_ixi_id(x):
    s = str(x).strip()

    if s.upper().startswith("IXI"):
        num = s[3:]
    else:
        num = s

    try:
        num = int(float(num))
    except Exception:
        return None

    return f"IXI{num:03d}"


def load_data(labels_file):
    # IXI_clean.xls is actually tab-separated text
    df_labels = pd.read_csv(labels_file, sep="\t")
    df_labels["IXI_ID"] = df_labels["IXI_ID"].apply(normalize_ixi_id)

    valid_ids = set(df_labels["IXI_ID"].dropna())

    print(f"Valid IXI IDs: {len(valid_ids)}")
    print(f"Example IDs: {list(valid_ids)[:10]}")

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data" / "exp0_gligan_brats24_ixi_t1_only1tumor"

    # recursively find synthetic_t1 files inside case folders
    nii_files = list(data_dir.rglob("*_synthetic_t1.nii.gz"))

    print(f"Found total synthetic scans: {len(nii_files)}")

    rows = []
    data_dicts = []
    missing_files = 0

    for nii in nii_files:
        if not nii.exists():
            print(f"[MISSING FILE] {nii}")
            missing_files += 1
            continue

        # example file:
        # IXI002-Guys-0828-T1_synthetic_t1.nii.gz
        subject_id = nii.name.split("-")[0].strip()
        subject_id = normalize_ixi_id(subject_id)

        if subject_id is None:
            print(f"Skipping malformed filename: {nii.name}")
            continue

        if subject_id not in valid_ids:
            continue

        age_row = df_labels.loc[df_labels["IXI_ID"] == subject_id, "AGE"]
        if len(age_row) == 0:
            continue

        age = float(age_row.values[0])

        case_folder = nii.parent.name

        rows.append({
            "IXI_ID": subject_id,
            "Age": age,
            "Case_Folder": case_folder,
            "Filename": nii.name,
            "Path": str(nii),
        })

        data_dicts.append({
            "image": str(nii),
            "label": age,
        })

    df = pd.DataFrame(rows)

    print(f"\nFinal dataset size: {len(df)}")
    print(f"Missing files skipped: {missing_files}")

    return df, data_dicts


def create_dataloader(data_dicts):
    dataset = SFCNDataset(data_dicts)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )
    return dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_model(model_path):
    model = SFCN()

    checkpoint = torch.load(model_path, map_location=device)

    try:
        model.load_state_dict(checkpoint)
    except RuntimeError:
        # if checkpoint was saved from DataParallel, strip "module."
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


def run_predictions(model_path, dataloader):
    model = initialize_model(model_path)

    bin_range = [42, 82]
    bin_step = 1
    sigma = 1

    _, bc = dpu.num2vect(np.array([50.0]), bin_range, bin_step, sigma)
    bc = np.array(bc).reshape(-1)

    predictions = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Running inference", unit="scan"):
            images = batch_data["image"].to(device)
            output = model(images)

            if isinstance(output, (list, tuple)):
                output = output[0]

            x = output.cpu().numpy().reshape(-1)
            prob = np.exp(x)
            pred = prob @ bc

            predictions.append(pred)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.array(predictions)



def main():
    script_dir = Path(__file__).resolve().parent

    labels_path = script_dir.parent / "data" / "labels" / "IXI_clean.xls"
    out_path = script_dir.parent / "data" / "predictions" / "exp0_gligan_brats24_ixi_t1_only1tumor_sfcn_predictions.csv"
    model_path = script_dir / "brain_age" / "run_20190719_00_epoch_best_mae.p"

    print(labels_path)

    df, data_dicts = load_data(labels_path)

    if len(data_dicts) == 0:
        raise RuntimeError("No valid MRI files found after filtering.")

    dataloader = create_dataloader(data_dicts)
    predicted_ages = run_predictions(model_path, dataloader)

    CA = df["Age"].values
    BA = predicted_ages.flatten()
    BAD = BA - CA

    df["Predicted_Brain_Age"] = BA
    df["Brain_Age_Difference"] = BAD

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Updated CSV file saved to: {out_path}")


if __name__ == "__main__":
    main()

