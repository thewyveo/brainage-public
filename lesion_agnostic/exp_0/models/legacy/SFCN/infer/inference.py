#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
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

        # shape should become (C, X, Y, Z) = (1, 160, 192, 160)
        data = np.expand_dims(data, axis=0)
        data = torch.tensor(data, dtype=torch.float32)

        return {
            "image": data,
            "label": torch.tensor(age, dtype=torch.float32),
        }


def load_data(labels_file):
    if str(labels_file).endswith(".xlsx"):
        df_labels = pd.read_excel(labels_file)
    else:
        df_labels = pd.read_csv(labels_file)

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data" / "preprocessed" / "original" / "BraTS" # or "original" for instead of "synthstrip"

    age_lookup = dict(zip(df_labels["BraTS Subject ID"], df_labels["Patient's Age"]))

    nii_files = list(data_dir.glob("*.nii.gz"))

    rows = []
    data_dicts = []

    for nii in nii_files:
        subject_id = nii.name.split("-t1")[0]

        if subject_id not in age_lookup:
            print(f"Skipping {subject_id} (no age found)")
            continue

        age = age_lookup[subject_id]

        rows.append({
            "BraTS Subject ID": subject_id,
            "Patient's Age": age,
            "Path": str(nii),
        })

        data_dicts.append({
            "image": str(nii),
            "label": age,
        })

    df = pd.DataFrame(rows)
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

    # Try direct load first
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError:
        # If checkpoint was saved from DataParallel, strip "module."
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
        for batch_data in dataloader:
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
    labels_path = script_dir.parent / "data" / "labels" / "BraTS_24.xlsx"
    out_path = script_dir.parent / "data" / "predictions" / "BraTS_24_predictions.csv"

    model_path = script_dir / "brain_age" / "run_20190719_00_epoch_best_mae.p"

    df, data_dicts = load_data(labels_path)
    dataloader = create_dataloader(data_dicts)

    predicted_ages = run_predictions(model_path, dataloader)

    CA = df["Patient's Age"].values
    BA = predicted_ages.flatten()
    BAD = BA - CA

    df["Predicted_Brain_Age"] = BA
    df["Brain_Age_Difference"] = BAD

    out_file = out_path.with_name(out_path.stem + "_predictions.csv")
    df.to_csv(out_file, index=False)
    print(f"Updated CSV file saved to: {out_file}")


if __name__ == "__main__":
    main()