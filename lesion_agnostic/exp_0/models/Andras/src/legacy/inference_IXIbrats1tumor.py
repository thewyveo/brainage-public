#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.multi_head import MultiTaskBrainAge
from src.dataset import BADataset


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


def load_checkpoint_weights(model: torch.nn.Module, checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state_dict[key[len("module."):]] = value
        else:
            cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    return model


def initialize_model(device: torch.device):
    model = MultiTaskBrainAge(
        n_classes=33,
        encoder_chs=(24, 48, 96, 192, 384),
    )
    return model.to(device)


def load_data(labels_file: Path, data_dir: Path):
    # IXI_clean.xls is actually tab-separated text
    df_labels = pd.read_csv(labels_file, sep="\t")
    df_labels.columns = [str(c).strip() for c in df_labels.columns]

    id_col = "IXI_ID"
    age_col = "AGE"

    if id_col not in df_labels.columns:
        raise KeyError(f"Missing column '{id_col}' in labels file. Found: {df_labels.columns.tolist()}")
    if age_col not in df_labels.columns:
        raise KeyError(f"Missing column '{age_col}' in labels file. Found: {df_labels.columns.tolist()}")

    df_labels[id_col] = df_labels[id_col].apply(normalize_ixi_id)
    valid_ids = set(df_labels[id_col].dropna())

    print(f"Valid IXI IDs: {len(valid_ids)}")
    print(f"Example IDs: {list(valid_ids)[:10]}")

    # recursive: case folders / synthetic_t1 files inside
    nii_files = list(data_dir.rglob("*_synthetic_t1.nii.gz"))
    print(f"Found total synthetic scans: {len(nii_files)}")

    rows = []
    file_paths = []
    ages = []
    missing_files = 0

    for nii in tqdm(nii_files, desc="Indexing files", unit="scan"):
        if not nii.exists():
            print(f"[MISSING FILE] {nii}")
            missing_files += 1
            continue

        # example:
        # IXI002-Guys-0828-T1_preprocessed_synthetic_t1.nii.gz
        # or
        # IXI002-Guys-0828-T1_synthetic_t1.nii.gz
        subject_id = nii.name.split("-")[0].strip()
        subject_id = normalize_ixi_id(subject_id)

        if subject_id is None:
            print(f"Skipping malformed filename: {nii.name}")
            continue

        if subject_id not in valid_ids:
            continue

        age_row = df_labels.loc[df_labels[id_col] == subject_id, age_col]
        if len(age_row) == 0:
            continue

        age = float(age_row.values[0])

        rows.append({
            "IXI_ID": subject_id,
            "Age": age,
            "Case_Folder": nii.parent.name,
            "Filename": nii.name,
            "Path": str(nii),
        })

        file_paths.append(str(nii))
        ages.append(age)

    if len(rows) == 0:
        raise ValueError(
            f"No matching subjects found.\n"
            f"Labels file: {labels_file}\n"
            f"Data dir: {data_dir}"
        )

    df = pd.DataFrame(rows)

    print(f"\nFinal dataset size: {len(df)}")
    print(f"Missing files skipped: {missing_files}")

    return df, file_paths, ages


def create_dataloader(file_paths, ages):
    dataset = BADataset(
        file_paths=file_paths,
        age_labels=ages,
        mode="test",
        transform=None,
        crop_size=(160, 192, 160),
        clamp=True,
        normalize=False,   # keep same as your original Joos setup
        crop=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return dataloader


def run_predictions(checkpoint_path: Path, dataloader: DataLoader, device: torch.device):
    model = initialize_model(device)
    model = load_checkpoint_weights(model, checkpoint_path, device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Running inference", total=len(dataloader), unit="scan"):
            images = batch_data["image"].to(device)

            seg_logits, age_pred = model(images)
            preds = age_pred.detach().cpu().numpy().reshape(-1)

            predictions.extend(preds.tolist())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.array(predictions, dtype=np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    labels_path = project_root / "Andras" / "data" / "labels" / "IXI_clean.xls"
    predictions_path = project_root / "Andras" / "data" / "predictions" / "exp0_gligan_brats24_ixi_t1_only1tumor_joos_predictions.csv"
    data_dir = project_root / "Andras" / "data" / "exp0_gligan_brats24_ixi_t1_only1tumor"
    checkpoint_path = project_root / "Andras" / "two_step_final_best_mae.pt"

    print(f"Device: {device}")
    print(f"Labels: {labels_path}")
    print(f"Data dir: {data_dir}")
    print(f"Checkpoint: {checkpoint_path}")

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    df, file_paths, ages = load_data(labels_path, data_dir)
    dataloader = create_dataloader(file_paths, ages)

    predicted_ages = run_predictions(checkpoint_path, dataloader, device)

    chronological_age = df["Age"].values.astype(np.float32)
    brain_age = predicted_ages.astype(np.float32)
    bag = brain_age - chronological_age

    df["Predicted_Brain_Age"] = brain_age
    df["Brain_Age_Difference"] = bag

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(predictions_path, index=False)

    print(f"Saved predictions to: {predictions_path}")


if __name__ == "__main__":
    main()