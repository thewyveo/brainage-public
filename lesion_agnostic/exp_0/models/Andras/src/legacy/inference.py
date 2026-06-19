#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path


import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


from src.multi_head import MultiTaskBrainAge
from src.dataset import BADataset




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
    if str(labels_file).endswith(".xlsx"):
        df_labels = pd.read_excel(labels_file)
    else:
        df_labels = pd.read_csv(labels_file)


    df_labels.columns = [str(c).strip() for c in df_labels.columns]


    id_col = "BraTS Subject ID"
    age_col = "Patient's Age"


    if id_col not in df_labels.columns:
        raise KeyError(f"Missing column '{id_col}' in labels file. Found: {df_labels.columns.tolist()}")
    if age_col not in df_labels.columns:
        raise KeyError(f"Missing column '{age_col}' in labels file. Found: {df_labels.columns.tolist()}")


    age_lookup = dict(zip(df_labels[id_col], df_labels[age_col]))


    nii_files = sorted(data_dir.glob("*.nii.gz"))


    rows = []
    file_paths = []
    ages = []


    for nii in nii_files:
        name = nii.name.replace("_preprocessed", "")
        subject_id = name.split("-t1")[0]
        subject_id = subject_id.replace(".nii.gz", "").replace(".nii", "")


        if subject_id not in age_lookup:
            print(f"Skipping {subject_id} (no age found)")
            continue


        age = float(age_lookup[subject_id])


        rows.append({
            "BraTS Subject ID": subject_id,
            "Patient's Age": age,
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
    return df, file_paths, ages




def create_dataloader(file_paths, ages):
    # Match the original dataset behavior for MRI-input models:
    # crop=True, normalize=True, no special transform
    dataset = BADataset(
    file_paths=file_paths,
    age_labels=ages,
    mode="test",
    transform=None,
    crop_size=(160, 192, 160),
    clamp=True,
    normalize=False,   # important
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
        for batch_data in dataloader:
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


    # Adjust these paths if needed
    labels_path = project_root / "Andras" / "data" / "labels" / "BraTS_24.xlsx"
    predictions_path = project_root / "Andras" / "data" / "predictions" / "BraTS_24_predictions.csv"
    data_dir = project_root / "Andras" / "data" / "preprocessed" / "BraTS"


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


    chronological_age = df["Patient's Age"].values.astype(np.float32)
    brain_age = predicted_ages.astype(np.float32)
    bag = brain_age - chronological_age


    df["Predicted_Brain_Age"] = brain_age
    df["Brain_Age_Difference"] = bag


    out_file = predictions_path
    df.to_csv(out_file, index=False)


    print(f"Saved predictions to: {out_file}")




if __name__ == "__main__":
    main()

