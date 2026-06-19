#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import argparse


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




def load_data(
    labels_file: Path,
    data_dir: Path,
    id_col: str,
    age_col: str,
    require_t1_suffix: bool,
    max_subjects: int | None,
):
    if str(labels_file).lower().endswith(".xlsx"):
        df_labels = pd.read_excel(labels_file)
    else:
        df_labels = pd.read_csv(labels_file, sep="\t")


    df_labels.columns = [str(c).strip() for c in df_labels.columns]


    if id_col not in df_labels.columns:
        raise KeyError(f"Missing column '{id_col}' in labels file. Found: {df_labels.columns.tolist()}")


    if age_col not in df_labels.columns:
        raise KeyError(f"Missing column '{age_col}' in labels file. Found: {df_labels.columns.tolist()}")


    df_labels[id_col] = df_labels[id_col].apply(normalize_ixi_id)
    valid_ids = set(df_labels[id_col].dropna())


    print(f"Valid IXI IDs: {len(valid_ids)}")
    print(f"Example IDs: {list(valid_ids)[:10]}")


    nii_files = sorted(list(data_dir.glob("*.nii")) + list(data_dir.glob("*.nii.gz")))
    print(f"Found total MRI files: {len(nii_files)}")


    rows = []
    file_paths = []
    ages = []
    missing_files = 0


    for nii in tqdm(nii_files, desc="Indexing files", unit="scan"):
        if max_subjects is not None and len(rows) >= max_subjects:
            break


        if not nii.exists():
            print(f"[MISSING FILE] {nii}")
            missing_files += 1
            continue


        if require_t1_suffix:
            if not (nii.name.endswith("-T1.nii") or nii.name.endswith("-T1.nii.gz")):
                continue


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




def create_dataloader(file_paths, ages, batch_size: int, num_workers: int):
    dataset = BADataset(
        file_paths=file_paths,
        age_labels=ages,
        mode="test",
        transform=None,
        crop_size=(160, 192, 160),
        clamp=True,
        normalize=True,
        crop=True,
    )


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


    return dataloader




def run_predictions(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, run_idx: int):
    model.eval()
    predictions = []


    with torch.no_grad():
        for batch_data in tqdm(
            dataloader,
            desc=f"Running inference repeat {run_idx}",
            total=len(dataloader),
            unit="batch",
        ):
            images = batch_data["image"].to(device)


            _, age_pred = model(images)
            preds = age_pred.detach().cpu().numpy().reshape(-1)


            predictions.extend(preds.tolist())


    return np.array(predictions, dtype=np.float32)




def parse_args():
    parser = argparse.ArgumentParser(
        description="Run repeated IXI brain-age inference to test model determinism."
    )


    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--predictions-path", type=Path, required=True)


    parser.add_argument("--id-col", type=str, default="IXI_ID")
    parser.add_argument("--age-col", type=str, default="AGE")


    parser.add_argument(
        "--require-t1-suffix",
        action="store_true",
        help="Only keep files ending exactly in -T1.nii or -T1.nii.gz.",
    )


    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])


    parser.add_argument(
        "--num-runs",
        type=int,
        required=True,
        help="Number of times to run inference on the exact same data.",
    )


    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Optional limit for small determinism tests, e.g. --max-subjects 10.",
    )


    return parser.parse_args()




def main():
    args = parse_args()


    if args.num_runs < 2:
        raise ValueError("--num-runs should be at least 2 for a determinism test.")


    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)


    print(f"Device: {device}")
    print(f"Labels: {args.labels_path}")
    print(f"Data dir: {args.data_dir}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Predictions: {args.predictions_path}")
    print(f"Number of repeated runs: {args.num_runs}")
    print(f"Max subjects: {args.max_subjects}")


    if not args.labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {args.labels_path}")


    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")


    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")


    args.predictions_path.parent.mkdir(parents=True, exist_ok=True)


    df, file_paths, ages = load_data(
        labels_file=args.labels_path,
        data_dir=args.data_dir,
        id_col=args.id_col,
        age_col=args.age_col,
        require_t1_suffix=args.require_t1_suffix,
        max_subjects=args.max_subjects,
    )


    dataloader = create_dataloader(
        file_paths=file_paths,
        ages=ages,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


    model = initialize_model(device)
    model = load_checkpoint_weights(model, args.checkpoint_path, device)
    model.eval()


    all_predictions = []


    for run_idx in range(1, args.num_runs + 1):
        predicted_ages = run_predictions(
            model=model,
            dataloader=dataloader,
            device=device,
            run_idx=run_idx,
        )
        all_predictions.append(predicted_ages)


    all_predictions = np.stack(all_predictions, axis=1)


    chronological_age = df["Age"].values.astype(np.float32)


    for run_idx in range(args.num_runs):
        run_number = run_idx + 1
        df[f"Predicted_Brain_Age_Run_{run_number}"] = all_predictions[:, run_idx]
        df[f"Brain_Age_Difference_Run_{run_number}"] = all_predictions[:, run_idx] - chronological_age


    first_run = all_predictions[:, [0]]
    abs_diffs_from_first = np.abs(all_predictions - first_run)


    df["Max_Abs_Prediction_Difference_Across_Runs"] = abs_diffs_from_first.max(axis=1)
    df["Mean_Abs_Prediction_Difference_Across_Runs"] = abs_diffs_from_first.mean(axis=1)


    global_max_diff = float(abs_diffs_from_first.max())
    global_mean_diff = float(abs_diffs_from_first.mean())


    df.to_csv(args.predictions_path, index=False)


    print(f"\nSaved repeated predictions to: {args.predictions_path}")
    print("\nDeterminism summary:")
    print(f"Global max absolute difference across runs:  {global_max_diff:.10f}")
    print(f"Global mean absolute difference across runs: {global_mean_diff:.10f}")


    if global_max_diff == 0.0:
        print("Result: perfectly deterministic across repeated inference runs.")
    else:
        print("Result: predictions changed across repeated inference runs.")


    del model


    if torch.cuda.is_available():
        torch.cuda.empty_cache()




if __name__ == "__main__":
    main()


r"""

$env:PYTHONPATH="exp_0\models\Andras"

py -3.10 exp_0\models\Andras\infer\determinism.py `
--labels-path data\labels\IXI_clean.xls `
--data-dir "\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\preprocessed\JOOS_IXI_PREP" `
--checkpoint-path exp_0\models\Andras\src\two_step_final_best_mae.pt `
--predictions-path data\predictions\Andras\JOOS_IXI_determinism.csv `
--num-runs 10 `
--max-subjects 10 `
--batch-size 1 `
--num-workers 0

py -3.10 exp_0\models\Andras\infer\determinism.py `
--labels-path data\labels\IXI_clean.xls `
--data-dir data\nonpytest `
--checkpoint-path exp_0\models\Andras\src\two_step_final_best_mae.pt `
--predictions-path data\predictions\Andras\JOOS_IXI_determinism.csv `
--num-runs 10 `
--max-subjects 1 `
--batch-size 1 `
--num-workers 0

"""