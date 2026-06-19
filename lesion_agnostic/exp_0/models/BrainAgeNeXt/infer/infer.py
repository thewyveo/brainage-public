#!/usr/bin/env python3
# -*- coding: utf-8 -*-


r"""
Repeat BrainAgeNeXt inference on a folder of MRI scans.


Example:
py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
  --input-dir "data\preprocessed\BrainAgeNeXt\IXI_10" `
  --labels-file "data\labels\IXI_clean.xls" `
  --output-dir "exp_0\results\BANXt_determinism_proof" `
  --runs 10 `
  --checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "data\preprocessed\BrainAgeNeXt\1_test" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "exp_0\results\1test" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "data\preprocessed\BrainAgeNeXt\BraTS_T1_BANeXt" `
--labels-file "data\labels\BraTS_24.xls" `
--output-dir "data\predictions\BrainAgeNeXt\BraTS_T1" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "data\preprocessed\BrainAgeNeXt\USB" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "exp_0\results\USB" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "data\brainid_gli_banextprepped" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "data\predictions\BrainAgeNeXt\BNX_GLI_BID" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "data\BNX_GLI_USB_PREP\BNX_GLI_USB_PREP" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "data\predictions\BrainAgeNeXt\BNX_GLI_USB" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "data\BNX_CM_PREP" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "data\predictions\BrainAgeNeXt\BNX_CM" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "data\BNX_GLI_LIT_PREP" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "data\predictions\BrainAgeNeXt\BNX_GLI_LIT" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "C:\Users\P102179\Downloads\IXI_GLI_n4_rigid" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "data\predictions\BrainAgeNeXt\BNX_GLI_rerun" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "C:\Users\P102179\Downloads\BNX_CM_PREP" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "data\predictions\BrainAgeNeXt\BNX_CM_rerun" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\rerun\prep\BNX_GLI_BID_PREP_rerun\BNX_GLI_BID_PREP_rerun" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "data\predictions\BrainAgeNeXt\BNX_GLI_BID_rerun" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "C:\Users\P102179\Downloads\BNX_USB_PREP" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "data\predictions\BrainAgeNeXt\BNX_USB_rerun" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "C:\Users\P102179\Downloads\IXI_USB_n4_rigid" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "data\predictions\BrainAgeNeXt\realreruns\IXI_USB_rerun" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\rerun\prep\BNX_IXI_CM_PREP" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "data\predictions\BrainAgeNeXt\realreruns\IXI_CM_rerun" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\rerun\prep\BNX_IXI_GLI_PREP" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "data\predictions\BrainAgeNeXt\realreruns\IXI_GLI_rerun" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\rerun\prep\BNX_GLI_LIT_PREP" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "data\predictions\BrainAgeNeXt\realreruns\GLI_LIT_rerun" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\rerun\prep\BNX_CM_LIT_PREP" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "data\predictions\BrainAgeNeXt\realreruns\CM_LIT_rerun" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "C:\Users\P102179\Downloads\BNX_CM_BID_PREP" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "data\predictions\BrainAgeNeXt\realreruns\CM_BID_rerun" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\rerun\prep\BNX_CM_USB_PREP" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "data\predictions\BrainAgeNeXt\realreruns\CM_USB_rerun" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "C:\Users\P102179\Downloads\BNX_GLI_BID_PREP" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "data\predictions\BrainAgeNeXt\realreruns\GLI_BID_rerun" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

py -3.10 exp_0\models\BrainAgeNeXt\infer\determinism_proof.py `
--input-dir "C:\Users\P102179\Downloads\BNX_GLI_USB_PREP" `
--labels-file "data\labels\IXI_clean.xls" `
--output-dir "data\predictions\BrainAgeNeXt\realreruns\GLI_USB_rerun" `
--runs 1 `
--checkpoints-dir "exp_0\models\BrainAgeNeXt\checkpoints"

-----------------------
Linux/macOS example:
python run_brainagenext_repeated.py \
  --input-dir /data/ten_scans \
  --labels-file /data/labels/IXI_clean.xls \
  --output-dir /data/repeatability_runs \
  --runs 5 \
  --checkpoints-dir /repo/lesion_agnostic/exp_0/BrainAgeNeXt/checkpoints
"""


from __future__ import annotations


import sys
import argparse
from pathlib import Path


import pandas as pd
import torch
from torch.utils.data import DataLoader
from monai.transforms import Compose, LoadImaged, Spacingd, CropForegroundd, SpatialPadd, CenterSpatialCropd
from monai.data import CacheDataset
import numpy as np
import torchio
import torch.nn as nn
from tqdm import tqdm
import re




_SCRIPT_DIR = Path(__file__).resolve().parent
_MEDNEXT_ROOT = _SCRIPT_DIR.parents[0] / "MedNeXt"
if _MEDNEXT_ROOT.exists():
    sys.path.insert(0, str(_MEDNEXT_ROOT))


from nnunet_mednext import create_mednext_encoder_v1  # pyright: ignore[reportMissingImports]




class MedNeXtEncReg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mednextv1 = create_mednext_encoder_v1(
            num_input_channels=1,
            num_classes=1,
            model_id="B",
            kernel_size=3,
            deep_supervision=True,
        )
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.regression_fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(64, 1),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mednextv1(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        age_estimate = self.regression_fc(x)
        return age_estimate.squeeze()




def prepare_transforms():
    x, y, z = (160, 192, 160)
    p = 1.0


    monai_transforms = [
        LoadImaged(keys=["image"], ensure_channel_first=True),
        Spacingd(keys=["image"], pixdim=(p, p, p)),
        CropForegroundd(keys=["image"], allow_smaller=True, source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(x, y, z)),
        CenterSpatialCropd(keys=["image"], roi_size=(x, y, z)),
    ]


    val_torchio_transforms = torchio.transforms.Compose(
        [torchio.transforms.ZNormalization(masking_method=lambda x: x > 0, include=["image"])]
    )


    return Compose(monai_transforms + [val_torchio_transforms])




def normalize_ixi_id(x) -> str | None:
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




def extract_ixi_id_from_path(nii: Path) -> str | None:
    candidates = [nii.name, nii.parent.name]


    for text in candidates:
        m = re.search(r"IXI(\d{3,4})", text)
        if m:
            return f"IXI{int(m.group(1)):03d}"


    return None




def is_valid_mri_file(nii: Path) -> bool:
    name = nii.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")




def load_data(labels_file: Path, input_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
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


    nii_files = sorted([p for p in input_dir.rglob("*") if p.is_file() and is_valid_mri_file(p)])


    rows = []
    data_dicts = []


    for nii in tqdm(nii_files, desc="Indexing files", unit="scan"):
        subject_id = extract_ixi_id_from_path(nii)
        if subject_id is None:
            print(f"[SKIP] Could not parse IXI ID from: {nii}")
            continue


        if subject_id not in valid_ids:
            print(f"[SKIP] Subject ID not found in labels: {subject_id} ({nii.name})")
            continue


        age_row = df_labels.loc[df_labels[id_col] == subject_id, age_col]
        if len(age_row) == 0:
            print(f"[SKIP] No age found for: {subject_id}")
            continue


        age = float(age_row.values[0])


        rows.append(
            {
                "IXI_ID": subject_id,
                "Age": age,
                "Filename": nii.name,
                "Path": str(nii),
            }
        )


        data_dicts.append(
            {
                "image": str(nii),
                "label": age,
            }
        )


    df = pd.DataFrame(rows)


    if len(df) == 0:
        raise RuntimeError(f"No valid scans found in: {input_dir}")


    return df, data_dicts




def create_dataloader(data_dicts: list[dict], transforms):
    dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=0.2, num_workers=0)
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
    )




def initialize_model(device: torch.device) -> nn.Module:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return MedNeXtEncReg().to(device)




def run_predictions_single_checkpoint(model_path: Path, dataloader, device: torch.device) -> np.ndarray:
    model = initialize_model(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()


    predictions = []


    with torch.no_grad():
        for batch_data in tqdm(
            dataloader,
            desc=f"Inference ({model_path.name})",
            total=len(dataloader),
            unit="scan",
        ):
            images = batch_data["image"].to(device)
            pred = model(images)
            pred_np = np.atleast_1d(pred.detach().cpu().numpy()).astype(np.float32)
            predictions.extend(pred_np.tolist())


    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


    return np.array(predictions, dtype=np.float32)




def get_model_paths(checkpoints_dir: Path) -> list[Path]:
    model_paths = [checkpoints_dir / f"BrainAge_{i}.pth" for i in range(1, 6)]
    for p in model_paths:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
    return model_paths




def correct_predictions(ca: np.ndarray, ba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ba_corr = np.where(ca > 18, ba + (ca * 0.062) - 2.96, ba).astype(np.float32)
    bad_corr = (ba_corr - ca).astype(np.float32)
    return ba_corr, bad_corr




def run_one_repeat(
    df_template: pd.DataFrame,
    dataloader,
    model_paths: list[Path],
    device: torch.device,
) -> pd.DataFrame:
    checkpoint_predictions = []
    for model_path in model_paths:
        preds = run_predictions_single_checkpoint(model_path, dataloader, device)
        checkpoint_predictions.append(preds)


    stacked = np.stack(checkpoint_predictions, axis=0)
    median_predictions = np.median(stacked, axis=0).astype(np.float32)


    df = df_template.copy()
    ca = df["Age"].values.astype(np.float32)
    ba_corr, bad_corr = correct_predictions(ca, median_predictions)


    df["Predicted_Brain_Age"] = ba_corr
    df["Brain_Age_Difference"] = bad_corr


    for idx, preds in enumerate(checkpoint_predictions, start=1):
        df[f"Raw_BA_checkpoint_{idx}"] = preds


    df["Raw_BA_ensemble_median"] = median_predictions
    return df




def build_aggregate_summary(run_csvs: list[Path], out_path: Path) -> None:
    frames = []
    for run_csv in run_csvs:
        df = pd.read_csv(run_csv)
        run_name = run_csv.stem
        df = df[["IXI_ID", "Age", "Filename", "Path", "Predicted_Brain_Age", "Brain_Age_Difference"]].copy()
        df = df.rename(
            columns={
                "Predicted_Brain_Age": f"{run_name}_PBA",
                "Brain_Age_Difference": f"{run_name}_BAD",
            }
        )
        frames.append(df)


    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["IXI_ID", "Age", "Filename", "Path"], how="outer")


    merged.to_csv(out_path, index=False)




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BrainAgeNeXt inference repeatedly on a folder of MRI scans."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing MRI scans.")
    parser.add_argument("--labels-file", type=Path, required=True, help="IXI labels file (tab-separated).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where run CSVs will be saved.")
    parser.add_argument("--checkpoints-dir", type=Path, required=True, help="Folder with BrainAge_1.pth ... BrainAge_5.pth")
    parser.add_argument("--runs", type=int, default=5, help="How many repeated full inference runs to perform.")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"], help="Force device.")
    args = parser.parse_args()


    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if not args.labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {args.labels_file}")
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")


    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device(args.device)


    print(f"Device: {device}")
    print(f"Input dir: {args.input_dir}")
    print(f"Labels file: {args.labels_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Runs: {args.runs}")


    args.output_dir.mkdir(parents=True, exist_ok=True)


    df_template, data_dicts = load_data(args.labels_file, args.input_dir)
    transforms = prepare_transforms()
    dataloader = create_dataloader(data_dicts, transforms)
    model_paths = get_model_paths(args.checkpoints_dir)


    run_csvs = []


    for run_idx in range(1, args.runs + 1):
        print(f"\n=== RUN {run_idx}/{args.runs} ===")
        df_run = run_one_repeat(df_template, dataloader, model_paths, device)
        out_csv = args.output_dir / f"brainagenext_predictions_run_{run_idx:03d}.csv"
        df_run.to_csv(out_csv, index=False)
        run_csvs.append(out_csv)
        print(f"Saved: {out_csv}")


    aggregate_csv = args.output_dir / "brainagenext_predictions_all_runs_wide.csv"
    build_aggregate_summary(run_csvs, aggregate_csv)
    print(f"Saved aggregate summary: {aggregate_csv}")




if __name__ == "__main__":
    main()

