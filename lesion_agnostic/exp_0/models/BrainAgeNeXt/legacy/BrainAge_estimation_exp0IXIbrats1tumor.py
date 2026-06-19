#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Francesco La Rosa
"""
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from monai.transforms import Compose, LoadImaged, ScaleIntensityd, Spacingd, CropForegroundd, SpatialPadd, CenterSpatialCropd
from monai.data import CacheDataset
import numpy as np
import os
import torchio
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure vendored MedNeXt package is importable when running this script directly.
_MEDNEXT_ROOT = Path(__file__).resolve().parents[1] / "MedNeXt"
if _MEDNEXT_ROOT.exists():
    sys.path.insert(0, str(_MEDNEXT_ROOT))

from nnunet_mednext import create_mednext_v1, create_mednext_encoder_v1  # pyright: ignore[reportMissingImports]


class MedNeXtEncReg(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MedNeXtEncReg, self).__init__()
        self.mednextv1 = create_mednext_encoder_v1(num_input_channels=1, num_classes=1, model_id='B', kernel_size=3, deep_supervision=True)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.regression_fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        mednext_out = self.mednextv1(x)
        x = mednext_out
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
        CenterSpatialCropd(keys=["image"], roi_size=(x, y, z))
    ]
    val_torchio_transforms = torchio.transforms.Compose(
        [torchio.transforms.ZNormalization(masking_method=lambda x: x > 0, keys=["image"], include=['image'])]
    )
    return Compose(monai_transforms + [val_torchio_transforms])


def normalize_ixi_id(x):
    s = str(x).strip()

    if s.upper().startswith("IXI"):
        num = s[3:]
    else:
        num = s

    try:
        num = int(float(num))
    except:
        return None

    return f"IXI{num:03d}"


def load_data(labels_file):

    df_labels = pd.read_csv(labels_file, sep="\t")
    df_labels["IXI_ID"] = df_labels["IXI_ID"].apply(normalize_ixi_id)
    valid_ids = set(df_labels["IXI_ID"])

    print(f"Valid IXI IDs: {len(valid_ids)}")

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data" / "exp0_gligan_brats24_ixi_t1_only1tumor"

    nii_files = list(data_dir.rglob("*_synthetic_t1.nii.gz"))

    print(f"Found total synthetic scans: {len(nii_files)}")

    rows = []
    data_dicts = []

    missing_files = 0

    for nii in nii_files:

        # 🔥 CRITICAL FIX
        if not nii.exists():
            print(f"[MISSING FILE] {nii}")
            missing_files += 1
            continue

        subject_id = nii.name.split("-")[0].strip()
        subject_id = normalize_ixi_id(subject_id)

        if subject_id is None:
            continue

        if subject_id not in valid_ids:
            continue

        age_row = df_labels.loc[df_labels["IXI_ID"] == subject_id, "AGE"]

        if len(age_row) == 0:
            continue

        age = float(age_row.values[0])

        rows.append({
            "IXI_ID": subject_id,
            "Age": age,
            "Path": str(nii)
        })

        data_dicts.append({
            "image": str(nii),
            "label": age
        })

    df = pd.DataFrame(rows)

    print(f"\nFinal dataset size: {len(df)}")
    print(f"Missing files skipped: {missing_files}")

    return df, data_dicts






#def create_dataloader(data_dicts, transforms):
#    dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=0.2, num_workers=4)
#    dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=torch.cuda.is_available())
#    return dataloader

def create_dataloader(data_dicts, transforms):
    dataset = CacheDataset(data=data_dicts, transform=transforms, cache_rate=0.2, num_workers=0)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        pin_memory=False
    )
    return dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_model():
    torch.cuda.empty_cache()
    return MedNeXtEncReg().to(device)


def run_predictions(model_path, dataloader):
    model = initialize_model()
    #model.load_state_dict(torch.load(model_path))
    
    # cpu
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_data in dataloader:
            images = batch_data['image'].to(device)
            pred = model(images)
            predictions.append(pred.cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.array(predictions)


def main():
    script_dir = Path(__file__).resolve().parent
    labels_path = script_dir.parent / "data" / "labels" / "IXI_clean.xls"
    models_folder = script_dir.parent / "checkpoints"

    print(labels_path)

    df, data_dicts = load_data(labels_path)
    transforms = prepare_transforms()
    dataloader = create_dataloader(data_dicts, transforms)

    all_model_paths = [
        os.path.join(os.path.dirname(__file__), f'BrainAge_{i}.pth') for i in range(1, 6)
    ]
    model_paths = [models_folder / model_path for model_path in all_model_paths]

    predictions_list = [run_predictions(model_path, dataloader) for model_path in model_paths]
    average_predictions = np.median(np.stack(predictions_list), axis=0)

    CA = df["Age"].values
    BA = average_predictions.flatten()
    BA_corr = np.where(CA > 18, BA + (CA * 0.062) - 2.96, BA)
    BAD_corr = BA_corr - CA
    
    df['Predicted_Brain_Age'] = BA_corr
    df['Brain_Age_Difference'] = BAD_corr

    out_file = labels_path.with_name(labels_path.stem + "_predictions.csv")
    df.to_csv(out_file, index=False)
    print('Updated CSV file saved.')


if __name__ == '__main__':
    main()
