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
from nnunet_mednext import create_mednext_v1, create_mednext_encoder_v1

from pathlib import Path

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


def load_data(labels_file):

    # load label table
    if str(labels_file).endswith(".xlsx"):
        df_labels = pd.read_excel(labels_file)
    else:
        df_labels = pd.read_csv(labels_file)

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data" / "preprocessed" / "BraTS"

    # build lookup dictionary once (fast)
    age_lookup = dict(zip(df_labels["BraTS Subject ID"], df_labels["Patient's Age"]))

    nii_files = list(data_dir.glob("*.nii.gz"))

    rows = []
    data_dicts = []

    for nii in nii_files:

        # extract subject id from filename
        subject_id = nii.name.split("-t1")[0]

        if subject_id not in age_lookup:
            print(f"Skipping {subject_id} (no age found)")
            continue

        age = age_lookup[subject_id]

        rows.append({
            "BraTS Subject ID": subject_id,
            "Patient's Age": age,
            "Path": str(nii)
        })

        data_dicts.append({
            "image": str(nii),
            "label": age
        })

    df = pd.DataFrame(rows)

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
    labels_path = script_dir.parent / "data" / "labels" / "BraTS_24.xlsx"

    print(labels_path)

    df, data_dicts = load_data(labels_path)
    transforms = prepare_transforms()
    dataloader = create_dataloader(data_dicts, transforms)

    model_paths = [
        os.path.join(os.path.dirname(__file__), f'BrainAge_{i}.pth') for i in range(1, 6)
    ]

    predictions_list = [run_predictions(model_path, dataloader) for model_path in model_paths]
    average_predictions = np.median(np.stack(predictions_list), axis=0)

    CA = df["Patient's Age"].values
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
