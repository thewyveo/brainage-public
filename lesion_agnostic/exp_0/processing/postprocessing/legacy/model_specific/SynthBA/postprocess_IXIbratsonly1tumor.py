#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd


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


def extract_ixi_id_from_path(path_str):
    path_str = str(path_str)
    filename = Path(path_str).name

    # expected examples:
    # IXI002-Guys-0828-T1_synthetic_t1.nii.gz
    # IXI072-HH-2324-T1_preprocessed_synthetic_t1.nii.gz
    # or longer paths ending in such names
    raw_id = filename.split("-")[0].strip()
    return normalize_ixi_id(raw_id)


def main():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # ---- adjust these if needed ----
    synthba_predictions_path = project_root / "data" / "predictions" / "synthba_predictions_ixionly.csv"
    labels_path = project_root / "data" / "labels" / "IXI_clean.xls"
    output_path = project_root / "data" / "predictions" / "synthba_ixionly_predictions_postprocessed.csv"

    print(f"SynthBA predictions file: {synthba_predictions_path}")
    print(f"Labels file: {labels_path}")

    if not synthba_predictions_path.exists():
        raise FileNotFoundError(f"SynthBA predictions file not found: {synthba_predictions_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    # ---- load synthba predictions ----
    if synthba_predictions_path.suffix.lower() == ".xlsx":
        df_pred = pd.read_excel(synthba_predictions_path)
    elif synthba_predictions_path.suffix.lower() == ".csv":
        df_pred = pd.read_csv(synthba_predictions_path)
    else:
        raise ValueError("SynthBA predictions file must be .xlsx or .csv")

    df_pred.columns = [str(c).strip() for c in df_pred.columns]

    if "path" not in df_pred.columns:
        raise KeyError(f"Missing column 'path' in SynthBA predictions. Found: {df_pred.columns.tolist()}")
    if "pred" not in df_pred.columns:
        raise KeyError(f"Missing column 'pred' in SynthBA predictions. Found: {df_pred.columns.tolist()}")

    # ---- load IXI labels (tab-separated fake .xls) ----
    df_labels = pd.read_csv(labels_path, sep="\t")
    df_labels.columns = [str(c).strip() for c in df_labels.columns]

    if "IXI_ID" not in df_labels.columns:
        raise KeyError(f"Missing column 'IXI_ID' in labels file. Found: {df_labels.columns.tolist()}")
    if "AGE" not in df_labels.columns:
        raise KeyError(f"Missing column 'AGE' in labels file. Found: {df_labels.columns.tolist()}")

    df_labels["IXI_ID"] = df_labels["IXI_ID"].apply(normalize_ixi_id)

    # ---- extract IDs from synthba paths ----
    df_pred["IXI_ID"] = df_pred["path"].apply(extract_ixi_id_from_path)

    # ---- merge age ----
    df_merged = df_pred.merge(
        df_labels[["IXI_ID", "AGE"]],
        on="IXI_ID",
        how="left"
    )

    # ---- check for unmatched cases ----
    unmatched = df_merged["AGE"].isna().sum()
    if unmatched > 0:
        print(f"Warning: {unmatched} predictions could not be matched to an AGE label.")

    # ---- numeric conversion ----
    df_merged["pred"] = pd.to_numeric(df_merged["pred"], errors="coerce")
    df_merged["AGE"] = pd.to_numeric(df_merged["AGE"], errors="coerce")

    # ---- BAG ----
    df_merged["BAG"] = df_merged["pred"] - df_merged["AGE"]

    # ---- reorder columns nicely ----
    preferred_order = ["IXI_ID", "AGE", "pred", "BAG", "path"]
    remaining_cols = [c for c in df_merged.columns if c not in preferred_order]
    df_merged = df_merged[preferred_order + remaining_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(output_path, index=False)

    print(f"Saved postprocessed file to: {output_path}")
    print(f"Total rows: {len(df_merged)}")
    print(f"Matched rows: {df_merged['AGE'].notna().sum()}")
    print(f"Unmatched rows: {unmatched}")


if __name__ == "__main__":
    main()

