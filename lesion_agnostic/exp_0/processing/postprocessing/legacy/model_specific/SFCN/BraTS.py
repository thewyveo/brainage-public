#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_subject_id_column(df: pd.DataFrame) -> str:
    candidates = [
        "BraTS Subject ID",
        "subject_id",
        "SubjectID",
        "Subject ID",
        "ID",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find a subject ID column. Available columns: {list(df.columns)}")


def build_group_summary(df: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    grouped = df.groupby(feature_col, dropna=False)

    summary = grouped.agg(
        n_subjects=("BraTS Subject ID", "count"),
        mean_age=("Patient's Age", "mean"),
        mean_pred_age=("Predicted_Brain_Age", "mean"),
        mean_bag=("Brain_Age_Difference", "mean"),
        mean_abs_error=("Absolute_Error", "mean"),
        std_bag=("Brain_Age_Difference", "std"),
        std_abs_error=("Absolute_Error", "std"),
    ).reset_index()

    return summary.sort_values("n_subjects", ascending=False)


def build_overall_summary(df: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "Variant": variant_name,
        "n_subjects": len(df),
        "mean_age": df["Patient's Age"].mean(),
        "std_age": df["Patient's Age"].std(),
        "mean_pred_age": df["Predicted_Brain_Age"].mean(),
        "std_pred_age": df["Predicted_Brain_Age"].std(),
        "mean_bag": df["Brain_Age_Difference"].mean(),
        "std_bag": df["Brain_Age_Difference"].std(),
        "mae": df["Absolute_Error"].mean(),
        "rmse": np.sqrt(np.mean((df["Predicted_Brain_Age"] - df["Patient's Age"]) ** 2)),
    }])


def process_variant(
    variant_name: str,
    predictions_path: Path,
    metadata_path: Path,
    output_dir: Path,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n=== Processing {variant_name} ===")
    print(f"Predictions: {predictions_path}")
    print(f"Metadata:    {metadata_path}")

    pred_df = normalize_columns(load_table(predictions_path))
    meta_df = normalize_columns(load_table(metadata_path))

    pred_id_col = find_subject_id_column(pred_df)
    meta_id_col = find_subject_id_column(meta_df)

    pred_df = pred_df.rename(columns={pred_id_col: "BraTS Subject ID"})
    meta_df = meta_df.rename(columns={meta_id_col: "BraTS Subject ID"})

    required_pred_cols = ["Patient's Age", "Path", "Predicted_Brain_Age", "Brain_Age_Difference"]
    missing_pred_cols = [col for col in required_pred_cols if col not in pred_df.columns]
    if missing_pred_cols:
        raise ValueError(
            f"{variant_name}: predictions file is missing required columns: {missing_pred_cols}. "
            f"Available columns: {list(pred_df.columns)}"
        )

    merged_df = pred_df.merge(
        meta_df,
        on="BraTS Subject ID",
        how="left",
        suffixes=("", "_meta")
    )

    merged_df["Predicted_Brain_Age"] = pd.to_numeric(merged_df["Predicted_Brain_Age"], errors="coerce")
    merged_df["Patient's Age"] = pd.to_numeric(merged_df["Patient's Age"], errors="coerce")
    merged_df["Brain_Age_Difference"] = pd.to_numeric(merged_df["Brain_Age_Difference"], errors="coerce")
    merged_df["Absolute_Error"] = np.abs(
        merged_df["Predicted_Brain_Age"] - merged_df["Patient's Age"]
    )
    merged_df["Variant"] = variant_name

    compact_df = merged_df[
        ["Variant", "BraTS Subject ID", "Patient's Age", "Predicted_Brain_Age", "Brain_Age_Difference"]
    ].copy()

    compact_df = compact_df.rename(columns={
        "Patient's Age": "Ground_Truth_Age",
        "Brain_Age_Difference": "BAG",
    })

    compact_out = output_dir / f"predictions_BraTS_{variant_name}.csv"
    compact_df.to_csv(compact_out, index=False)
    print(f"Saved compact file: {compact_out}")

    overall_df = build_overall_summary(merged_df, variant_name)

    subgroup_summaries = []
    available_features = [col for col in feature_columns if col in merged_df.columns]

    for feature in available_features:
        summary = build_group_summary(merged_df, feature)
        summary.insert(0, "Feature_Name", feature)
        summary.insert(0, "Variant", variant_name)
        subgroup_summaries.append(summary)

    if subgroup_summaries:
        subgroup_df = pd.concat(subgroup_summaries, ignore_index=True)
    else:
        subgroup_df = pd.DataFrame()

    return overall_df, subgroup_df


def main():
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent

    metadata_path = root_dir / "data" / "labels" / "BraTS_24.xlsx"
    output_dir = root_dir / "data" / "postprocessed" / "BraTS"
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = {
        "original": root_dir / "data" / "predictions" / "BraTS_24_original_predictions.csv",
        "synthstrip": root_dir / "data" / "predictions" / "BraTS_24_synthstrip_predictions.csv",
    }

    feature_columns = [
        "Site",
        "Magnetic Field Strength",
        "Manufacturer",
        "Sex",
        "Glioma Type",
    ]

    all_overall = []
    all_subgroups = []

    for variant_name, predictions_path in variants.items():
        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing predictions file for {variant_name}: {predictions_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

        overall_df, subgroup_df = process_variant(
            variant_name=variant_name,
            predictions_path=predictions_path,
            metadata_path=metadata_path,
            output_dir=output_dir,
            feature_columns=feature_columns,
        )

        all_overall.append(overall_df)
        if not subgroup_df.empty:
            all_subgroups.append(subgroup_df)

    overall_combined = pd.concat(all_overall, ignore_index=True)
    overall_out = output_dir / "BraTS_preprocessing_variants_overall_summary.csv"
    overall_combined.to_csv(overall_out, index=False)
    print(f"\nSaved overall comparison: {overall_out}")

    if all_subgroups:
        subgroup_combined = pd.concat(all_subgroups, ignore_index=True)
        subgroup_out = output_dir / "BraTS_preprocessing_variants_subgroup_summary.csv"
        subgroup_combined.to_csv(subgroup_out, index=False)
        print(f"Saved subgroup comparison: {subgroup_out}")

    print("Done.")


if __name__ == "__main__":
    main()
