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
    raise ValueError(
        f"Could not find a subject ID column. Available columns: {list(df.columns)}"
    )




def build_group_summary(df: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    grouped = df.groupby(feature_col, dropna=False)


    summary = grouped.agg(
        n_subjects=("BraTS Subject ID", "count"),
        mean_age=("Patient's Age", "mean"),
        std_age=("Patient's Age", "std"),
        mean_pred_age=("Predicted_Brain_Age", "mean"),
        std_pred_age=("Predicted_Brain_Age", "std"),
        mean_bag=("Brain_Age_Difference", "mean"),
        std_bag=("Brain_Age_Difference", "std"),
        mean_abs_error=("Absolute_Error", "mean"),
        std_abs_error=("Absolute_Error", "std"),
    ).reset_index()


    summary = summary.sort_values("n_subjects", ascending=False)
    return summary




def build_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame([{
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
    return summary




def main():
    script_dir = Path(__file__).resolve().parent


    predictions_path = script_dir / "data" / "predictions" / "BraTS_24_predictions.csv"
    metadata_path = script_dir / "data" / "labels" / "BraTS_24.xlsx"
    output_dir = script_dir / "data" / "postprocessed" / "BraTS"
    output_dir.mkdir(parents=True, exist_ok=True)


    print(f"Loading predictions from: {predictions_path}")
    print(f"Loading metadata from:    {metadata_path}")


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
            f"Predictions file is missing required columns: {missing_pred_cols}. "
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


    predictions_brats_df = merged_df[
        ["BraTS Subject ID", "Patient's Age", "Predicted_Brain_Age", "Brain_Age_Difference"]
    ].copy()


    predictions_brats_df = predictions_brats_df.rename(columns={
        "Patient's Age": "Ground_Truth_Age",
        "Brain_Age_Difference": "BAG",
    })


    predictions_brats_out = output_dir / "predictions_BraTS.csv"
    predictions_brats_df.to_csv(predictions_brats_out, index=False)
    print(f"Saved compact predictions file to: {predictions_brats_out}")


    merged_out = output_dir / "BraTS_predictions_with_metadata.csv"
    merged_df.to_csv(merged_out, index=False)
    print(f"Saved merged file to: {merged_out}")


    overall_summary = build_overall_summary(merged_df)
    overall_out = output_dir / "overall_summary.csv"
    overall_summary.to_csv(overall_out, index=False)
    print(f"Saved overall summary to: {overall_out}")


    feature_columns = [
        "Site",
        "Magnetic Field Strength",
        "Manufacturer",
        "Sex",
        "Glioma Type",
    ]


    available_features = [col for col in feature_columns if col in merged_df.columns]
    missing_features = [col for col in feature_columns if col not in merged_df.columns]


    print(f"Available feature columns: {available_features}")
    if missing_features:
        print(f"Missing feature columns:   {missing_features}")


    all_summaries = []


    for feature in available_features:
        summary = build_group_summary(merged_df, feature)
        summary["Feature_Name"] = feature


        out_file = output_dir / f"{feature.replace(' ', '_').replace('/', '_')}_summary.csv"
        summary.to_csv(out_file, index=False)
        print(f"Saved summary for {feature} to: {out_file}")


        all_summaries.append(summary)


    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined_out = output_dir / "all_feature_summaries.csv"
        combined.to_csv(combined_out, index=False)
        print(f"Saved combined summary to: {combined_out}")


    print("Done.")




if __name__ == "__main__":
    main()

