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




def extract_subject_id_from_prediction_path(path_str: str) -> str:
    name = Path(str(path_str).replace("\\", "/")).name


    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]


    if name.endswith("_preprocessed"):
        name = name[:-13]


    if "-t1" in name:
        return name.split("-t1")[0]


    return name




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


    predictions_path = script_dir.parent / "data" / "predictions" / "synthba_predictions.csv"
    metadata_path = script_dir.parent / "data" / "labels" / "BraTS_24.xlsx"
    output_dir = script_dir.parent / "data" / "predictions" / "subgroup_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_df = normalize_columns(load_table(predictions_path))
    meta_df = normalize_columns(load_table(metadata_path))


    if "path" not in pred_df.columns:
        raise ValueError(f"Predictions CSV must contain a 'path' column. Found: {list(pred_df.columns)}")
    if "pred" not in pred_df.columns:
        raise ValueError(f"Predictions CSV must contain a 'pred' column. Found: {list(pred_df.columns)}")


    meta_id_col = find_subject_id_column(meta_df)
    meta_df = meta_df.rename(columns={meta_id_col: "BraTS Subject ID"})


    pred_df["BraTS Subject ID"] = pred_df["path"].apply(extract_subject_id_from_prediction_path)
    pred_df = pred_df.rename(columns={
        "path": "Path",
        "pred": "Predicted_Brain_Age",
    })


    merged_df = pred_df.merge(
        meta_df,
        on="BraTS Subject ID",
        how="left",
        suffixes=("", "_meta")
    )


    if "Patient's Age" not in merged_df.columns:
        raise ValueError(
            "Column \"Patient's Age\" not found after merge. "
            f"Available columns: {list(merged_df.columns)}"
        )


    merged_df["Predicted_Brain_Age"] = pd.to_numeric(merged_df["Predicted_Brain_Age"], errors="coerce")
    merged_df["Patient's Age"] = pd.to_numeric(merged_df["Patient's Age"], errors="coerce")


    merged_df["Brain_Age_Difference"] = (
        merged_df["Predicted_Brain_Age"] - merged_df["Patient's Age"]
    )
    merged_df["Absolute_Error"] = np.abs(
        merged_df["Predicted_Brain_Age"] - merged_df["Patient's Age"]
    )


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

