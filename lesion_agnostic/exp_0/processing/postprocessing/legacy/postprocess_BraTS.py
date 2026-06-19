#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_existing_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


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


def standardize_prediction_df(
    pred_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    pred_df = normalize_columns(pred_df)
    meta_df = normalize_columns(meta_df)

    meta_id_col = find_subject_id_column(meta_df)
    meta_df = meta_df.rename(columns={meta_id_col: "BraTS Subject ID"})

    # Case 1: already in full postprocessed format
    expected_full_cols = {
        "BraTS Subject ID",
        "Patient's Age",
        "Predicted_Brain_Age",
        "Brain_Age_Difference",
    }
    if expected_full_cols.issubset(set(pred_df.columns)):
        pred_id_col = find_subject_id_column(pred_df)
        pred_df = pred_df.rename(columns={pred_id_col: "BraTS Subject ID"})

        pred_df["Predicted_Brain_Age"] = pd.to_numeric(
            pred_df["Predicted_Brain_Age"], errors="coerce"
        )
        pred_df["Patient's Age"] = pd.to_numeric(
            pred_df["Patient's Age"], errors="coerce"
        )
        pred_df["Brain_Age_Difference"] = pd.to_numeric(
            pred_df["Brain_Age_Difference"], errors="coerce"
        )

        merged_df = pred_df.merge(
            meta_df,
            on="BraTS Subject ID",
            how="left",
            suffixes=("", "_meta")
        )

    # Case 2: compact postprocessed format
    elif {"BraTS Subject ID", "Ground_Truth_Age", "Predicted_Brain_Age", "BAG"}.issubset(set(pred_df.columns)):
        pred_id_col = find_subject_id_column(pred_df)
        pred_df = pred_df.rename(columns={
            pred_id_col: "BraTS Subject ID",
            "Ground_Truth_Age": "Patient's Age",
            "BAG": "Brain_Age_Difference",
        })

        pred_df["Predicted_Brain_Age"] = pd.to_numeric(
            pred_df["Predicted_Brain_Age"], errors="coerce"
        )
        pred_df["Patient's Age"] = pd.to_numeric(
            pred_df["Patient's Age"], errors="coerce"
        )
        pred_df["Brain_Age_Difference"] = pd.to_numeric(
            pred_df["Brain_Age_Difference"], errors="coerce"
        )

        merged_df = pred_df.merge(
            meta_df,
            on="BraTS Subject ID",
            how="left",
            suffixes=("", "_meta")
        )

    # Case 3: raw SynthBA style path,pred
    elif {"path", "pred"}.issubset(set(pred_df.columns)):
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
                f"{model_name}: Column \"Patient's Age\" not found after merge."
            )

        merged_df["Predicted_Brain_Age"] = pd.to_numeric(
            merged_df["Predicted_Brain_Age"], errors="coerce"
        )
        merged_df["Patient's Age"] = pd.to_numeric(
            merged_df["Patient's Age"], errors="coerce"
        )
        merged_df["Brain_Age_Difference"] = (
            merged_df["Predicted_Brain_Age"] - merged_df["Patient's Age"]
        )

    else:
        raise ValueError(
            f"{model_name}: unsupported prediction format. Columns found: {list(pred_df.columns)}"
        )

    merged_df["Absolute_Error"] = np.abs(
        merged_df["Predicted_Brain_Age"] - merged_df["Patient's Age"]
    )
    merged_df["Squared_Error"] = (
        merged_df["Predicted_Brain_Age"] - merged_df["Patient's Age"]
    ) ** 2
    merged_df["Model"] = model_name

    keep_cols = ["Model", "BraTS Subject ID", "Patient's Age", "Predicted_Brain_Age",
                 "Brain_Age_Difference", "Absolute_Error", "Path"]

    feature_cols = [
        "Site",
        "Magnetic Field Strength",
        "Manufacturer",
        "Sex",
        "Glioma Type",
    ]
    for col in feature_cols:
        if col in merged_df.columns:
            keep_cols.append(col)

    keep_cols = [c for c in keep_cols if c in merged_df.columns]
    merged_df = merged_df[keep_cols].copy()

    return merged_df


def build_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("Model", dropna=False)

    summary = grouped.agg(
        n_subjects=("BraTS Subject ID", "count"),
        mean_age=("Patient's Age", "mean"),
        mean_pred_age=("Predicted_Brain_Age", "mean"),
        mean_bag=("Brain_Age_Difference", "mean"),
        std_bag=("Brain_Age_Difference", "std"),
        mae=("Absolute_Error", "mean"),
        rmse=("Absolute_Error", lambda x: np.sqrt(np.mean(np.square(x)))),
    ).reset_index()

    return summary.sort_values("mae", ascending=True)


def build_feature_summary(df: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    grouped = df.groupby(["Model", feature_col], dropna=False)

    summary = grouped.agg(
        n_subjects=("BraTS Subject ID", "count"),
        mean_age=("Patient's Age", "mean"),
        mean_pred_age=("Predicted_Brain_Age", "mean"),
        mean_bag=("Brain_Age_Difference", "mean"),
        std_bag=("Brain_Age_Difference", "std"),
        mae=("Absolute_Error", "mean"),
        rmse=("Absolute_Error", lambda x: np.sqrt(np.mean(np.square(x)))),
    ).reset_index()

    summary.insert(1, "Feature_Name", feature_col)
    return summary.sort_values(["Model", "n_subjects"], ascending=[True, False])


def make_bar_plot(summary_df: pd.DataFrame, value_col: str, out_path: Path, title: str, ylabel: str):
    plt.figure(figsize=(10, 6))
    plt.bar(summary_df["Model"], summary_df[value_col])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_bag_boxplot(all_predictions_df: pd.DataFrame, out_path: Path):
    models = list(all_predictions_df["Model"].dropna().unique())
    data = [
        all_predictions_df.loc[all_predictions_df["Model"] == model, "Brain_Age_Difference"].dropna().values
        for model in models
    ]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=models)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("BAG")
    plt.title("BAG distribution by model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def make_scatter_plot(all_predictions_df: pd.DataFrame, out_path: Path):
    models = list(all_predictions_df["Model"].dropna().unique())
    n_models = len(models)

    fig, axes = plt.subplots(n_models, 1, figsize=(7, 4 * n_models), squeeze=False)

    for i, model in enumerate(models):
        ax = axes[i, 0]
        sub = all_predictions_df[all_predictions_df["Model"] == model].copy()

        x = sub["Patient's Age"].to_numpy(dtype=float)
        y = sub["Predicted_Brain_Age"].to_numpy(dtype=float)

        ax.scatter(x, y)
        if len(x) > 0:
            mn = min(np.nanmin(x), np.nanmin(y))
            mx = max(np.nanmax(x), np.nanmax(y))
            ax.plot([mn, mx], [mn, mx], linestyle="--")
        ax.set_title(model)
        ax.set_xlabel("Chronological Age")
        ax.set_ylabel("Predicted Brain Age")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    script_dir = Path(__file__).resolve().parent
    exp0_dir = script_dir.parent
    output_dir = script_dir / "results" / "BraTS"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_candidates = [
        exp0_dir / "SFCN" / "data" / "labels" / "BraTS_24.xlsx",
        exp0_dir / "Andras" / "data" / "labels" / "BraTS_24.xlsx",
        exp0_dir / "BrainAgeNeXt" / "data" / "labels" / "BraTS_24.xlsx",
        exp0_dir / "DenseNetSynthBA" / "data" / "labels" / "BraTS_24.xlsx",
    ]
    metadata_path = find_existing_path(metadata_candidates)
    if metadata_path is None:
        raise FileNotFoundError("Could not find BraTS_24.xlsx in any expected model directory.")

    model_prediction_paths = {
        "Andras_TwoStep": [
            exp0_dir / "Andras" / "data" / "predictions" / "BraTS_24_predictions.csv",
            exp0_dir / "Andras" / "data" / "postprocessed" / "BraTS" / "predictions_BraTS.csv",
        ],
        "BrainAgeNeXt": [
            exp0_dir / "BrainAgeNeXt" / "data" / "predictions" / "BraTS_24_predictions.csv",
            exp0_dir / "BrainAgeNeXt" / "data" / "postprocessed" / "BraTS" / "predictions_BraTS.csv",
        ],
        "DenseNetSynthBA": [
            exp0_dir / "DenseNetSynthBA" / "data" / "predictions" / "synthba_predictions.csv",
            exp0_dir / "DenseNetSynthBA" / "data" / "postprocessed" / "BraTS" / "predictions_BraTS.csv",
        ],
        "SFCN_Original": [
            exp0_dir / "SFCN" / "data" / "predictions" / "BraTS_24_original_predictions.csv",
            exp0_dir / "SFCN" / "data" / "postprocessed" / "BraTS" / "predictions_BraTS_original.csv",
        ],
        "SFCN_SynthStrip": [
            exp0_dir / "SFCN" / "data" / "predictions" / "BraTS_24_synthstrip_predictions.csv",
            exp0_dir / "SFCN" / "data" / "postprocessed" / "BraTS" / "predictions_BraTS_synthstrip.csv",
        ],
    }

    meta_df = normalize_columns(load_table(metadata_path))

    all_predictions = []
    found_models = []

    print(f"Using metadata file: {metadata_path}")

    for model_name, candidates in model_prediction_paths.items():
        pred_path = find_existing_path(candidates)
        if pred_path is None:
            print(f"Skipping {model_name}: no prediction file found.")
            continue

        print(f"Loading {model_name} predictions from: {pred_path}")
        pred_df = load_table(pred_path)
        standardized = standardize_prediction_df(pred_df, meta_df, model_name)
        all_predictions.append(standardized)
        found_models.append(model_name)

    if not all_predictions:
        raise RuntimeError("No prediction files found for any model.")

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)

    compact_df = all_predictions_df[
        ["Model", "BraTS Subject ID", "Patient's Age", "Predicted_Brain_Age", "Brain_Age_Difference"]
    ].copy()
    compact_df = compact_df.rename(columns={
        "Patient's Age": "Ground_Truth_Age",
        "Brain_Age_Difference": "BAG",
    })
    compact_out = output_dir / "BraTS_all_models_predictions_compact.csv"
    compact_df.to_csv(compact_out, index=False)
    print(f"Saved compact predictions: {compact_out}")

    summary_df = build_model_summary(all_predictions_df)
    summary_out = output_dir / "BraTS_model_summary.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"Saved model summary: {summary_out}")

    feature_columns = [
        "Site",
        "Magnetic Field Strength",
        "Manufacturer",
        "Sex",
        "Glioma Type",
    ]
    available_features = [col for col in feature_columns if col in all_predictions_df.columns]

    if available_features:
        feature_summaries = []
        for feature in available_features:
            feature_summary = build_feature_summary(all_predictions_df, feature)
            feature_summaries.append(feature_summary)

        feature_df = pd.concat(feature_summaries, ignore_index=True)
        feature_out = output_dir / "BraTS_model_subgroup_summary.csv"
        feature_df.to_csv(feature_out, index=False)
        print(f"Saved subgroup summary: {feature_out}")

    mae_plot = output_dir / "BraTS_model_mae_bar.png"
    rmse_plot = output_dir / "BraTS_model_rmse_bar.png"
    bag_plot = output_dir / "BraTS_model_bag_boxplot.png"
    scatter_plot = output_dir / "BraTS_model_age_scatter.png"

    make_bar_plot(summary_df, "mae", mae_plot, "BraTS MAE by model", "MAE")
    make_bar_plot(summary_df, "rmse", rmse_plot, "BraTS RMSE by model", "RMSE")
    make_bag_boxplot(all_predictions_df, bag_plot)
    make_scatter_plot(all_predictions_df, scatter_plot)

    print(f"Saved plot: {mae_plot}")
    print(f"Saved plot: {rmse_plot}")
    print(f"Saved plot: {bag_plot}")
    print(f"Saved plot: {scatter_plot}")

    print("Done.")


if __name__ == "__main__":
    main()
