#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
Example usage:

py -3.10 postprocessing/compare_model_sets.py ^
  --set baseline=./exp_0/baseline/postprocessing_tumored ^
  --set lesions=./exp_0/lesions/postprocessing_tumored ^
  --set normalized=./exp_0/normalized/postprocessing_tumored ^
  --output-dir ./postprocessing/comparison_outputs

You can pass as many --set arguments as you want.
Each set directory should contain:
  - combined_predictions_normalized.csv
  - model_summary.csv

py -3.10 postprocessing/overall_global_postprocessing.py `
    --set IXIBRATS=./postprocessing/results/IXI_brats24_only1tumor `
    --set IXIONLY=./postprocessing/results/onlyIXI `
    --output-dir ./postprocessing/results/IXI_vs_IXIBRATS

py -3.10 exp_0\processing\postprocessing\overall\overall_global_postprocessing.py `
    --set IXIBRATSSCALED=exp_0\processing\postprocessing\model_specific\BrainAgeNeXt\insights\scale_effect_full\predictions_with_effective_voxels.csv `
    --set IXIONLY=exp_0\results\onlyIXI\ixi_only_faithfullpreprocess.csv `
    --output-dir exp_0\processing\postprocessing\model_specific\BrainAgeNeXt\insights\scale_effect_full_vs_only
"""


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

SUMMARY_REQUIRED_COLUMNS = [
    "Model",
    "N",
    "MAE",
    "RMSE",
    "BAG_Mean",
    "BAG_STD",
    "Mean_Absolute_BAG",
]

COMBINED_REQUIRED_COLUMNS = [
    "Model",
    "Age",
    "Predicted_Brain_Age",
    "BAG",
]

DEFAULT_SUMMARY_FILENAME = "model_summary.csv"
DEFAULT_COMBINED_FILENAME = "combined_predictions_normalized.csv"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_required_columns(df: pd.DataFrame, required_cols: List[str], df_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{df_name} is missing required columns: {missing}\n"
            f"Found columns: {df.columns.tolist()}"
        )


def sanitize_label(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    while "__" in "".join(out):
        out = list("".join(out).replace("__", "_"))
    return "".join(out).strip("_")


def parse_set_argument(raw: str) -> Tuple[str, Path]:
    """
    Expected format:
      label=/path/to/folder
    """
    if "=" not in raw:
        raise ValueError(
            f"Invalid --set argument: {raw}\n"
            f"Expected format: label=/path/to/folder"
        )

    label, path_str = raw.split("=", 1)
    label = label.strip()
    path_str = path_str.strip()

    if not label:
        raise ValueError(f"Invalid --set argument: {raw}. Empty label.")
    if not path_str:
        raise ValueError(f"Invalid --set argument: {raw}. Empty path.")

    return label, Path(path_str)


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_one_set(
    set_label: str,
    set_dir: Path,
    summary_filename: str,
    combined_filename: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not set_dir.exists():
        raise FileNotFoundError(f"Set directory not found: {set_dir}")

    summary_path = set_dir / summary_filename
    combined_path = set_dir / combined_filename

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file for set '{set_label}': {summary_path}")
    if not combined_path.exists():
        raise FileNotFoundError(f"Missing combined file for set '{set_label}': {combined_path}")

    summary_df = pd.read_csv(summary_path)
    combined_df = pd.read_csv(combined_path)

    summary_df.columns = [str(c).strip() for c in summary_df.columns]
    combined_df.columns = [str(c).strip() for c in combined_df.columns]

    ensure_required_columns(summary_df, SUMMARY_REQUIRED_COLUMNS, f"{set_label} / {summary_filename}")
    ensure_required_columns(combined_df, COMBINED_REQUIRED_COLUMNS, f"{set_label} / {combined_filename}")

    summary_df = summary_df.copy()
    combined_df = combined_df.copy()

    summary_df["Set"] = set_label
    combined_df["Set"] = set_label

    # Numeric coercion for safety
    summary_numeric_cols = [
        "N",
        "MAE",
        "RMSE",
        "BAG_Mean",
        "BAG_STD",
        "Mean_Absolute_BAG",
    ]
    for col in summary_numeric_cols:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

    combined_numeric_cols = ["Age", "Predicted_Brain_Age", "BAG"]
    for col in combined_numeric_cols:
        combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

    summary_df = summary_df.dropna(subset=["Model", "MAE", "RMSE", "BAG_Mean", "Mean_Absolute_BAG"]).reset_index(drop=True)
    combined_df = combined_df.dropna(subset=["Model", "Age", "Predicted_Brain_Age", "BAG"]).reset_index(drop=True)

    return summary_df, combined_df


def load_all_sets(
    set_specs: List[Tuple[str, Path]],
    summary_filename: str,
    combined_filename: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_summary = []
    all_combined = []

    for set_label, set_dir in set_specs:
        summary_df, combined_df = load_one_set(
            set_label=set_label,
            set_dir=set_dir,
            summary_filename=summary_filename,
            combined_filename=combined_filename,
        )
        all_summary.append(summary_df)
        all_combined.append(combined_df)

    if len(all_summary) == 0:
        raise RuntimeError("No valid sets were loaded.")

    summary_all = pd.concat(all_summary, ignore_index=True)
    combined_all = pd.concat(all_combined, ignore_index=True)

    return summary_all, combined_all


# -----------------------------------------------------------------------------
# Derived tables
# -----------------------------------------------------------------------------

def compute_metrics_from_combined(combined_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (set_label, model_name), group in combined_df.groupby(["Set", "Model"], dropna=False):
        age = group["Age"].to_numpy(dtype=np.float32)
        pred = group["Predicted_Brain_Age"].to_numpy(dtype=np.float32)
        bag = group["BAG"].to_numpy(dtype=np.float32)

        abs_err = np.abs(pred - age)
        sq_err = (pred - age) ** 2

        rows.append({
            "Set": set_label,
            "Model": model_name,
            "N_Recomputed": int(len(group)),
            "MAE_Recomputed": float(np.mean(abs_err)),
            "RMSE_Recomputed": float(np.sqrt(np.mean(sq_err))),
            "BAG_Mean_Recomputed": float(np.mean(bag)),
            "BAG_STD_Recomputed": float(np.std(bag)),
            "BAG_Median_Recomputed": float(np.median(bag)),
            "Mean_Absolute_BAG_Recomputed": float(np.mean(np.abs(bag))),
            "Correlation_Age_vs_Pred_Recomputed": safe_corr(age, pred),
            "Correlation_Age_vs_BAG_Recomputed": safe_corr(age, bag),
        })

    return pd.DataFrame(rows).sort_values(["Set", "Model"]).reset_index(drop=True)


def build_metric_pivot(summary_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    pivot = summary_df.pivot(index="Model", columns="Set", values=metric_col)
    return pivot.sort_index().sort_index(axis=1)


def build_set_rankings(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for set_label, group in summary_df.groupby("Set", dropna=False):
        tmp = group.copy()

        tmp["Rank_MAE"] = tmp["MAE"].rank(method="min", ascending=True)
        tmp["Rank_RMSE"] = tmp["RMSE"].rank(method="min", ascending=True)
        tmp["Rank_Mean_Absolute_BAG"] = tmp["Mean_Absolute_BAG"].rank(method="min", ascending=True)
        tmp["Rank_Abs_BAG_Mean"] = tmp["BAG_Mean"].abs().rank(method="min", ascending=True)

        rows.append(tmp[[
            "Set",
            "Model",
            "MAE",
            "RMSE",
            "BAG_Mean",
            "Mean_Absolute_BAG",
            "Rank_MAE",
            "Rank_RMSE",
            "Rank_Mean_Absolute_BAG",
            "Rank_Abs_BAG_Mean",
        ]])

    return pd.concat(rows, ignore_index=True).sort_values(["Set", "Rank_MAE", "Model"]).reset_index(drop=True)


def build_cross_set_model_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each model, compare how it behaves across sets.
    """
    rows = []

    for model_name, group in summary_df.groupby("Model", dropna=False):
        if len(group) == 0:
            continue

        rows.append({
            "Model": model_name,
            "Num_Sets": int(len(group)),
            "Best_Set_by_MAE": str(group.loc[group["MAE"].idxmin(), "Set"]),
            "Best_MAE": float(group["MAE"].min()),
            "Worst_Set_by_MAE": str(group.loc[group["MAE"].idxmax(), "Set"]),
            "Worst_MAE": float(group["MAE"].max()),
            "MAE_Range": float(group["MAE"].max() - group["MAE"].min()),
            "Best_Set_by_Mean_Absolute_BAG": str(group.loc[group["Mean_Absolute_BAG"].idxmin(), "Set"]),
            "Best_Mean_Absolute_BAG": float(group["Mean_Absolute_BAG"].min()),
            "Worst_Set_by_Mean_Absolute_BAG": str(group.loc[group["Mean_Absolute_BAG"].idxmax(), "Set"]),
            "Worst_Mean_Absolute_BAG": float(group["Mean_Absolute_BAG"].max()),
            "Mean_Absolute_BAG_Range": float(group["Mean_Absolute_BAG"].max() - group["Mean_Absolute_BAG"].min()),
        })

    return pd.DataFrame(rows).sort_values("Model").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def save_grouped_metric_barplot(
    summary_df: pd.DataFrame,
    metric_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    pivot = build_metric_pivot(summary_df, metric_col)

    if pivot.empty:
        return

    models = list(pivot.index)
    sets = list(pivot.columns)

    x = np.arange(len(models))
    width = 0.8 / max(len(sets), 1)

    plt.figure(figsize=(max(8, 1.4 * len(models)), 6))

    for i, set_label in enumerate(sets):
        vals = pivot[set_label].to_numpy(dtype=float)
        offsets = x - 0.4 + width / 2 + i * width
        plt.bar(offsets, vals, width=width, label=set_label)

    plt.xticks(x, models, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_grouped_bag_boxplot(combined_df: pd.DataFrame, out_path: Path) -> None:
    order = (
        combined_df[["Set", "Model"]]
        .drop_duplicates()
        .sort_values(["Set", "Model"])
        .values.tolist()
    )

    labels = []
    values = []

    for set_label, model_name in order:
        group = combined_df[(combined_df["Set"] == set_label) & (combined_df["Model"] == model_name)]
        if len(group) == 0:
            continue
        labels.append(f"{set_label}\n{model_name}")
        values.append(group["BAG"].to_numpy())

    if len(values) == 0:
        return

    plt.figure(figsize=(max(10, 0.9 * len(labels)), 6))
    plt.boxplot(values, labels=labels, showfliers=False)
    plt.axhline(0.0, linestyle="--")
    plt.ylabel("BAG")
    plt.title("BAG Distribution by Set and Model")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_grouped_mae_delta_heatmap_like(summary_df: pd.DataFrame, out_path: Path, metric_col: str, title: str) -> None:
    """
    Not a true heatmap dependency-wise, but a simple imshow-based plot with matplotlib only.
    Rows = models, cols = sets.
    """
    pivot = build_metric_pivot(summary_df, metric_col)

    if pivot.empty:
        return

    arr = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(max(6, 1.2 * arr.shape[1]), max(4, 0.6 * arr.shape[0] + 2)))
    im = plt.imshow(arr, aspect="auto")
    plt.colorbar(im, label=metric_col)

    plt.xticks(np.arange(arr.shape[1]), pivot.columns, rotation=30, ha="right")
    plt.yticks(np.arange(arr.shape[0]), pivot.index)
    plt.title(title)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isfinite(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_scatter_by_set(combined_df: pd.DataFrame, out_dir: Path) -> None:
    for set_label, group in combined_df.groupby("Set", dropna=False):
        plt.figure(figsize=(7, 7))

        all_vals = []
        for model_name, sub in group.groupby("Model", dropna=False):
            x = sub["Age"].to_numpy(dtype=float)
            y = sub["Predicted_Brain_Age"].to_numpy(dtype=float)
            all_vals.append(x)
            all_vals.append(y)
            plt.scatter(x, y, alpha=0.5, label=model_name)

        if len(all_vals) > 0:
            all_concat = np.concatenate(all_vals)
            lo = float(np.min(all_concat))
            hi = float(np.max(all_concat))
            plt.plot([lo, hi], [lo, hi], linestyle="--")

        plt.xlabel("Chronological Age")
        plt.ylabel("Predicted Brain Age")
        plt.title(f"{set_label}: Age vs Predicted Brain Age")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{sanitize_label(set_label)}_scatter_age_vs_pred.png", dpi=200)
        plt.close()


def save_overall_scatter(combined_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 8))

    all_vals = []
    for set_label, group in combined_df.groupby("Set", dropna=False):
        x = group["Age"].to_numpy(dtype=float)
        y = group["Predicted_Brain_Age"].to_numpy(dtype=float)
        all_vals.append(x)
        all_vals.append(y)
        plt.scatter(x, y, alpha=0.35, label=set_label)

    if len(all_vals) > 0:
        all_concat = np.concatenate(all_vals)
        lo = float(np.min(all_concat))
        hi = float(np.max(all_concat))
        plt.plot([lo, hi], [lo, hi], linestyle="--")

    plt.xlabel("Chronological Age")
    plt.ylabel("Predicted Brain Age")
    plt.title("All Sets: Age vs Predicted Brain Age")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Args / main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare multiple postprocessing result folders.\n"
            "Each folder should contain model_summary.csv and combined_predictions_normalized.csv."
        )
    )

    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        required=True,
        help=(
            "Set specification in the form label=/path/to/folder . "
            "Can be used multiple times."
        ),
    )

    parser.add_argument(
        "--summary-filename",
        type=str,
        default=DEFAULT_SUMMARY_FILENAME,
        help=f"Summary filename inside each set folder. Default: {DEFAULT_SUMMARY_FILENAME}",
    )

    parser.add_argument(
        "--combined-filename",
        type=str,
        default=DEFAULT_COMBINED_FILENAME,
        help=f"Combined predictions filename inside each set folder. Default: {DEFAULT_COMBINED_FILENAME}",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where all comparison CSVs and plots will be saved.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_specs: List[Tuple[str, Path]] = [parse_set_argument(s) for s in args.sets]

    if args.output_dir is None:
        output_dir = Path.cwd() / "compare_model_sets_outputs"
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading sets:")
    for label, path in set_specs:
        print(f"  {label}: {path}")

    summary_all, combined_all = load_all_sets(
        set_specs=set_specs,
        summary_filename=args.summary_filename,
        combined_filename=args.combined_filename,
    )

    recomputed_metrics_df = compute_metrics_from_combined(combined_all)
    rankings_df = build_set_rankings(summary_all)
    cross_set_model_df = build_cross_set_model_comparison(summary_all)

    merged_check_df = pd.merge(
        summary_all,
        recomputed_metrics_df,
        on=["Set", "Model"],
        how="outer",
    )

    # Save raw merged inputs
    summary_all = summary_all.sort_values(["Set", "Model"]).reset_index(drop=True)
    combined_all = combined_all.sort_values(["Set", "Model"]).reset_index(drop=True)

    summary_all.to_csv(output_dir / "all_sets_model_summary_merged.csv", index=False)
    combined_all.to_csv(output_dir / "all_sets_combined_predictions_merged.csv", index=False)
    recomputed_metrics_df.to_csv(output_dir / "recomputed_metrics_from_combined.csv", index=False)
    rankings_df.to_csv(output_dir / "set_rankings.csv", index=False)
    cross_set_model_df.to_csv(output_dir / "cross_set_model_comparison.csv", index=False)
    merged_check_df.to_csv(output_dir / "summary_vs_recomputed_check.csv", index=False)

    # Save pivots
    for metric in ["MAE", "RMSE", "BAG_Mean", "Mean_Absolute_BAG", "BAG_STD"]:
        pivot_df = build_metric_pivot(summary_all, metric)
        pivot_df.to_csv(output_dir / f"pivot_{metric}.csv")

    # Plots
    save_grouped_metric_barplot(
        summary_df=summary_all,
        metric_col="MAE",
        out_path=output_dir / "grouped_barplot_mae.png",
        title="MAE by Model Across Sets",
        ylabel="MAE",
    )
    save_grouped_metric_barplot(
        summary_df=summary_all,
        metric_col="RMSE",
        out_path=output_dir / "grouped_barplot_rmse.png",
        title="RMSE by Model Across Sets",
        ylabel="RMSE",
    )
    save_grouped_metric_barplot(
        summary_df=summary_all,
        metric_col="Mean_Absolute_BAG",
        out_path=output_dir / "grouped_barplot_mean_absolute_bag.png",
        title="Mean Absolute BAG by Model Across Sets",
        ylabel="Mean Absolute BAG",
    )
    save_grouped_metric_barplot(
        summary_df=summary_all,
        metric_col="BAG_Mean",
        out_path=output_dir / "grouped_barplot_bag_mean.png",
        title="Mean BAG by Model Across Sets",
        ylabel="BAG Mean",
    )

    save_grouped_bag_boxplot(
        combined_df=combined_all,
        out_path=output_dir / "grouped_boxplot_bag_by_set_and_model.png",
    )

    save_grouped_mae_delta_heatmap_like(
        summary_df=summary_all,
        out_path=output_dir / "heatmap_like_mae.png",
        metric_col="MAE",
        title="MAE Across Sets and Models",
    )
    save_grouped_mae_delta_heatmap_like(
        summary_df=summary_all,
        out_path=output_dir / "heatmap_like_mean_absolute_bag.png",
        metric_col="Mean_Absolute_BAG",
        title="Mean Absolute BAG Across Sets and Models",
    )

    save_scatter_by_set(combined_all, output_dir)
    save_overall_scatter(combined_all, output_dir / "overall_scatter_all_sets.png")

    print("\nSaved outputs to:")
    print(f"  {output_dir}")

    print("\nQuick summary:")
    print(
        summary_all[[
            "Set",
            "Model",
            "N",
            "MAE",
            "RMSE",
            "BAG_Mean",
            "BAG_STD",
            "Mean_Absolute_BAG",
        ]].to_string(index=False)
    )

    print("\nBest MAE per set:")
    best_mae = (
        summary_all.sort_values(["Set", "MAE", "Model"])
        .groupby("Set", as_index=False)
        .first()[["Set", "Model", "MAE"]]
    )
    print(best_mae.to_string(index=False))


if __name__ == "__main__":
    main()