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
py -3.10 ./postprocessing/overall_postprocessing.py `
  --sfcn SFCN/data/predictions/exp0_gligan_brats24_ixi_t1_only1tumor_sfcn_predictions.csv `
  --brainagenext BrainAgeNeXt/data/predictions/IXI_bratsonly1tumor_predictions.csv `
  --synthba DenseNetSynthBA/data/predictions/synthba_predictions_postprocessed.csv `
  --joos Andras/data/predictions/exp0_gligan_brats24_ixi_t1_only1tumor_joos_predictions.csv

py -3.10 ./postprocessing/overall_postprocessing.py `
  --sfcn SFCN/data/predictions/ixi_t1_only_sfcn_predictions.csv `
  --brainagenext BrainAgeNeXt/data/predictions/ixi_t1_only_brainagenext_predictions.csv `
  --synthba DenseNetSynthBA/postprocessing/synthba_ixionly_predictions_postprocessed.csv `
  --joos Andras/data/predictions/ixi_t1_only_joos_predictions.csv
"""

# -----------------------------------------------------------------------------
# Config / discovery
# -----------------------------------------------------------------------------

DEFAULT_MODEL_PATTERNS: Dict[str, List[str]] = {
    "SFCN": [
        "*sfcn*.csv",
        "*SFCN*.csv",
    ],
    "BrainAgeNeXt": [
        "*brainagenext*.csv",
        "*BrainAgeNeXt*.csv",
    ],
    "SynthBA": [
        "*synthba*.csv",
        "*SynthBA*.csv",
    ],
    "Joos": [
        "*joos*.csv",
        "*Joos*.csv",
        "*two_step*.csv",
        "*andras*.csv",
    ],
}


# -----------------------------------------------------------------------------
# Column handling
# -----------------------------------------------------------------------------

AGE_CANDIDATES = [
    "AGE",
    "Age",
    "Patient's Age",
    "age",
]

PRED_CANDIDATES = [
    "Predicted_Brain_Age",
    "pred",
    "Prediction",
    "pred_age",
    "brain_age",
    "BA",
]

BAG_CANDIDATES = [
    "BAG",
    "Brain_Age_Difference",
    "BAD",
    "bag",
]


def first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_prediction_df(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    age_col = first_existing_column(df, AGE_CANDIDATES)
    pred_col = first_existing_column(df, PRED_CANDIDATES)
    bag_col = first_existing_column(df, BAG_CANDIDATES)

    if age_col is None:
        raise KeyError(
            f"[{model_name}] Could not find age column. "
            f"Expected one of: {AGE_CANDIDATES}. Found: {df.columns.tolist()}"
        )
    if pred_col is None:
        raise KeyError(
            f"[{model_name}] Could not find prediction column. "
            f"Expected one of: {PRED_CANDIDATES}. Found: {df.columns.tolist()}"
        )

    out = pd.DataFrame()
    out["Age"] = pd.to_numeric(df[age_col], errors="coerce")
    out["Predicted_Brain_Age"] = pd.to_numeric(df[pred_col], errors="coerce")

    if bag_col is not None:
        out["BAG"] = pd.to_numeric(df[bag_col], errors="coerce")
    else:
        out["BAG"] = out["Predicted_Brain_Age"] - out["Age"]

    # Try to keep an ID/path for traceability if present
    optional_cols = ["IXI_ID", "BraTS Subject ID", "Path", "path", "Filename", "Case_Folder"]
    for c in optional_cols:
        if c in df.columns:
            out[c] = df[c]

    out["Model"] = model_name
    out = out.dropna(subset=["Age", "Predicted_Brain_Age", "BAG"]).reset_index(drop=True)

    return out


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def compute_model_metrics(df: pd.DataFrame, model_name: str) -> Dict[str, float]:
    age = df["Age"].to_numpy(dtype=np.float32)
    pred = df["Predicted_Brain_Age"].to_numpy(dtype=np.float32)
    bag = df["BAG"].to_numpy(dtype=np.float32)

    abs_err = np.abs(pred - age)
    sq_err = (pred - age) ** 2

    metrics = {
        "Model": model_name,
        "N": int(len(df)),
        "Chronological_Age_Mean": float(np.mean(age)),
        "Chronological_Age_STD": float(np.std(age)),
        "Predicted_Brain_Age_Mean": float(np.mean(pred)),
        "Predicted_Brain_Age_STD": float(np.std(pred)),
        "BAG_Mean": float(np.mean(bag)),
        "BAG_STD": float(np.std(bag)),
        "BAG_Median": float(np.median(bag)),
        "BAG_Min": float(np.min(bag)),
        "BAG_Max": float(np.max(bag)),
        "Mean_Absolute_BAG": float(np.mean(np.abs(bag))),
        "MAE": float(np.mean(abs_err)),
        "RMSE": float(np.sqrt(np.mean(sq_err))),
        "Correlation_Age_vs_Pred": safe_corr(age, pred),
        "Correlation_Age_vs_BAG": safe_corr(age, bag),
    }
    return metrics


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def save_scatter_per_model(df: pd.DataFrame, model_name: str, out_dir: Path) -> None:
    x = df["Age"].to_numpy()
    y = df["Predicted_Brain_Age"].to_numpy()

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.7)
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Chronological Age")
    plt.ylabel("Predicted Brain Age")
    plt.title(f"{model_name}: Age vs Predicted Brain Age")
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_scatter_age_vs_pred.png", dpi=200)
    plt.close()


def save_bag_hist_per_model(df: pd.DataFrame, model_name: str, out_dir: Path) -> None:
    bag = df["BAG"].to_numpy()

    plt.figure(figsize=(7, 5))
    plt.hist(bag, bins=30)
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("Brain Age Gap (BAG)")
    plt.ylabel("Count")
    plt.title(f"{model_name}: BAG Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / f"{model_name}_bag_hist.png", dpi=200)
    plt.close()


def save_combined_scatter(model_dfs: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    plt.figure(figsize=(7, 7))

    all_vals = []
    for model_name, df in model_dfs.items():
        x = df["Age"].to_numpy()
        y = df["Predicted_Brain_Age"].to_numpy()
        all_vals.append(x)
        all_vals.append(y)
        plt.scatter(x, y, alpha=0.5, label=model_name)

    if all_vals:
        all_concat = np.concatenate(all_vals)
        lo = float(np.min(all_concat))
        hi = float(np.max(all_concat))
        plt.plot([lo, hi], [lo, hi], linestyle="--")

    plt.xlabel("Chronological Age")
    plt.ylabel("Predicted Brain Age")
    plt.title("All Models: Age vs Predicted Brain Age")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "combined_scatter_age_vs_pred.png", dpi=200)
    plt.close()


def save_combined_bag_hist(model_dfs: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))

    for model_name, df in model_dfs.items():
        bag = df["BAG"].to_numpy()
        plt.hist(bag, bins=30, alpha=0.45, label=model_name)

    plt.axvline(0.0, linestyle="--")
    plt.xlabel("Brain Age Gap (BAG)")
    plt.ylabel("Count")
    plt.title("All Models: BAG Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "combined_bag_hist.png", dpi=200)
    plt.close()


def save_bag_boxplot(model_dfs: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    labels = []
    values = []

    for model_name, df in model_dfs.items():
        labels.append(model_name)
        values.append(df["BAG"].to_numpy())

    plt.figure(figsize=(8, 5))
    plt.boxplot(values, labels=labels, showfliers=False)
    plt.axhline(0.0, linestyle="--")
    plt.ylabel("Brain Age Gap (BAG)")
    plt.title("BAG by Model")
    plt.tight_layout()
    plt.savefig(out_dir / "bag_boxplot_by_model.png", dpi=200)
    plt.close()


def save_metric_barplot(summary_df: pd.DataFrame, metric_col: str, out_dir: Path, filename: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["Model"], summary_df[metric_col])
    plt.ylabel(metric_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------

def discover_prediction_files(predictions_dir: Path) -> Dict[str, Path]:
    discovered: Dict[str, Path] = {}

    for model_name, patterns in DEFAULT_MODEL_PATTERNS.items():
        matches: List[Path] = []
        for pattern in patterns:
            matches.extend(predictions_dir.glob(pattern))

        matches = sorted(set(matches))
        if len(matches) == 1:
            discovered[model_name] = matches[0]
        elif len(matches) > 1:
            # pick the shortest / most direct match deterministically
            matches = sorted(matches, key=lambda p: (len(str(p.name)), str(p.name)))
            discovered[model_name] = matches[0]

    return discovered


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overall postprocessing for tumored-data brain age predictions.")

    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Directory containing per-model prediction CSVs. If omitted, uses ../data/predictions relative to this script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where summary CSV and plots will be saved.",
    )

    parser.add_argument("--sfcn", type=Path, default=None, help="Optional explicit path to SFCN prediction CSV.")
    parser.add_argument("--brainagenext", type=Path, default=None, help="Optional explicit path to BrainAgeNeXt prediction CSV.")
    parser.add_argument("--synthba", type=Path, default=None, help="Optional explicit path to SynthBA prediction CSV.")
    parser.add_argument("--joos", type=Path, default=None, help="Optional explicit path to Joos prediction CSV.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.predictions_dir is None:
        predictions_dir = script_dir.parent / "data" / "predictions"
    else:
        predictions_dir = args.predictions_dir

    if args.output_dir is None:
        output_dir = predictions_dir / "postprocessing_tumored"
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_dir.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_dir}")

    discovered = discover_prediction_files(predictions_dir)

    explicit_map: Dict[str, Optional[Path]] = {
        "SFCN": args.sfcn,
        "BrainAgeNeXt": args.brainagenext,
        "SynthBA": args.synthba,
        "Joos": args.joos,
    }

    model_files: Dict[str, Path] = {}
    for model_name in ["SFCN", "BrainAgeNeXt", "SynthBA", "Joos"]:
        if explicit_map[model_name] is not None:
            model_files[model_name] = explicit_map[model_name]
        elif model_name in discovered:
            model_files[model_name] = discovered[model_name]

    if len(model_files) == 0:
        raise RuntimeError(
            f"No prediction files found.\n"
            f"Predictions dir: {predictions_dir}\n"
            f"Tried patterns: {DEFAULT_MODEL_PATTERNS}"
        )

    print("Using prediction files:")
    for model_name, path in model_files.items():
        print(f"  {model_name}: {path}")

    model_dfs: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, float]] = []

    for model_name, path in model_files.items():
        if not path.exists():
            print(f"[SKIP] Missing file for {model_name}: {path}")
            continue

        df_raw = pd.read_csv(path)
        df_norm = normalize_prediction_df(df_raw, model_name)

        if len(df_norm) == 0:
            print(f"[SKIP] No valid rows after normalization for {model_name}: {path}")
            continue

        model_dfs[model_name] = df_norm
        summary_rows.append(compute_model_metrics(df_norm, model_name))

        save_scatter_per_model(df_norm, model_name, output_dir)
        save_bag_hist_per_model(df_norm, model_name, output_dir)

    if len(model_dfs) == 0:
        raise RuntimeError("No valid model dataframes were produced.")

    summary_df = pd.DataFrame(summary_rows).sort_values(by="Model").reset_index(drop=True)
    summary_csv = output_dir / "model_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    combined_df = pd.concat(model_dfs.values(), ignore_index=True)
    combined_csv = output_dir / "combined_predictions_normalized.csv"
    combined_df.to_csv(combined_csv, index=False)

    save_combined_scatter(model_dfs, output_dir)
    save_combined_bag_hist(model_dfs, output_dir)
    save_bag_boxplot(model_dfs, output_dir)
    save_metric_barplot(
        summary_df=summary_df,
        metric_col="MAE",
        out_dir=output_dir,
        filename="barplot_mae_by_model.png",
        title="MAE by Model",
    )
    save_metric_barplot(
        summary_df=summary_df,
        metric_col="Mean_Absolute_BAG",
        out_dir=output_dir,
        filename="barplot_mean_absolute_bag_by_model.png",
        title="Mean Absolute BAG by Model",
    )
    save_metric_barplot(
        summary_df=summary_df,
        metric_col="BAG_Mean",
        out_dir=output_dir,
        filename="barplot_bag_mean_by_model.png",
        title="Mean BAG by Model",
    )

    print(f"\nSaved summary CSV to: {summary_csv}")
    print(f"Saved normalized combined CSV to: {combined_csv}")
    print(f"Saved plots to: {output_dir}")

    print("\nQuick summary:")
    print(summary_df[[
        "Model",
        "N",
        "MAE",
        "RMSE",
        "BAG_Mean",
        "BAG_STD",
        "Mean_Absolute_BAG",
        "Correlation_Age_vs_Pred",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()