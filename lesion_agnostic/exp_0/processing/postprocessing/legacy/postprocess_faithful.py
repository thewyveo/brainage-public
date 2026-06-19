#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = [
        "IXI_ID",
        "Age",
        "Predicted_Brain_Age",
        "Brain_Age_Difference",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{path} missing required column: {col}")

    df = df.copy()
    df["IXI_ID"] = df["IXI_ID"].astype(str).str.strip()
    return df


def paired_shift_plot(df: pd.DataFrame, out_path: Path) -> None:
    d = df.sort_values("PBA_base").reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    x = np.arange(len(d))

    for i, row in d.iterrows():
        plt.plot(
            [i, i],
            [row["PBA_base"], row["PBA_synth"]],
            color="gray",
            alpha=0.25,
            linewidth=1,
        )

    plt.scatter(x, d["PBA_base"], color="royalblue", label="Baseline", s=18)
    plt.scatter(x, d["PBA_synth"], color="crimson", label="Synthetic", s=18)

    plt.xlabel("Subjects")
    plt.ylabel("Predicted Brain Age")
    plt.title("Per-subject prediction shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def direct_comparison_scatter(df: pd.DataFrame, out_path: Path) -> None:
    minv = float(min(df["PBA_base"].min(), df["PBA_synth"].min()))
    maxv = float(max(df["PBA_base"].max(), df["PBA_synth"].max()))

    plt.figure(figsize=(7, 7))
    plt.scatter(df["PBA_base"], df["PBA_synth"], color="darkorchid", alpha=0.75, s=24)
    plt.plot([minv, maxv], [minv, maxv], color="black", linewidth=1.2)
    plt.xlabel("Baseline predicted brain age")
    plt.ylabel("Synthetic predicted brain age")
    plt.title("Direct prediction comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def overlaid_prediction_histogram(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(df["PBA_base"], bins=20, alpha=0.6, label="Baseline", color="royalblue")
    plt.hist(df["PBA_synth"], bins=20, alpha=0.6, label="Synthetic", color="crimson")
    plt.xlabel("Predicted Brain Age")
    plt.ylabel("Count")
    plt.title("Distribution of predicted brain age")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def side_by_side_boxplots(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    bp = plt.boxplot(
        [df["PBA_base"].values, df["PBA_synth"].values],
        labels=["Baseline", "Synthetic"],
        patch_artist=True,
    )

    colors = ["royalblue", "crimson"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    plt.ylabel("Predicted Brain Age")
    plt.title("Predicted brain age: baseline vs synthetic")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def bag_distribution_histogram(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(df["BAD_base"], bins=20, alpha=0.6, label="Baseline BAG", color="seagreen")
    plt.hist(df["BAD_synth"], bins=20, alpha=0.6, label="Synthetic BAG", color="darkorange")
    plt.xlabel("Brain Age Difference")
    plt.ylabel("Count")
    plt.title("Distribution of brain age difference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def age_vs_prediction_overlay(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    plt.scatter(df["Age"], df["PBA_base"], color="royalblue", alpha=0.7, label="Baseline", s=22)
    plt.scatter(df["Age"], df["PBA_synth"], color="crimson", alpha=0.7, label="Synthetic", s=22)
    plt.plot(
        [df["Age"].min(), df["Age"].max()],
        [df["Age"].min(), df["Age"].max()],
        color="black",
        linewidth=1.0,
        linestyle="--",
        label="Ideal: predicted = age",
    )
    plt.xlabel("Chronological Age")
    plt.ylabel("Predicted Brain Age")
    plt.title("Predicted brain age vs chronological age")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def difference_histogram(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(df["PBA_diff"], bins=20, color="slateblue", alpha=0.8)
    plt.axvline(0, color="black", linewidth=1)
    plt.axvline(df["PBA_diff"].mean(), color="red", linewidth=1.2, linestyle="--")
    plt.xlabel("Synthetic - Baseline predicted brain age")
    plt.ylabel("Count")
    plt.title("Distribution of prediction differences")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def combined_distribution_panel(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["PBA_base"], bins=20, alpha=0.6, label="Baseline", color="royalblue")
    axes[0].hist(df["PBA_synth"], bins=20, alpha=0.6, label="Synthetic", color="crimson")
    axes[0].set_xlabel("Predicted Brain Age")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Predicted brain age")
    axes[0].legend()

    axes[1].hist(df["BAD_base"], bins=20, alpha=0.6, label="Baseline BAG", color="seagreen")
    axes[1].hist(df["BAD_synth"], bins=20, alpha=0.6, label="Synthetic BAG", color="darkorange")
    axes[1].set_xlabel("Brain Age Difference")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Brain age difference")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def combined_scatter_panel(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    minv = float(min(df["PBA_base"].min(), df["PBA_synth"].min()))
    maxv = float(max(df["PBA_base"].max(), df["PBA_synth"].max()))

    axes[0].scatter(df["PBA_base"], df["PBA_synth"], color="darkorchid", alpha=0.75, s=24)
    axes[0].plot([minv, maxv], [minv, maxv], color="black", linewidth=1.2)
    axes[0].set_xlabel("Baseline predicted brain age")
    axes[0].set_ylabel("Synthetic predicted brain age")
    axes[0].set_title("Direct comparison")

    axes[1].scatter(df["Age"], df["PBA_base"], color="royalblue", alpha=0.7, label="Baseline", s=22)
    axes[1].scatter(df["Age"], df["PBA_synth"], color="crimson", alpha=0.7, label="Synthetic", s=22)
    axes[1].plot(
        [df["Age"].min(), df["Age"].max()],
        [df["Age"].min(), df["Age"].max()],
        color="black",
        linewidth=1.0,
        linestyle="--",
        label="Ideal",
    )
    axes[1].set_xlabel("Chronological Age")
    axes[1].set_ylabel("Predicted Brain Age")
    axes[1].set_title("Prediction vs age")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def write_summary(df: pd.DataFrame, out_path: Path) -> None:
    diff = df["PBA_diff"].values
    abs_diff = df["PBA_abs_diff"].values
    bad_diff = df["BAD_diff"].values

    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    bias = float(np.mean(diff))
    corr = float(np.corrcoef(df["PBA_base"], df["PBA_synth"])[0, 1])

    baseline_mae_vs_age = float(np.mean(np.abs(df["PBA_base"] - df["Age"])))
    synthetic_mae_vs_age = float(np.mean(np.abs(df["PBA_synth"] - df["Age"])))

    lines = []
    lines.append(f"Subjects compared: {len(df)}")
    lines.append("")
    lines.append("Direct comparison between CSVs")
    lines.append(f"  MAE between predictions        = {mae:.6f}")
    lines.append(f"  RMSE between predictions       = {rmse:.6f}")
    lines.append(f"  Mean signed difference         = {bias:.6f}")
    lines.append(f"  Median signed difference       = {np.median(diff):.6f}")
    lines.append(f"  Std signed difference          = {np.std(diff, ddof=1):.6f}")
    lines.append(f"  Min signed difference          = {np.min(diff):.6f}")
    lines.append(f"  Max signed difference          = {np.max(diff):.6f}")
    lines.append(f"  Correlation                    = {corr:.6f}")
    lines.append("")
    lines.append("Absolute difference")
    lines.append(f"  Mean absolute difference       = {np.mean(abs_diff):.6f}")
    lines.append(f"  Median absolute difference     = {np.median(abs_diff):.6f}")
    lines.append(f"  Max absolute difference        = {np.max(abs_diff):.6f}")
    lines.append("")
    lines.append("Against chronological age")
    lines.append(f"  Baseline MAE vs Age            = {baseline_mae_vs_age:.6f}")
    lines.append(f"  Synthetic MAE vs Age           = {synthetic_mae_vs_age:.6f}")
    lines.append(f"  ΔMAE (Synthetic - Baseline)    = {synthetic_mae_vs_age - baseline_mae_vs_age:.6f}")
    lines.append("")
    lines.append("Brain Age Difference shift")
    lines.append(f"  Mean ΔBAD                      = {np.mean(bad_diff):.6f}")
    lines.append(f"  Median ΔBAD                    = {np.median(bad_diff):.6f}")
    lines.append(f"  Std ΔBAD                       = {np.std(bad_diff, ddof=1):.6f}")
    lines.append("")
    lines.append("Counts")
    lines.append(f"  Synthetic > Baseline           = {(diff > 0).sum()}")
    lines.append(f"  Synthetic < Baseline           = {(diff < 0).sum()}")
    lines.append(f"  Synthetic = Baseline           = {(diff == 0).sum()}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def compare_csvs(csv_baseline: Path, csv_synth: Path, output_dir: Path) -> None:
    df_base = load_csv(csv_baseline)
    df_synth = load_csv(csv_synth)

    df_base = df_base.rename(columns={
        "Age": "Age_base",
        "Predicted_Brain_Age": "PBA_base",
        "Brain_Age_Difference": "BAD_base",
        "Filename": "Filename_base",
        "Path": "Path_base",
    })

    df_synth = df_synth.rename(columns={
        "Age": "Age_synth",
        "Predicted_Brain_Age": "PBA_synth",
        "Brain_Age_Difference": "BAD_synth",
        "Filename": "Filename_synth",
        "Path": "Path_synth",
    })

    merged = pd.merge(
        df_base[["IXI_ID", "Age_base", "PBA_base", "BAD_base", "Filename_base", "Path_base"]],
        df_synth[["IXI_ID", "Age_synth", "PBA_synth", "BAD_synth", "Filename_synth", "Path_synth"]],
        on="IXI_ID",
        how="inner",
    )

    if len(merged) == 0:
        raise RuntimeError("No overlapping IXI_IDs between the two CSVs.")

    merged["Age"] = merged["Age_base"]
    merged["PBA_diff"] = merged["PBA_synth"] - merged["PBA_base"]
    merged["PBA_abs_diff"] = np.abs(merged["PBA_diff"])
    merged["BAD_diff"] = merged["BAD_synth"] - merged["BAD_base"]

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "paired_comparison.csv", index=False)

    paired_shift_plot(merged, output_dir / "per_subject_shift.png")
    direct_comparison_scatter(merged, output_dir / "direct_prediction_comparison.png")
    overlaid_prediction_histogram(merged, output_dir / "prediction_distribution_overlay.png")
    side_by_side_boxplots(merged, output_dir / "prediction_boxplots.png")
    bag_distribution_histogram(merged, output_dir / "bag_distribution_overlay.png")
    age_vs_prediction_overlay(merged, output_dir / "age_vs_prediction_overlay.png")
    difference_histogram(merged, output_dir / "prediction_difference_histogram.png")
    combined_distribution_panel(merged, output_dir / "combined_distribution_panel.png")
    combined_scatter_panel(merged, output_dir / "combined_scatter_panel.png")
    write_summary(merged, output_dir / "summary.txt")

    mae = float(np.mean(merged["PBA_abs_diff"]))
    rmse = float(np.sqrt(np.mean(merged["PBA_diff"] ** 2)))
    baseline_mae_vs_age = float(np.mean(np.abs(merged["PBA_base"] - merged["Age"])))
    synthetic_mae_vs_age = float(np.mean(np.abs(merged["PBA_synth"] - merged["Age"])))

    print(f"Subjects compared: {len(merged)}")
    print(f"MAE between predictions: {mae:.4f}")
    print(f"RMSE between predictions: {rmse:.4f}")
    print(f"Baseline MAE vs age: {baseline_mae_vs_age:.4f}")
    print(f"Synthetic MAE vs age: {synthetic_mae_vs_age:.4f}")
    print(f"Saved outputs to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Directly compare baseline vs synthetic predictions for the same model and subjects."
    )
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--synthetic-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    compare_csvs(args.baseline_csv, args.synthetic_csv, args.output_dir)


if __name__ == "__main__":
    main()
