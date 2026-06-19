#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


r"""
py -3.10 exp_0\processing\postprocessing\overall\global_scaled.py `
    --scaled-csv exp_0\processing\postprocessing\model_specific\BrainAgeNeXt\insights\scale_effect_full\predictions_with_effective_voxels.csv `
    --ixi-only-csv exp_0\results\onlyIXI\combined_predictions_normalized.csv `
    --output-dir exp_0\processing\postprocessing\model_specific\BrainAgeNeXt\insights\scale_effect_GLOBAL
"""


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


def load_scaled(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = [
        "IXI_ID",
        "Age",
        "Filename",
        "Predicted_Brain_Age",
        "BAG",
        "effective_voxels",
        "mask_code",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in scaled CSV")

    df = df.copy()
    df["IXI_ID"] = df["IXI_ID"].astype(str).str.strip()
    df["Age"] = pd.to_numeric(df["Age"], errors="raise")
    df["Predicted_Brain_Age"] = pd.to_numeric(df["Predicted_Brain_Age"], errors="raise")
    df["BAG"] = pd.to_numeric(df["BAG"], errors="raise")
    df["effective_voxels"] = pd.to_numeric(df["effective_voxels"], errors="raise")

    if "abs_BAG" not in df.columns:
        df["abs_BAG"] = df["BAG"].abs()
    else:
        df["abs_BAG"] = pd.to_numeric(df["abs_BAG"], errors="raise")

    return df


def load_ixi_only(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = [
        "IXI_ID",
        "Age",
        "Predicted_Brain_Age",
        "BAG",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in IXI-only CSV")

    df = df.copy()
    df["IXI_ID"] = df["IXI_ID"].astype(str).str.strip()
    df["Age"] = pd.to_numeric(df["Age"], errors="raise")
    df["Predicted_Brain_Age"] = pd.to_numeric(df["Predicted_Brain_Age"], errors="raise")
    df["BAG"] = pd.to_numeric(df["BAG"], errors="raise")
    df["abs_BAG"] = df["BAG"].abs()

    return df.rename(
        columns={
            "Predicted_Brain_Age": "healthy_Predicted_Brain_Age",
            "BAG": "healthy_BAG",
            "abs_BAG": "healthy_abs_BAG",
            "Age": "healthy_Age",
            "Filename": "healthy_Filename" if "Filename" in df.columns else "Filename",
            "Path": "healthy_Path" if "Path" in df.columns else "Path",
        }
    )


# -----------------------------------------------------------------------------
# Merge / derive
# -----------------------------------------------------------------------------


def merge_scaled_with_healthy(scaled_df: pd.DataFrame, healthy_df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in healthy_df.columns if c in {
        "IXI_ID",
        "healthy_Age",
        "healthy_Predicted_Brain_Age",
        "healthy_BAG",
        "healthy_abs_BAG",
        "healthy_Filename",
        "healthy_Path",
        "Model",
    }]

    merged = scaled_df.merge(
        healthy_df[keep_cols],
        on="IXI_ID",
        how="left",
    )

    if merged["healthy_BAG"].isna().any():
        missing = sorted(merged.loc[merged["healthy_BAG"].isna(), "IXI_ID"].unique().tolist())
        raise ValueError(f"Could not match healthy baseline rows for IXI IDs: {missing[:20]}")

    merged["delta_BAG_vs_healthy"] = merged["BAG"] - merged["healthy_BAG"]
    merged["delta_abs_BAG_vs_healthy"] = merged["abs_BAG"] - merged["healthy_abs_BAG"]

    merged["healthy_to_scaled_prediction_shift"] = (
        merged["Predicted_Brain_Age"] - merged["healthy_Predicted_Brain_Age"]
    )

    return merged


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------


def compute_global_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append(("n_rows", len(df)))
    rows.append(("n_subjects", df["IXI_ID"].nunique()))
    rows.append(("n_masks", df["mask_code"].nunique()))
    rows.append(("mean_effective_voxels", df["effective_voxels"].mean()))
    rows.append(("median_effective_voxels", df["effective_voxels"].median()))

    rows.append(("mean_scaled_BAG", df["BAG"].mean()))
    rows.append(("mean_scaled_abs_BAG", df["abs_BAG"].mean()))
    rows.append(("mean_healthy_BAG", df["healthy_BAG"].mean()))
    rows.append(("mean_healthy_abs_BAG", df["healthy_abs_BAG"].mean()))

    rows.append(("mean_delta_BAG_vs_healthy", df["delta_BAG_vs_healthy"].mean()))
    rows.append(("mean_delta_abs_BAG_vs_healthy", df["delta_abs_BAG_vs_healthy"].mean()))
    rows.append(("median_delta_abs_BAG_vs_healthy", df["delta_abs_BAG_vs_healthy"].median()))

    if len(df) >= 2 and df["effective_voxels"].nunique() >= 2:
        rows.append(("corr_voxels_vs_scaled_BAG", df["effective_voxels"].corr(df["BAG"])))
        rows.append(("corr_voxels_vs_scaled_abs_BAG", df["effective_voxels"].corr(df["abs_BAG"])))
        rows.append(("corr_voxels_vs_delta_BAG_vs_healthy", df["effective_voxels"].corr(df["delta_BAG_vs_healthy"])))
        rows.append(("corr_voxels_vs_delta_abs_BAG_vs_healthy", df["effective_voxels"].corr(df["delta_abs_BAG_vs_healthy"])))

    return pd.DataFrame(rows, columns=["metric", "value"])


def compute_voxel_bin_summary(df: pd.DataFrame, n_bins: int = 8) -> pd.DataFrame:
    tmp = df.copy()

    try:
        tmp["voxel_bin"] = pd.qcut(tmp["effective_voxels"], q=n_bins, duplicates="drop")
    except ValueError:
        tmp["voxel_bin"] = pd.cut(tmp["effective_voxels"], bins=n_bins)

    summary = (
        tmp.groupby("voxel_bin", observed=False, as_index=False)
        .agg(
            n=("IXI_ID", "size"),
            mean_effective_voxels=("effective_voxels", "mean"),
            median_effective_voxels=("effective_voxels", "median"),
            mean_scaled_BAG=("BAG", "mean"),
            mean_scaled_abs_BAG=("abs_BAG", "mean"),
            mean_healthy_BAG=("healthy_BAG", "mean"),
            mean_healthy_abs_BAG=("healthy_abs_BAG", "mean"),
            mean_delta_BAG_vs_healthy=("delta_BAG_vs_healthy", "mean"),
            std_delta_BAG_vs_healthy=("delta_BAG_vs_healthy", "std"),
            mean_delta_abs_BAG_vs_healthy=("delta_abs_BAG_vs_healthy", "mean"),
            std_delta_abs_BAG_vs_healthy=("delta_abs_BAG_vs_healthy", "std"),
        )
        .sort_values("mean_effective_voxels")
        .reset_index(drop=True)
    )

    return summary


def compute_subject_mask_slopes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (ixi_id, mask_code), group in df.groupby(["IXI_ID", "mask_code"]):
        group = group.sort_values("effective_voxels")

        if group["effective_voxels"].nunique() < 2:
            continue

        x = group["effective_voxels"].to_numpy(dtype=float)
        y1 = group["delta_BAG_vs_healthy"].to_numpy(dtype=float)
        y2 = group["delta_abs_BAG_vs_healthy"].to_numpy(dtype=float)

        slope1 = float(np.polyfit(x, y1, 1)[0])
        slope2 = float(np.polyfit(x, y2, 1)[0])

        rows.append(
            {
                "IXI_ID": ixi_id,
                "mask_code": mask_code,
                "n_points": len(group),
                "min_effective_voxels": float(group["effective_voxels"].min()),
                "max_effective_voxels": float(group["effective_voxels"].max()),
                "delta_BAG_vs_healthy_slope_per_voxel": slope1,
                "delta_abs_BAG_vs_healthy_slope_per_voxel": slope2,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_delta_bag_vs_voxels(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(df["effective_voxels"], df["delta_BAG_vs_healthy"], alpha=0.45)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Effective tumor voxels")
    plt.ylabel("ΔBAG vs healthy baseline")
    plt.title("Tumor-size effect on BAG relative to healthy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_delta_abs_bag_vs_voxels(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(df["effective_voxels"], df["delta_abs_BAG_vs_healthy"], alpha=0.45)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Effective tumor voxels")
    plt.ylabel("Δ|BAG| vs healthy baseline")
    plt.title("Tumor-size effect on absolute BAG relative to healthy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_mean_delta_abs_bag_by_voxel_bin(summary: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        summary["mean_effective_voxels"],
        summary["mean_delta_abs_BAG_vs_healthy"],
        yerr=summary["std_delta_abs_BAG_vs_healthy"],
        marker="o",
        capsize=4,
    )
    plt.axhline(0, linewidth=1)
    plt.xlabel("Mean effective tumor voxels (bin)")
    plt.ylabel("Mean Δ|BAG| vs healthy")
    plt.title("Mean increase in |BAG| as tumor voxels increase")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_mean_delta_bag_by_voxel_bin(summary: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        summary["mean_effective_voxels"],
        summary["mean_delta_BAG_vs_healthy"],
        yerr=summary["std_delta_BAG_vs_healthy"],
        marker="o",
        capsize=4,
    )
    plt.axhline(0, linewidth=1)
    plt.xlabel("Mean effective tumor voxels (bin)")
    plt.ylabel("Mean ΔBAG vs healthy")
    plt.title("Mean BAG shift as tumor voxels increase")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_subject_mask_delta_abs_bag_trajectories(df: pd.DataFrame, out_path: Path, max_groups: int = 30) -> None:
    plt.figure(figsize=(10, 6))

    grouped = list(df.groupby(["IXI_ID", "mask_code"]))
    grouped = grouped[:max_groups]

    for (_, _), group in grouped:
        group = group.sort_values("effective_voxels")
        plt.plot(
            group["effective_voxels"],
            group["delta_abs_BAG_vs_healthy"],
            marker="o",
            alpha=0.5,
        )

    plt.axhline(0, linewidth=1)
    plt.xlabel("Effective tumor voxels")
    plt.ylabel("Δ|BAG| vs healthy")
    plt.title("Per subject/mask |BAG| shift vs healthy baseline")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_subject_mask_delta_bag_trajectories(df: pd.DataFrame, out_path: Path, max_groups: int = 30) -> None:
    plt.figure(figsize=(10, 6))

    grouped = list(df.groupby(["IXI_ID", "mask_code"]))
    grouped = grouped[:max_groups]

    for (_, _), group in grouped:
        group = group.sort_values("effective_voxels")
        plt.plot(
            group["effective_voxels"],
            group["delta_BAG_vs_healthy"],
            marker="o",
            alpha=0.5,
        )

    plt.axhline(0, linewidth=1)
    plt.xlabel("Effective tumor voxels")
    plt.ylabel("ΔBAG vs healthy")
    plt.title("Per subject/mask BAG shift vs healthy baseline")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_delta_abs_bag_slope_histogram(slopes_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(slopes_df["delta_abs_BAG_vs_healthy_slope_per_voxel"], bins=30)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Slope of Δ|BAG| vs effective voxels")
    plt.ylabel("Count")
    plt.title("Distribution of tumor-size sensitivity slopes")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scaled_vs_healthy_abs_bag(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(df["healthy_abs_BAG"], df["abs_BAG"], alpha=0.4)
    all_vals = np.concatenate([df["healthy_abs_BAG"].values, df["abs_BAG"].values])
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))
    plt.plot([vmin, vmax], [vmin, vmax], linewidth=1)
    plt.xlabel("Healthy |BAG|")
    plt.ylabel("Scaled synthetic |BAG|")
    plt.title("Healthy vs scaled synthetic absolute BAG")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------------------------------------------------------
# Text summary
# -----------------------------------------------------------------------------


def write_text_summary(df: pd.DataFrame, summary_bins: pd.DataFrame, slopes_df: pd.DataFrame, out_path: Path) -> None:
    lines = []
    lines.append(f"Total scaled rows: {len(df)}")
    lines.append(f"Unique subjects: {df['IXI_ID'].nunique()}")
    lines.append(f"Unique masks: {df['mask_code'].nunique()}")
    lines.append("")
    lines.append("Effective voxel count summary:")
    lines.append(f"  min    = {df['effective_voxels'].min():.4f}")
    lines.append(f"  max    = {df['effective_voxels'].max():.4f}")
    lines.append(f"  mean   = {df['effective_voxels'].mean():.4f}")
    lines.append(f"  median = {df['effective_voxels'].median():.4f}")
    lines.append("")

    lines.append("Scaled BAG summary:")
    lines.append(f"  mean BAG       = {df['BAG'].mean():.4f}")
    lines.append(f"  mean |BAG|     = {df['abs_BAG'].mean():.4f}")
    lines.append("")

    lines.append("Healthy baseline BAG summary:")
    lines.append(f"  mean BAG       = {df['healthy_BAG'].mean():.4f}")
    lines.append(f"  mean |BAG|     = {df['healthy_abs_BAG'].mean():.4f}")
    lines.append("")

    lines.append("Difference vs healthy baseline:")
    lines.append(f"  mean ΔBAG      = {df['delta_BAG_vs_healthy'].mean():.4f}")
    lines.append(f"  mean Δ|BAG|    = {df['delta_abs_BAG_vs_healthy'].mean():.4f}")
    lines.append(f"  median Δ|BAG|  = {df['delta_abs_BAG_vs_healthy'].median():.4f}")
    lines.append("")

    if len(df) >= 2 and df["effective_voxels"].nunique() >= 2:
        lines.append("Correlations:")
        lines.append(f"  corr(voxels, ΔBAG)   = {df['effective_voxels'].corr(df['delta_BAG_vs_healthy']):.6f}")
        lines.append(f"  corr(voxels, Δ|BAG|) = {df['effective_voxels'].corr(df['delta_abs_BAG_vs_healthy']):.6f}")
        lines.append("")

    if not slopes_df.empty:
        lines.append("Within subject/mask slope summary:")
        lines.append(
            f"  mean slope ΔBAG/voxel      = {slopes_df['delta_BAG_vs_healthy_slope_per_voxel'].mean():.8f}"
        )
        lines.append(
            f"  mean slope Δ|BAG|/voxel    = {slopes_df['delta_abs_BAG_vs_healthy_slope_per_voxel'].mean():.8f}"
        )
        lines.append("")

    lines.append("Voxel-bin summary:")
    for _, row in summary_bins.iterrows():
        lines.append(
            f"  mean_vox={row['mean_effective_voxels']:.2f} | "
            f"n={int(row['n'])} | "
            f"mean_ΔBAG={row['mean_delta_BAG_vs_healthy']:.4f} | "
            f"mean_Δ|BAG|={row['mean_delta_abs_BAG_vs_healthy']:.4f}"
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare scaled IXI+BraTS predictions against healthy IXI-only baseline, focusing on tumor-size effect on BAG."
    )
    parser.add_argument("--scaled-csv", type=Path, required=True)
    parser.add_argument("--ixi-only-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-subject-lines", type=int, default=30)
    parser.add_argument("--n-voxel-bins", type=int, default=8)
    args = parser.parse_args()

    if not args.scaled_csv.exists():
        raise FileNotFoundError(f"Scaled CSV not found: {args.scaled_csv}")
    if not args.ixi_only_csv.exists():
        raise FileNotFoundError(f"IXI-only CSV not found: {args.ixi_only_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    scaled_df = load_scaled(args.scaled_csv)
    healthy_df = load_ixi_only(args.ixi_only_csv)

    df = merge_scaled_with_healthy(scaled_df, healthy_df)
    df.to_csv(args.output_dir / "scaled_vs_healthy_merged.csv", index=False)

    global_summary = compute_global_summary(df)
    global_summary.to_csv(args.output_dir / "global_summary.csv", index=False)

    summary_bins = compute_voxel_bin_summary(df, n_bins=args.n_voxel_bins)
    summary_bins.to_csv(args.output_dir / "summary_by_voxel_bins.csv", index=False)

    slopes_df = compute_subject_mask_slopes(df)
    slopes_df.to_csv(args.output_dir / "subject_mask_slopes.csv", index=False)

    plot_delta_bag_vs_voxels(df, args.output_dir / "delta_bag_vs_effective_voxels.png")
    plot_delta_abs_bag_vs_voxels(df, args.output_dir / "delta_abs_bag_vs_effective_voxels.png")
    plot_mean_delta_bag_by_voxel_bin(summary_bins, args.output_dir / "mean_delta_bag_by_voxel_bins.png")
    plot_mean_delta_abs_bag_by_voxel_bin(summary_bins, args.output_dir / "mean_delta_abs_bag_by_voxel_bins.png")
    plot_subject_mask_delta_bag_trajectories(
        df,
        args.output_dir / "subject_mask_delta_bag_trajectories.png",
        max_groups=args.max_subject_lines,
    )
    plot_subject_mask_delta_abs_bag_trajectories(
        df,
        args.output_dir / "subject_mask_delta_abs_bag_trajectories.png",
        max_groups=args.max_subject_lines,
    )
    plot_scaled_vs_healthy_abs_bag(df, args.output_dir / "scaled_vs_healthy_abs_bag.png")

    if not slopes_df.empty:
        plot_delta_abs_bag_slope_histogram(
            slopes_df,
            args.output_dir / "delta_abs_bag_slope_histogram.png",
        )

    write_text_summary(df, summary_bins, slopes_df, args.output_dir / "summary.txt")

    print(f"Done. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
