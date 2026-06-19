#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import argparse
import re


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd




r"""
py -3.10 exp_0\processing\postprocessing\overall\scale_comparison.py `
    --predictions-csv data\predictions\BrainAgeNeXt\IXI_BraTS_SCALED_full2.csv `
    --voxels-csv exp_0\processing\preprocessing\utils\insights\BraTS_Mask_Analysis_2\all_tumors.csv `
    --output-dir exp_0\processing\postprocessing\model_specific\BrainAgeNeXt\insights\scale_effect_full
"""




def extract_scale_from_filename(filename: str) -> float:
    """
    Example:
    IXI002__B00005-100__s0p33.nii.gz -> 0.33
    """
    name = Path(filename).name
    name = name.replace("_preprocessed", "")
    match = re.search(r"__s([0-9]+(?:p[0-9]+)?)\.nii(?:\.gz)?$", name)
    if not match:
        raise ValueError(f"Could not parse scale from filename: {filename}")
    return float(match.group(1).replace("p", "."))




def extract_mask_code_from_filename(filename: str) -> str:
    """
    Example:
    IXI002__B00005-100__s0p33.nii.gz -> B00005-100
    """
    name = Path(filename).name
    match = re.search(r"__(B[0-9]{5}-[0-9]{3})__s", name)
    if not match:
        raise ValueError(f"Could not parse mask code from filename: {filename}")
    return match.group(1)




def extract_mask_code_from_case(case_value: str) -> str:
    """
    Example:
    BraTS-GLI-00005-100-seg -> B00005-100
    """
    s = str(case_value).strip()
    match = re.search(r"BraTS-GLI-([0-9]{5}-[0-9]{3})", s)
    if not match:
        raise ValueError(f"Could not parse BraTS case code from voxel dataset row: {case_value}")
    return f"B{match.group(1)}"




def load_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)


    required_cols = [
        "IXI_ID",
        "Age",
        "Filename",
        "Predicted_Brain_Age",
        "Brain_Age_Difference",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in predictions CSV")


    df = df.copy()
    df["IXI_ID"] = df["IXI_ID"].astype(str).str.strip()
    df["Age"] = pd.to_numeric(df["Age"], errors="raise")
    df["Predicted_Brain_Age"] = pd.to_numeric(df["Predicted_Brain_Age"], errors="raise")
    df["Brain_Age_Difference"] = pd.to_numeric(df["Brain_Age_Difference"], errors="raise")


    df["scale"] = df["Filename"].apply(extract_scale_from_filename)
    df["mask_code"] = df["Filename"].apply(extract_mask_code_from_filename)


    # BAG = predicted - ground truth age
    # Prefer explicit BAG column, but recompute to be safe / consistent.
    df["BAG"] = df["Predicted_Brain_Age"] - df["Age"]


    return df




def load_voxel_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)


    required_cols = ["case", "voxels"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in voxel CSV")


    df = df.copy()
    df["mask_code"] = df["case"].apply(extract_mask_code_from_case)
    df["voxels"] = pd.to_numeric(df["voxels"], errors="raise")


    # Optional extra columns if present
    for col in ["bbox_x", "bbox_y", "bbox_z", "fits_96"]:
        if col not in df.columns:
            df[col] = np.nan


    return df




def merge_predictions_with_voxels(pred_df: pd.DataFrame, vox_df: pd.DataFrame) -> pd.DataFrame:
    vox_small = vox_df[["mask_code", "voxels", "bbox_x", "bbox_y", "bbox_z", "fits_96"]].copy()
    vox_small = vox_small.rename(columns={"voxels": "base_mask_voxels"})


    merged = pred_df.merge(vox_small, on="mask_code", how="left")


    if merged["base_mask_voxels"].isna().any():
        missing = sorted(merged.loc[merged["base_mask_voxels"].isna(), "mask_code"].unique().tolist())
        raise ValueError(f"Could not match voxel information for these mask codes: {missing}")


    # 3D volumetric scaling
    merged["effective_voxels"] = merged["base_mask_voxels"] * (merged["scale"] ** 3)
    merged["abs_BAG"] = merged["BAG"].abs()
    return merged




def add_subject_mask_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (IXI_ID, mask_code), use the smallest effective voxel count as baseline.
    Then compute delta-BAG relative to that baseline.
    """
    df = df.copy()


    df = df.sort_values(["IXI_ID", "mask_code", "effective_voxels"]).reset_index(drop=True)


    baseline = (
        df.groupby(["IXI_ID", "mask_code"], as_index=False)
        .first()[["IXI_ID", "mask_code", "effective_voxels", "BAG", "abs_BAG"]]
        .rename(
            columns={
                "effective_voxels": "baseline_effective_voxels",
                "BAG": "baseline_BAG",
                "abs_BAG": "baseline_abs_BAG",
            }
        )
    )


    df = df.merge(baseline, on=["IXI_ID", "mask_code"], how="left")
    df["delta_voxels_from_baseline"] = df["effective_voxels"] - df["baseline_effective_voxels"]
    df["delta_BAG_from_baseline"] = df["BAG"] - df["baseline_BAG"]
    df["delta_abs_BAG_from_baseline"] = df["abs_BAG"] - df["baseline_abs_BAG"]


    return df




def compute_global_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overall accuracy / BAG metrics on the synthetic set.
    """
    mae = float(np.mean(np.abs(df["Predicted_Brain_Age"] - df["Age"])))
    rmse = float(np.sqrt(np.mean((df["Predicted_Brain_Age"] - df["Age"]) ** 2)))
    mean_bag = float(df["BAG"].mean())
    std_bag = float(df["BAG"].std(ddof=1)) if len(df) > 1 else 0.0
    mean_abs_bag = float(df["abs_BAG"].mean())
    median_abs_bag = float(df["abs_BAG"].median())


    out = pd.DataFrame(
        {
            "metric": [
                "n_rows",
                "n_subjects",
                "n_masks",
                "mean_effective_voxels",
                "median_effective_voxels",
                "mae",
                "rmse",
                "mean_BAG",
                "std_BAG",
                "mean_abs_BAG",
                "median_abs_BAG",
            ],
            "value": [
                len(df),
                df["IXI_ID"].nunique(),
                df["mask_code"].nunique(),
                df["effective_voxels"].mean(),
                df["effective_voxels"].median(),
                mae,
                rmse,
                mean_bag,
                std_bag,
                mean_abs_bag,
                median_abs_bag,
            ],
        }
    )
    return out




def compute_voxel_bin_summary(df: pd.DataFrame, n_bins: int = 8) -> pd.DataFrame:
    """
    Bin by effective voxel count and summarize BAG behavior.
    """
    df = df.copy()


    # qcut may fail if too many duplicates; fallback to cut
    try:
        df["voxel_bin"] = pd.qcut(df["effective_voxels"], q=n_bins, duplicates="drop")
    except ValueError:
        df["voxel_bin"] = pd.cut(df["effective_voxels"], bins=n_bins)


    summary = (
        df.groupby("voxel_bin", observed=False, as_index=False)
        .agg(
            n=("BAG", "size"),
            mean_effective_voxels=("effective_voxels", "mean"),
            median_effective_voxels=("effective_voxels", "median"),
            mean_BAG=("BAG", "mean"),
            std_BAG=("BAG", "std"),
            mean_abs_BAG=("abs_BAG", "mean"),
            std_abs_BAG=("abs_BAG", "std"),
            mean_delta_BAG_from_baseline=("delta_BAG_from_baseline", "mean"),
            mean_delta_abs_BAG_from_baseline=("delta_abs_BAG_from_baseline", "mean"),
        )
        .sort_values("mean_effective_voxels")
        .reset_index(drop=True)
    )


    return summary




def compute_subject_mask_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (subject, mask), fit:
        BAG ~ effective_voxels
        abs_BAG ~ effective_voxels
    and save the slope.
    """
    rows = []


    for (ixi_id, mask_code), group in df.groupby(["IXI_ID", "mask_code"]):
        group = group.sort_values("effective_voxels")


        if group["effective_voxels"].nunique() < 2:
            continue


        x = group["effective_voxels"].to_numpy(dtype=float)
        y_bag = group["BAG"].to_numpy(dtype=float)
        y_abs = group["abs_BAG"].to_numpy(dtype=float)


        bag_slope = float(np.polyfit(x, y_bag, 1)[0])
        abs_slope = float(np.polyfit(x, y_abs, 1)[0])


        rows.append(
            {
                "IXI_ID": ixi_id,
                "mask_code": mask_code,
                "n_points": len(group),
                "min_effective_voxels": float(group["effective_voxels"].min()),
                "max_effective_voxels": float(group["effective_voxels"].max()),
                "bag_slope_per_voxel": bag_slope,
                "abs_bag_slope_per_voxel": abs_slope,
            }
        )


    return pd.DataFrame(rows)




def plot_bag_vs_effective_voxels(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(df["effective_voxels"], df["BAG"], alpha=0.45)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Effective tumor voxels")
    plt.ylabel("BAG = Predicted age - True age")
    plt.title("BAG vs effective tumor voxel count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def plot_abs_bag_vs_effective_voxels(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.scatter(df["effective_voxels"], df["abs_BAG"], alpha=0.45)
    plt.xlabel("Effective tumor voxels")
    plt.ylabel("|BAG|")
    plt.title("Absolute BAG vs effective tumor voxel count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def plot_mean_abs_bag_by_voxel_bin(summary: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        summary["mean_effective_voxels"],
        summary["mean_abs_BAG"],
        yerr=summary["std_abs_BAG"],
        marker="o",
        capsize=4,
    )
    plt.xlabel("Mean effective tumor voxels (bin)")
    plt.ylabel("Mean |BAG|")
    plt.title("Mean absolute BAG vs effective tumor voxel count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def plot_mean_bag_by_voxel_bin(summary: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        summary["mean_effective_voxels"],
        summary["mean_BAG"],
        yerr=summary["std_BAG"],
        marker="o",
        capsize=4,
    )
    plt.axhline(0, linewidth=1)
    plt.xlabel("Mean effective tumor voxels (bin)")
    plt.ylabel("Mean BAG")
    plt.title("Mean BAG vs effective tumor voxel count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def plot_subject_mask_delta_bag_trajectories(df: pd.DataFrame, out_path: Path, max_groups: int = 30) -> None:
    plt.figure(figsize=(10, 6))


    grouped = list(df.groupby(["IXI_ID", "mask_code"]))
    grouped = grouped[:max_groups]


    for (ixi_id, mask_code), group in grouped:
        group = group.sort_values("effective_voxels")
        plt.plot(
            group["effective_voxels"],
            group["delta_BAG_from_baseline"],
            marker="o",
            alpha=0.5,
        )


    plt.axhline(0, linewidth=1)
    plt.xlabel("Effective tumor voxels")
    plt.ylabel("ΔBAG relative to smallest tumor")
    plt.title("Per subject/mask BAG shift as tumor voxels increase")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def plot_subject_mask_delta_abs_bag_trajectories(df: pd.DataFrame, out_path: Path, max_groups: int = 30) -> None:
    plt.figure(figsize=(10, 6))


    grouped = list(df.groupby(["IXI_ID", "mask_code"]))
    grouped = grouped[:max_groups]


    for (ixi_id, mask_code), group in grouped:
        group = group.sort_values("effective_voxels")
        plt.plot(
            group["effective_voxels"],
            group["delta_abs_BAG_from_baseline"],
            marker="o",
            alpha=0.5,
        )


    plt.axhline(0, linewidth=1)
    plt.xlabel("Effective tumor voxels")
    plt.ylabel("Δ|BAG| relative to smallest tumor")
    plt.title("Per subject/mask absolute BAG shift as tumor voxels increase")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def plot_subject_mask_slope_histogram(slopes_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(slopes_df["abs_bag_slope_per_voxel"], bins=30)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Slope of |BAG| vs effective voxels")
    plt.ylabel("Count")
    plt.title("Distribution of within-subject/mask |BAG| slopes")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def plot_bag_vs_effective_voxels_with_trend(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    x = df["effective_voxels"].to_numpy(dtype=float)
    y = df["BAG"].to_numpy(dtype=float)


    plt.scatter(x, y, alpha=0.35)


    if len(df) >= 2 and np.unique(x).size >= 2:
        coeff = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 200)
        yy = coeff[0] * xx + coeff[1]
        plt.plot(xx, yy, linewidth=2)


    plt.axhline(0, linewidth=1)
    plt.xlabel("Effective tumor voxels")
    plt.ylabel("BAG")
    plt.title("BAG vs effective tumor voxels with linear trend")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def write_text_summary(df: pd.DataFrame, voxel_bin_summary: pd.DataFrame, slopes_df: pd.DataFrame, out_path: Path) -> None:
    lines = []
    lines.append(f"Total rows: {len(df)}")
    lines.append(f"Unique IXI subjects: {df['IXI_ID'].nunique()}")
    lines.append(f"Unique donor masks: {df['mask_code'].nunique()}")
    lines.append("")


    lines.append("Effective voxel count summary:")
    lines.append(f"  min    = {df['effective_voxels'].min():.4f}")
    lines.append(f"  max    = {df['effective_voxels'].max():.4f}")
    lines.append(f"  mean   = {df['effective_voxels'].mean():.4f}")
    lines.append(f"  median = {df['effective_voxels'].median():.4f}")
    lines.append("")


    lines.append("BAG summary:")
    lines.append(f"  min        = {df['BAG'].min():.4f}")
    lines.append(f"  max        = {df['BAG'].max():.4f}")
    lines.append(f"  mean       = {df['BAG'].mean():.4f}")
    lines.append(f"  median     = {df['BAG'].median():.4f}")
    lines.append(f"  mean |BAG| = {df['abs_BAG'].mean():.4f}")
    lines.append(f"  median|BAG|= {df['abs_BAG'].median():.4f}")
    lines.append("")


    if len(df) >= 2 and df["effective_voxels"].nunique() >= 2:
        corr_bag = df["effective_voxels"].corr(df["BAG"])
        corr_abs_bag = df["effective_voxels"].corr(df["abs_BAG"])
        lines.append("Correlations:")
        lines.append(f"  corr(effective_voxels, BAG)   = {corr_bag:.6f}")
        lines.append(f"  corr(effective_voxels, |BAG|) = {corr_abs_bag:.6f}")
        lines.append("")


    if not slopes_df.empty:
        lines.append("Within-subject/mask slope summary:")
        lines.append(f"  mean BAG slope per voxel    = {slopes_df['bag_slope_per_voxel'].mean():.8f}")
        lines.append(f"  mean |BAG| slope per voxel  = {slopes_df['abs_bag_slope_per_voxel'].mean():.8f}")
        lines.append(f"  median BAG slope per voxel  = {slopes_df['bag_slope_per_voxel'].median():.8f}")
        lines.append(f"  median |BAG| slope per voxel= {slopes_df['abs_bag_slope_per_voxel'].median():.8f}")
        lines.append("")


    lines.append("Voxel-bin summary:")
    for _, row in voxel_bin_summary.iterrows():
        lines.append(
            f"  mean_vox={row['mean_effective_voxels']:.2f} | "
            f"n={int(row['n'])} | "
            f"mean_BAG={row['mean_BAG']:.4f} | "
            f"mean_|BAG|={row['mean_abs_BAG']:.4f} | "
            f"mean_delta_|BAG|={row['mean_delta_abs_BAG_from_baseline']:.4f}"
        )


    out_path.write_text("\n".join(lines), encoding="utf-8")




def main():
    parser = argparse.ArgumentParser(
        description="Analyze BAG as a function of effective tumor voxel count."
    )
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--voxels-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-subject-lines", type=int, default=30)
    parser.add_argument("--n-voxel-bins", type=int, default=8)
    args = parser.parse_args()


    if not args.predictions_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {args.predictions_csv}")
    if not args.voxels_csv.exists():
        raise FileNotFoundError(f"Voxels CSV not found: {args.voxels_csv}")


    args.output_dir.mkdir(parents=True, exist_ok=True)


    pred_df = load_predictions(args.predictions_csv)
    vox_df = load_voxel_table(args.voxels_csv)


    df = merge_predictions_with_voxels(pred_df, vox_df)
    df = add_subject_mask_baseline(df)


    df.to_csv(args.output_dir / "predictions_with_effective_voxels.csv", index=False)


    metrics_df = compute_global_metrics(df)
    metrics_df.to_csv(args.output_dir / "global_metrics.csv", index=False)


    voxel_bin_summary = compute_voxel_bin_summary(df, n_bins=args.n_voxel_bins)
    voxel_bin_summary.to_csv(args.output_dir / "summary_by_voxel_bins.csv", index=False)


    slopes_df = compute_subject_mask_slopes(df)
    slopes_df.to_csv(args.output_dir / "subject_mask_slopes.csv", index=False)


    plot_bag_vs_effective_voxels(df, args.output_dir / "bag_vs_effective_voxels.png")
    plot_abs_bag_vs_effective_voxels(df, args.output_dir / "abs_bag_vs_effective_voxels.png")
    plot_bag_vs_effective_voxels_with_trend(df, args.output_dir / "bag_vs_effective_voxels_with_trend.png")
    plot_mean_bag_by_voxel_bin(voxel_bin_summary, args.output_dir / "mean_bag_by_voxel_bins.png")
    plot_mean_abs_bag_by_voxel_bin(voxel_bin_summary, args.output_dir / "mean_abs_bag_by_voxel_bins.png")
    plot_subject_mask_delta_bag_trajectories(
        df,
        args.output_dir / "subject_mask_delta_bag_vs_voxels.png",
        args.max_subject_lines,
    )
    plot_subject_mask_delta_abs_bag_trajectories(
        df,
        args.output_dir / "subject_mask_delta_abs_bag_vs_voxels.png",
        args.max_subject_lines,
    )


    if not slopes_df.empty:
        plot_subject_mask_slope_histogram(
            slopes_df,
            args.output_dir / "subject_mask_abs_bag_slope_histogram.png",
        )


    write_text_summary(df, voxel_bin_summary, slopes_df, args.output_dir / "summary.txt")


    print(f"Done. Outputs saved to: {args.output_dir}")




if __name__ == "__main__":
    main()

