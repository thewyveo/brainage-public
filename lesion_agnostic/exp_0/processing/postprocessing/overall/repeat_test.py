#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Analyze repeatability of repeated BrainAgeNeXt inference runs.


Example:
py -3.10 exp_0\processing\postprocessing\overall\repeat_test.py `
  --runs-dir "exp_0\results\BANXt_determinism_proof" `
  --output-dir "exp_0\results\BANXt_determinism_proof\repeatability_analysis" `
  --tolerance 1e-6
"""


from __future__ import annotations


import argparse
from pathlib import Path


import numpy as np
import pandas as pd




def load_run_csvs(runs_dir: Path) -> list[Path]:
    csvs = sorted(runs_dir.glob("brainagenext_predictions_run_*.csv"))
    if not csvs:
        raise RuntimeError(f"No run CSVs found in: {runs_dir}")
    return csvs




def load_and_align(csvs: list[Path]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    merged = None
    pba_cols = []
    bad_cols = []


    for idx, csv_path in enumerate(csvs, start=1):
        df = pd.read_csv(csv_path)


        required = ["IXI_ID", "Age", "Filename", "Path", "Predicted_Brain_Age", "Brain_Age_Difference"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{csv_path} missing columns: {missing}")


        df = df[required].copy()
        pba_col = f"PBA_run_{idx:03d}"
        bad_col = f"BAD_run_{idx:03d}"
        df = df.rename(
            columns={
                "Predicted_Brain_Age": pba_col,
                "Brain_Age_Difference": bad_col,
            }
        )


        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["IXI_ID", "Age", "Filename", "Path"], how="inner")


        pba_cols.append(pba_col)
        bad_cols.append(bad_col)


    if merged is None or len(merged) == 0:
        raise RuntimeError("No overlapping scans across runs.")


    pba = merged[pba_cols].to_numpy(dtype=np.float64)
    bad = merged[bad_cols].to_numpy(dtype=np.float64)


    return merged, pba, bad




def summarize_repeatability(values: np.ndarray, prefix: str, tol: float) -> pd.DataFrame:
    row_min = values.min(axis=1)
    row_max = values.max(axis=1)
    row_mean = values.mean(axis=1)
    row_std = values.std(axis=1, ddof=0)
    row_range = row_max - row_min


    exact_equal = np.all(values == values[:, [0]], axis=1)
    near_equal = np.all(np.abs(values - values[:, [0]]) <= tol, axis=1)


    df = pd.DataFrame(
        {
            f"{prefix}_mean": row_mean,
            f"{prefix}_std": row_std,
            f"{prefix}_min": row_min,
            f"{prefix}_max": row_max,
            f"{prefix}_range": row_range,
            f"{prefix}_exact_equal": exact_equal,
            f"{prefix}_within_tol": near_equal,
        }
    )
    return df




def pairwise_run_differences(values: np.ndarray) -> dict:
    n_runs = values.shape[1]
    max_abs = []
    mean_abs = []


    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            diff = np.abs(values[:, i] - values[:, j])
            max_abs.append(float(diff.max()))
            mean_abs.append(float(diff.mean()))


    return {
        "pairwise_max_abs_diff_overall": float(np.max(max_abs)) if max_abs else 0.0,
        "pairwise_mean_abs_diff_overall": float(np.mean(mean_abs)) if mean_abs else 0.0,
    }




def write_report(
    out_path: Path,
    n_scans: int,
    n_runs: int,
    tol: float,
    pba_summary: pd.DataFrame,
    bad_summary: pd.DataFrame,
    pba_pairwise: dict,
    bad_pairwise: dict,
) -> None:
    lines = []
    lines.append(f"Scans analyzed: {n_scans}")
    lines.append(f"Runs analyzed: {n_runs}")
    lines.append(f"Tolerance: {tol}")
    lines.append("")
    lines.append("Predicted Brain Age repeatability")
    lines.append(f"  Exact repeatability count       = {int(pba_summary['PBA_exact_equal'].sum())}/{n_scans}")
    lines.append(f"  Within tolerance count          = {int(pba_summary['PBA_within_tol'].sum())}/{n_scans}")
    lines.append(f"  Max per-scan range              = {float(pba_summary['PBA_range'].max()):.12f}")
    lines.append(f"  Mean per-scan range             = {float(pba_summary['PBA_range'].mean()):.12f}")
    lines.append(f"  Mean per-scan std               = {float(pba_summary['PBA_std'].mean()):.12f}")
    lines.append(f"  Pairwise max abs diff overall   = {pba_pairwise['pairwise_max_abs_diff_overall']:.12f}")
    lines.append(f"  Pairwise mean abs diff overall  = {pba_pairwise['pairwise_mean_abs_diff_overall']:.12f}")
    lines.append("")
    lines.append("Brain Age Difference repeatability")
    lines.append(f"  Exact repeatability count       = {int(bad_summary['BAD_exact_equal'].sum())}/{n_scans}")
    lines.append(f"  Within tolerance count          = {int(bad_summary['BAD_within_tol'].sum())}/{n_scans}")
    lines.append(f"  Max per-scan range              = {float(bad_summary['BAD_range'].max()):.12f}")
    lines.append(f"  Mean per-scan range             = {float(bad_summary['BAD_range'].mean()):.12f}")
    lines.append(f"  Mean per-scan std               = {float(bad_summary['BAD_std'].mean()):.12f}")
    lines.append(f"  Pairwise max abs diff overall   = {bad_pairwise['pairwise_max_abs_diff_overall']:.12f}")
    lines.append(f"  Pairwise mean abs diff overall  = {bad_pairwise['pairwise_mean_abs_diff_overall']:.12f}")


    out_path.write_text("\n".join(lines), encoding="utf-8")




def main() -> None:
    parser = argparse.ArgumentParser(description="Test repeatability of repeated BrainAgeNeXt inference runs.")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()


    args.output_dir.mkdir(parents=True, exist_ok=True)


    csvs = load_run_csvs(args.runs_dir)
    merged, pba, bad = load_and_align(csvs)


    pba_summary = summarize_repeatability(pba, prefix="PBA", tol=args.tolerance)
    bad_summary = summarize_repeatability(bad, prefix="BAD", tol=args.tolerance)


    merged_out = pd.concat([merged[["IXI_ID", "Age", "Filename", "Path"]], pba_summary, bad_summary], axis=1)
    merged_out.to_csv(args.output_dir / "repeatability_per_scan.csv", index=False)


    pba_pairwise = pairwise_run_differences(pba)
    bad_pairwise = pairwise_run_differences(bad)


    write_report(
        out_path=args.output_dir / "repeatability_report.txt",
        n_scans=len(merged),
        n_runs=pba.shape[1],
        tol=args.tolerance,
        pba_summary=pba_summary,
        bad_summary=bad_summary,
        pba_pairwise=pba_pairwise,
        bad_pairwise=bad_pairwise,
    )


    print(f"Saved per-scan repeatability CSV to: {args.output_dir / 'repeatability_per_scan.csv'}")
    print(f"Saved report to: {args.output_dir / 'repeatability_report.txt'}")




if __name__ == "__main__":
    main()

