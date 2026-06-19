#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Statistical testing for baseline vs synthetic brain-age predictions.


Expected CSV format for BOTH --baseline-csv and --synthetic-csv:
IXI_ID
Age
Filename
Path
Predicted_Brain_Age
Brain_Age_Difference


Example
-------
py -3.10 exp_0\processing\postprocessing\overall\significance.py `
  --baseline-csv "data\predictions\BrainAgeNeXt\ixi_only_faithfullpreprocess.csv" `
  --synthetic-csv "exp_0\results\GliGAN_FAITHFUL\brainagenext_predictions_run_001.csv" `
  --output-dir "exp_0\results\true_gligan" `
  --permutations 10000
"""


from __future__ import annotations


import argparse
from pathlib import Path


import numpy as np
import pandas as pd
from scipy import stats




# =============================================================================
# Loading and preparation
# =============================================================================


REQUIRED_COLS = [
    "IXI_ID",
    "Age",
    "Filename",
    "Path",
    "Predicted_Brain_Age",
    "Brain_Age_Difference",
]




def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)


    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"{path} missing required column: {col}")


    df = df.copy()
    df["IXI_ID"] = df["IXI_ID"].astype(str).str.strip()
    return df




def merge_csvs(baseline_csv: Path, synthetic_csv: Path) -> pd.DataFrame:
    df_base = load_csv(baseline_csv).rename(
        columns={
            "Age": "Age_base",
            "Filename": "Filename_base",
            "Path": "Path_base",
            "Predicted_Brain_Age": "PBA_base",
            "Brain_Age_Difference": "BAD_base",
        }
    )


    df_synth = load_csv(synthetic_csv).rename(
        columns={
            "Age": "Age_synth",
            "Filename": "Filename_synth",
            "Path": "Path_synth",
            "Predicted_Brain_Age": "PBA_synth",
            "Brain_Age_Difference": "BAD_synth",
        }
    )


    merged = pd.merge(
        df_base[["IXI_ID", "Age_base", "Filename_base", "Path_base", "PBA_base", "BAD_base"]],
        df_synth[["IXI_ID", "Age_synth", "Filename_synth", "Path_synth", "PBA_synth", "BAD_synth"]],
        on="IXI_ID",
        how="inner",
    )


    if merged.empty:
        raise RuntimeError("No overlapping IXI_ID values found between baseline and synthetic CSVs.")


    # Age should match for same subject; keep baseline age as canonical
    merged["Age"] = merged["Age_base"]


    # Prediction shifts
    merged["PBA_diff"] = merged["PBA_synth"] - merged["PBA_base"]
    merged["BAD_diff"] = merged["BAD_synth"] - merged["BAD_base"]


    # Absolute shifts
    merged["PBA_abs_diff"] = np.abs(merged["PBA_diff"])
    merged["BAD_abs_diff"] = np.abs(merged["BAD_diff"])


    # Absolute BAG / absolute age error
    merged["ABS_BAD_base"] = np.abs(merged["BAD_base"])
    merged["ABS_BAD_synth"] = np.abs(merged["BAD_synth"])
    merged["ABS_BAD_diff"] = merged["ABS_BAD_synth"] - merged["ABS_BAD_base"]


    return merged




# =============================================================================
# Statistics helpers
# =============================================================================


def mean_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(np.mean(values))


    if n < 2:
        return mean, mean


    se = stats.sem(values, nan_policy="omit")
    if not np.isfinite(se):
        return mean, mean


    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return mean - tcrit * se, mean + tcrit * se




def cohens_d_paired(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return 0.0
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diff.mean() / sd)




def rank_biserial_from_wilcoxon(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return 0.0


    ranks = stats.rankdata(np.abs(diff))
    w_pos = ranks[diff > 0].sum()
    w_neg = ranks[diff < 0].sum()
    denom = n * (n + 1) / 2.0
    return float((w_pos - w_neg) / denom)




def sign_flip_permutation_test(diff: np.ndarray, n_perm: int = 10000, seed: int = 42) -> tuple[float, float]:
    """
    Two-sided sign-flip permutation test on the mean of paired differences.
    """
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    observed = float(np.mean(diff))


    if len(diff) == 0:
        return observed, 1.0


    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    perm_means = np.mean(signs * diff[None, :], axis=1)
    p = (np.sum(np.abs(perm_means) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, float(p)




# =============================================================================
# Report blocks
# =============================================================================


def run_signed_shift_tests(diff: np.ndarray, label: str, permutations: int) -> list[str]:
    """
    Tests whether the mean/median signed shift differs from 0.
    This is the key block for directional bias.
    """
    lines = []
    diff = np.asarray(diff, dtype=float)
    n = len(diff)


    ci_lo, ci_hi = mean_ci(diff)


    lines.append(label)
    lines.append(f"  n                           = {n}")
    lines.append(f"  mean                        = {diff.mean():.6f}")
    lines.append(f"  median                      = {np.median(diff):.6f}")
    lines.append(f"  std                         = {diff.std(ddof=1):.6f}" if n > 1 else "  std                         = 0.000000")
    lines.append(f"  95% CI of mean              = [{ci_lo:.6f}, {ci_hi:.6f}]")


    if n >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(diff)
        lines.append(f"  Shapiro-Wilk W              = {shapiro_stat:.6f}")
        lines.append(f"  Shapiro-Wilk p              = {shapiro_p:.6g}")
    else:
        lines.append("  Shapiro-Wilk                = not enough samples")


    if n >= 2:
        t_stat, t_p = stats.ttest_1samp(diff, popmean=0.0)
        lines.append(f"  one-sample t statistic      = {t_stat:.6f}")
        lines.append(f"  one-sample t p-value        = {t_p:.6g}")
    else:
        lines.append("  one-sample t-test           = not enough samples")


    try:
        w_stat, w_p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        lines.append(f"  Wilcoxon statistic          = {w_stat:.6f}")
        lines.append(f"  Wilcoxon p-value            = {w_p:.6g}")
    except ValueError as e:
        lines.append(f"  Wilcoxon                    = not available ({e})")


    lines.append(f"  Cohen's d (paired)          = {cohens_d_paired(diff):.6f}")


    try:
        rbc = rank_biserial_from_wilcoxon(diff)
        lines.append(f"  rank-biserial correlation   = {rbc:.6f}")
    except Exception as e:
        lines.append(f"  rank-biserial correlation   = unavailable ({e})")


    perm_mean, perm_p = sign_flip_permutation_test(diff, n_perm=permutations)
    lines.append(f"  permutation mean            = {perm_mean:.6f}")
    lines.append(f"  permutation p-value         = {perm_p:.6g}")
    lines.append("")
    return lines




def run_absolute_shift_tests(values: np.ndarray, label: str, permutations: int) -> list[str]:
    """
    Tests whether the absolute shift is > 0.
    This is the key block for non-directional perturbation / sensitivity.
    """
    lines = []
    values = np.asarray(values, dtype=float)
    n = len(values)


    ci_lo, ci_hi = mean_ci(values)


    lines.append(label)
    lines.append(f"  n                           = {n}")
    lines.append(f"  mean                        = {values.mean():.6f}")
    lines.append(f"  median                      = {np.median(values):.6f}")
    lines.append(f"  std                         = {values.std(ddof=1):.6f}" if n > 1 else "  std                         = 0.000000")
    lines.append(f"  95% CI of mean              = [{ci_lo:.6f}, {ci_hi:.6f}]")


    if n >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(values)
        lines.append(f"  Shapiro-Wilk W              = {shapiro_stat:.6f}")
        lines.append(f"  Shapiro-Wilk p              = {shapiro_p:.6g}")
    else:
        lines.append("  Shapiro-Wilk                = not enough samples")


    if n >= 2:
        t_stat, t_p = stats.ttest_1samp(values, popmean=0.0)
        lines.append(f"  one-sample t statistic      = {t_stat:.6f}")
        lines.append(f"  one-sample t p-value        = {t_p:.6g}")
    else:
        lines.append("  one-sample t-test           = not enough samples")


    try:
        # one-sided because values are nonnegative and question is whether mean shift > 0
        w_stat, w_p = stats.wilcoxon(values, zero_method="wilcox", alternative="greater")
        lines.append(f"  Wilcoxon statistic          = {w_stat:.6f}")
        lines.append(f"  Wilcoxon p-value (> 0)      = {w_p:.6g}")
    except ValueError as e:
        lines.append(f"  Wilcoxon                    = not available ({e})")


    perm_mean, perm_p = sign_flip_permutation_test(values, n_perm=permutations)
    lines.append(f"  permutation mean            = {perm_mean:.6f}")
    lines.append(f"  permutation p-value         = {perm_p:.6g}")
    lines.append("")
    return lines




def run_abs_bag_paired_test(abs_bad_base: np.ndarray, abs_bad_synth: np.ndarray) -> list[str]:
    """
    Tests whether |BAG| changed after adding synthetic tumor.
    Since |BAG| = absolute age prediction error per subject, this is effectively a paired MAE-style analysis.
    """
    lines = []
    abs_bad_base = np.asarray(abs_bad_base, dtype=float)
    abs_bad_synth = np.asarray(abs_bad_synth, dtype=float)
    diff = abs_bad_synth - abs_bad_base


    lines.append("Paired absolute error test: |BAG| / absolute age-prediction error")
    lines.append(f"  baseline mean |BAG|         = {abs_bad_base.mean():.6f}")
    lines.append(f"  synthetic mean |BAG|        = {abs_bad_synth.mean():.6f}")
    lines.append(f"  mean paired difference      = {diff.mean():.6f}")


    if len(diff) >= 2:
        t_stat, t_p = stats.ttest_rel(abs_bad_synth, abs_bad_base)
        lines.append(f"  paired t statistic          = {t_stat:.6f}")
        lines.append(f"  paired t p-value            = {t_p:.6g}")
    else:
        lines.append("  paired t-test               = not enough samples")


    try:
        w_stat, w_p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        lines.append(f"  Wilcoxon statistic          = {w_stat:.6f}")
        lines.append(f"  Wilcoxon p-value            = {w_p:.6g}")
    except ValueError as e:
        lines.append(f"  Wilcoxon                    = not available ({e})")


    lines.append(f"  Cohen's d (paired)          = {cohens_d_paired(diff):.6f}")
    lines.append("")
    return lines




# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Statistical testing for baseline vs synthetic brain-age predictions."
    )
    parser.add_argument("--baseline-csv", type=Path, required=True, help="Baseline prediction CSV.")
    parser.add_argument("--synthetic-csv", type=Path, required=True, help="Synthetic-tumor prediction CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for report outputs.")
    parser.add_argument("--permutations", type=int, default=10000, help="Number of sign-flip permutations.")
    args = parser.parse_args()


    df = merge_csvs(args.baseline_csv, args.synthetic_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)


    # Save merged paired table for transparency
    paired_csv = args.output_dir / "paired_comparison.csv"
    df.to_csv(paired_csv, index=False)


    lines = []
    lines.append("PAIRED STATISTICAL TESTS")
    lines.append("========================")
    lines.append(f"Subjects: {len(df)}")
    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("  - Signed shift tests evaluate directional bias (systematic upward/downward change).")
    lines.append("  - Absolute shift tests evaluate non-directional perturbation / lesion sensitivity.")
    lines.append("  - |BAG| compares absolute age-prediction error before vs after synthetic tumor insertion.")
    lines.append("")


    lines.extend(
        run_signed_shift_tests(
            df["PBA_diff"].values,
            "Signed predicted brain age shift: PBA_synth - PBA_base",
            args.permutations,
        )
    )


    lines.extend(
        run_signed_shift_tests(
            df["BAD_diff"].values,
            "Signed BAG shift: BAG_synth - BAG_base",
            args.permutations,
        )
    )


    lines.extend(
        run_absolute_shift_tests(
            df["PBA_abs_diff"].values,
            "Absolute predicted brain age difference: |PBA_synth - PBA_base|",
            args.permutations,
        )
    )


    lines.extend(
        run_absolute_shift_tests(
            df["BAD_abs_diff"].values,
            "Absolute BAG difference: |BAG_synth - BAG_base|",
            args.permutations,
        )
    )


    lines.extend(
        run_abs_bag_paired_test(
            df["ABS_BAD_base"].values,
            df["ABS_BAD_synth"].values,
        )
    )


    out_txt = args.output_dir / "paired_stats_report.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")


    summary_table = pd.DataFrame(
        {
            "metric": [
                "n_subjects",
                "mean_PBA_diff",
                "median_PBA_diff",
                "mean_BAD_diff",
                "median_BAD_diff",
                "mean_abs_PBA_diff",
                "median_abs_PBA_diff",
                "mean_abs_BAD_diff",
                "median_abs_BAD_diff",
                "mean_abs_BAG_baseline",
                "mean_abs_BAG_synthetic",
                "mean_abs_BAG_change",
            ],
            "value": [
                len(df),
                float(df["PBA_diff"].mean()),
                float(df["PBA_diff"].median()),
                float(df["BAD_diff"].mean()),
                float(df["BAD_diff"].median()),
                float(df["PBA_abs_diff"].mean()),
                float(df["PBA_abs_diff"].median()),
                float(df["BAD_abs_diff"].mean()),
                float(df["BAD_abs_diff"].median()),
                float(df["ABS_BAD_base"].mean()),
                float(df["ABS_BAD_synth"].mean()),
                float((df["ABS_BAD_synth"] - df["ABS_BAD_base"]).mean()),
            ],
        }
    )
    out_csv = args.output_dir / "paired_stats_summary.csv"
    summary_table.to_csv(out_csv, index=False)


    print(f"Saved paired comparison table to: {paired_csv}")
    print(f"Saved statistical report to: {out_txt}")
    print(f"Saved summary table to: {out_csv}")




if __name__ == "__main__":
    main()

