#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Example:
py -3.10 exp_0\processing\postprocessing\overall\stat.py `
  --paired-csv "exp_0\results\IXI_BraTS_CM_extended\paired_comparison.csv" `
  --output-dir "exp_0\results\IXI_BraTS_CM_extended\paired_stats" `
  --permutations 10000
"""


from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import stats




def load_paired_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)


    required_cols = [
        "IXI_ID",
        "Age_base",
        "PBA_base",
        "BAD_base",
        "Filename_base",
        "Path_base",
        "Age_synth",
        "PBA_synth",
        "BAD_synth",
        "Filename_synth",
        "Path_synth",
        "Age",
        "PBA_diff",
        "PBA_abs_diff",
        "BAD_diff",
        "BAD_abs_diff",
        "ABS_BAD_base",
        "ABS_BAD_synth",
    ]


    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{path} missing required column: {col}")


    df = df.copy()
    df["IXI_ID"] = df["IXI_ID"].astype(str).str.strip()


    # Derived columns for absolute prediction error against chronological age
    df["BASE_abs_err"] = np.abs(df["PBA_base"] - df["Age"])
    df["SYNTH_abs_err"] = np.abs(df["PBA_synth"] - df["Age"])
    df["ABS_ERR_diff"] = df["SYNTH_abs_err"] - df["BASE_abs_err"]


    return df




def cohens_d_paired(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return 0.0
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return diff.mean() / sd




def rank_biserial_from_wilcoxon(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return 0.0


    ranks = stats.rankdata(np.abs(diff))
    w_pos = ranks[diff > 0].sum()
    w_neg = ranks[diff < 0].sum()
    return (w_pos - w_neg) / (n * (n + 1) / 2.0)




def mean_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()


    if n < 2:
        return mean, mean


    se = stats.sem(values, nan_policy="omit")
    if not np.isfinite(se):
        return mean, mean


    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return mean - tcrit * se, mean + tcrit * se




def sign_flip_permutation_test(diff: np.ndarray, n_perm: int = 10000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    observed = float(np.mean(diff))


    if len(diff) == 0:
        return observed, 1.0


    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    perm_means = np.mean(signs * diff[None, :], axis=1)
    p = (np.sum(np.abs(perm_means) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, float(p)




def run_signed_tests(diff: np.ndarray, label: str, permutations: int) -> list[str]:
    lines = []
    diff = np.asarray(diff, dtype=float)
    n = len(diff)


    lines.append(label)
    lines.append(f"  n                           = {n}")
    lines.append(f"  mean                        = {diff.mean():.6f}")
    lines.append(f"  median                      = {np.median(diff):.6f}")
    lines.append(f"  std                         = {diff.std(ddof=1):.6f}" if n > 1 else "  std                         = 0.000000")
    ci_lo, ci_hi = mean_ci(diff)
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




def run_abs_tests(values: np.ndarray, label: str, permutations: int) -> list[str]:
    lines = []
    values = np.asarray(values, dtype=float)
    n = len(values)


    lines.append(label)
    lines.append(f"  n                           = {n}")
    lines.append(f"  mean                        = {values.mean():.6f}")
    lines.append(f"  median                      = {np.median(values):.6f}")
    lines.append(f"  std                         = {values.std(ddof=1):.6f}" if n > 1 else "  std                         = 0.000000")
    ci_lo, ci_hi = mean_ci(values)
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
        # Since these values are absolute and nonnegative, testing > 0 is sensible.
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




def run_paired_abs_error_test(base_abs: np.ndarray, synth_abs: np.ndarray) -> list[str]:
    lines = []
    base_abs = np.asarray(base_abs, dtype=float)
    synth_abs = np.asarray(synth_abs, dtype=float)
    diff = synth_abs - base_abs


    lines.append("Paired absolute error test: |BAG| / MAE proxy")
    lines.append(f"  baseline mean |BAG|         = {base_abs.mean():.6f}")
    lines.append(f"  synthetic mean |BAG|        = {synth_abs.mean():.6f}")
    lines.append(f"  mean paired difference      = {diff.mean():.6f}")


    if len(diff) >= 2:
        t_stat, t_p = stats.ttest_rel(synth_abs, base_abs)
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




def main():
    parser = argparse.ArgumentParser(
        description="Paired statistical testing for a single already-merged baseline vs synthetic brain-age CSV."
    )
    parser.add_argument("--paired-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10000)
    args = parser.parse_args()


    df = load_paired_csv(args.paired_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)


    lines = []
    lines.append("PAIRED STATISTICAL TESTS")
    lines.append("========================")
    lines.append(f"Subjects: {len(df)}")
    lines.append("")


    lines.extend(run_signed_tests(
        df["PBA_diff"].values,
        "Signed predicted brain age shift: PBA_synth - PBA_base",
        args.permutations
    ))


    lines.extend(run_signed_tests(
        df["BAD_diff"].values,
        "Signed BAG shift: BAG_synth - BAG_base",
        args.permutations
    ))


    lines.extend(run_abs_tests(
        df["PBA_abs_diff"].values,
        "Absolute predicted brain age difference: |PBA_synth - PBA_base|",
        args.permutations
    ))


    lines.extend(run_abs_tests(
        df["BAD_abs_diff"].values,
        "Absolute BAG difference: |BAG_synth - BAG_base|",
        args.permutations
    ))


    lines.extend(run_paired_abs_error_test(
        df["BASE_abs_err"].values,
        df["SYNTH_abs_err"].values
    ))


    out_txt = args.output_dir / "paired_stats_report.txt"
    out_csv = args.output_dir / "paired_stats_table.csv"


    out_txt.write_text("\n".join(lines), encoding="utf-8")


    summary_table = pd.DataFrame({
        "metric": [
            "mean_PBA_diff",
            "mean_BAD_diff",
            "mean_abs_PBA_diff",
            "mean_abs_BAD_diff",
            "mean_abs_BAG_baseline",
            "mean_abs_BAG_synthetic",
            "mean_abs_error_change",
        ],
        "value": [
            float(df["PBA_diff"].mean()),
            float(df["BAD_diff"].mean()),
            float(df["PBA_abs_diff"].mean()),
            float(df["BAD_abs_diff"].mean()),
            float(df["BASE_abs_err"].mean()),
            float(df["SYNTH_abs_err"].mean()),
            float(df["ABS_ERR_diff"].mean()),
        ],
    })
    summary_table.to_csv(out_csv, index=False)


    print(f"Saved statistical report to: {out_txt}")
    print(f"Saved summary table to: {out_csv}")




if __name__ == "__main__":
    main()

