#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Statistical testing for baseline vs synthetic brain-age predictions,
including age-stratified analysis in 5-year bins and age-trend tests.


Expected CSV format for BOTH --baseline-csv and --synthetic-csv:
IXI_ID
Age
Filename
Path
Predicted_Brain_Age
Brain_Age_Difference


Example
-------
py -3.10 exp_0\processing\postprocessing\overall\significance2.py `
  --baseline-csv "data\predictions\BrainAgeNeXt\ixi_only_faithfullpreprocess.csv" `
  --synthetic-csv "data\predictions\BrainAgeNeXt\IXI_BraTS_Guiz_CM.csv" `
  --output-dir "exp_0\results\IXI_BraTS_CM_extended" `
  --permutations 10000 `
  --age-bin-width 5 `
  --min-bin-size 10

py -3.10 exp_0\processing\postprocessing\overall\significance2.py `
  --baseline-csv "data\predictions\BrainAgeNeXt\ixi_only_faithfullpreprocess.csv" `
  --synthetic-csv "data\predictions\BrainAgeNeXt\GliGAN_FAITHFUL\brainagenext_predictions_run_001.csv" `
  --output-dir "exp_0\results\true_gligan" `
  --permutations 10000 `
  --age-bin-width 5 `
  --min-bin-size 10
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


    merged["Age"] = merged["Age_base"]


    merged["PBA_diff"] = merged["PBA_synth"] - merged["PBA_base"]
    merged["BAD_diff"] = merged["BAD_synth"] - merged["BAD_base"]


    merged["PBA_abs_diff"] = np.abs(merged["PBA_diff"])
    merged["BAD_abs_diff"] = np.abs(merged["BAD_diff"])


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
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    observed = float(np.mean(diff))


    if len(diff) == 0:
        return observed, 1.0


    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    perm_means = np.mean(signs * diff[None, :], axis=1)
    p = (np.sum(np.abs(perm_means) >= abs(observed)) + 1) / (n_perm + 1)
    return observed, float(p)




def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """
    BH-FDR adjusted p-values in original order.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return []


    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)


    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adjusted[i] = prev


    out = np.empty(n, dtype=float)
    out[order] = np.minimum(adjusted, 1.0)
    return out.tolist()




# =============================================================================
# Core test blocks
# =============================================================================


def run_signed_shift_tests(diff: np.ndarray, label: str, permutations: int) -> list[str]:
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
# Age binning and age-trend analyses
# =============================================================================


def make_age_bins(df: pd.DataFrame, age_bin_width: int) -> pd.DataFrame:
    out = df.copy()
    min_age = int(np.floor(out["Age"].min() / age_bin_width) * age_bin_width)
    max_age = int(np.ceil(out["Age"].max() / age_bin_width) * age_bin_width)


    edges = np.arange(min_age, max_age + age_bin_width, age_bin_width, dtype=int)
    if len(edges) < 2:
        edges = np.array([min_age, min_age + age_bin_width], dtype=int)


    labels = [f"{edges[i]}-{edges[i + 1] - 1}" for i in range(len(edges) - 1)]
    out["Age_bin"] = pd.cut(
        out["Age"],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    return out




def run_age_bin_signed_tests(
    df: pd.DataFrame,
    diff_col: str,
    bin_col: str,
    label: str,
    permutations: int,
    min_bin_size: int,
) -> tuple[list[str], pd.DataFrame]:
    lines = []
    rows = []


    lines.append(label)
    lines.append("-" * len(label))


    for bin_name, sub in df.groupby(bin_col, observed=False):
        if pd.isna(bin_name):
            continue


        diff = sub[diff_col].to_numpy(dtype=float)
        n = len(diff)
        mean = float(np.mean(diff))
        median = float(np.median(diff))
        ci_lo, ci_hi = mean_ci(diff)


        t_p = np.nan
        w_p = np.nan
        perm_p = np.nan
        spearman_age_p = np.nan


        if n >= 2:
            _, t_p = stats.ttest_1samp(diff, popmean=0.0)


        if n >= min_bin_size:
            try:
                _, w_p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                w_p = np.nan


            _, perm_p = sign_flip_permutation_test(diff, n_perm=permutations)


        rows.append(
            {
                "Age_bin": str(bin_name),
                "n": n,
                "mean": mean,
                "median": median,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "t_p": t_p,
                "wilcoxon_p": w_p,
                "permutation_p": perm_p,
                "cohens_d": cohens_d_paired(diff),
                "rank_biserial": rank_biserial_from_wilcoxon(diff) if n > 0 else np.nan,
                "positive_count": int((diff > 0).sum()),
                "negative_count": int((diff < 0).sum()),
                "zero_count": int((diff == 0).sum()),
            }
        )


    result_df = pd.DataFrame(rows)


    if not result_df.empty:
        for col in ["t_p", "wilcoxon_p", "permutation_p"]:
            valid = result_df[col].notna()
            adjusted = [np.nan] * len(result_df)
            if valid.any():
                adj_vals = benjamini_hochberg(result_df.loc[valid, col].tolist())
                valid_idx = np.where(valid.to_numpy())[0]
                for idx, val in zip(valid_idx, adj_vals):
                    adjusted[idx] = val
            result_df[f"{col}_fdr_bh"] = adjusted


    if result_df.empty:
        lines.append("  No valid age bins.")
        lines.append("")
        return lines, result_df


    for _, row in result_df.iterrows():
        lines.append(f"  Age bin                      = {row['Age_bin']}")
        lines.append(f"    n                          = {int(row['n'])}")
        lines.append(f"    mean                       = {row['mean']:.6f}")
        lines.append(f"    median                     = {row['median']:.6f}")
        lines.append(f"    95% CI of mean             = [{row['ci_lo']:.6f}, {row['ci_hi']:.6f}]")
        lines.append(f"    positive / negative / zero = {int(row['positive_count'])} / {int(row['negative_count'])} / {int(row['zero_count'])}")
        lines.append(f"    t-test p                   = {row['t_p']:.6g}" if pd.notna(row["t_p"]) else "    t-test p                   = NA")
        lines.append(f"    Wilcoxon p                 = {row['wilcoxon_p']:.6g}" if pd.notna(row["wilcoxon_p"]) else "    Wilcoxon p                 = NA")
        lines.append(f"    permutation p              = {row['permutation_p']:.6g}" if pd.notna(row["permutation_p"]) else "    permutation p              = NA")
        lines.append(f"    FDR-adjusted Wilcoxon p    = {row['wilcoxon_p_fdr_bh']:.6g}" if pd.notna(row.get("wilcoxon_p_fdr_bh", np.nan)) else "    FDR-adjusted Wilcoxon p    = NA")
        lines.append(f"    Cohen's d                  = {row['cohens_d']:.6f}")
        lines.append(f"    rank-biserial              = {row['rank_biserial']:.6f}" if pd.notna(row["rank_biserial"]) else "    rank-biserial              = NA")
        lines.append("")


    return lines, result_df




def linear_regression_test(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)


    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r": float(r_value),
        "r2": float(r_value ** 2),
        "p": float(p_value),
        "stderr": float(std_err),
    }




def quadratic_regression_test(x: np.ndarray, y: np.ndarray) -> dict:
    """
    OLS test for y ~ 1 + age + age^2 using normal equations.
    Returns global F-test p-value vs intercept-only.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)


    if n < 4:
        return {
            "beta0": np.nan,
            "beta1": np.nan,
            "beta2": np.nan,
            "r2": np.nan,
            "f_stat": np.nan,
            "p": np.nan,
        }


    X = np.column_stack([np.ones(n), x, x ** 2])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta


    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


    p_model = X.shape[1] - 1
    df_model = p_model
    df_resid = n - X.shape[1]


    if df_resid <= 0 or ss_tot <= 0:
        return {
            "beta0": float(beta[0]),
            "beta1": float(beta[1]),
            "beta2": float(beta[2]),
            "r2": r2,
            "f_stat": np.nan,
            "p": np.nan,
        }


    ss_model = ss_tot - ss_res
    ms_model = ss_model / df_model
    ms_resid = ss_res / df_resid
    f_stat = ms_model / ms_resid if ms_resid > 0 else np.nan
    p = 1.0 - stats.f.cdf(f_stat, df_model, df_resid) if np.isfinite(f_stat) else np.nan


    return {
        "beta0": float(beta[0]),
        "beta1": float(beta[1]),
        "beta2": float(beta[2]),
        "r2": float(r2),
        "f_stat": float(f_stat),
        "p": float(p),
    }




def run_age_trend_tests(df: pd.DataFrame, diff_col: str, label: str) -> list[str]:
    lines = []
    x = df["Age"].to_numpy(dtype=float)
    y = df[diff_col].to_numpy(dtype=float)


    lines.append(label)


    if len(df) >= 3:
        rho, spearman_p = stats.spearmanr(x, y)
        lines.append(f"  Spearman rho                = {rho:.6f}")
        lines.append(f"  Spearman p-value            = {spearman_p:.6g}")
    else:
        lines.append("  Spearman                    = not enough samples")


    if len(df) >= 3:
        lin = linear_regression_test(x, y)
        lines.append(f"  Linear slope                = {lin['slope']:.6f}")
        lines.append(f"  Linear intercept            = {lin['intercept']:.6f}")
        lines.append(f"  Linear r                    = {lin['r']:.6f}")
        lines.append(f"  Linear R^2                  = {lin['r2']:.6f}")
        lines.append(f"  Linear p-value              = {lin['p']:.6g}")
        lines.append(f"  Linear stderr               = {lin['stderr']:.6f}")
    else:
        lines.append("  Linear regression           = not enough samples")


    quad = quadratic_regression_test(x, y)
    lines.append(f"  Quadratic beta0             = {quad['beta0']:.6f}" if np.isfinite(quad["beta0"]) else "  Quadratic beta0             = NA")
    lines.append(f"  Quadratic beta1             = {quad['beta1']:.6f}" if np.isfinite(quad["beta1"]) else "  Quadratic beta1             = NA")
    lines.append(f"  Quadratic beta2             = {quad['beta2']:.6f}" if np.isfinite(quad["beta2"]) else "  Quadratic beta2             = NA")
    lines.append(f"  Quadratic R^2               = {quad['r2']:.6f}" if np.isfinite(quad["r2"]) else "  Quadratic R^2               = NA")
    lines.append(f"  Quadratic model F           = {quad['f_stat']:.6f}" if np.isfinite(quad["f_stat"]) else "  Quadratic model F           = NA")
    lines.append(f"  Quadratic model p-value     = {quad['p']:.6g}" if np.isfinite(quad["p"]) else "  Quadratic model p-value     = NA")
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
    parser.add_argument("--age-bin-width", type=int, default=5, help="Age bin width in years.")
    parser.add_argument("--min-bin-size", type=int, default=10, help="Minimum subjects for Wilcoxon/permutation in a bin.")
    args = parser.parse_args()


    df = merge_csvs(args.baseline_csv, args.synthetic_csv)
    df = make_age_bins(df, age_bin_width=args.age_bin_width)


    args.output_dir.mkdir(parents=True, exist_ok=True)


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
    lines.append("  - Age-bin tests evaluate whether signed effects differ across age ranges.")
    lines.append("  - Age-trend tests evaluate whether lesion-induced shift changes as a function of age.")
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


    # Age-trend analyses
    lines.append("AGE-TREND ANALYSIS")
    lines.append("==================")
    lines.append("")
    lines.extend(run_age_trend_tests(df, "PBA_diff", "Age trend tests for signed predicted brain age shift"))
    lines.extend(run_age_trend_tests(df, "BAD_diff", "Age trend tests for signed BAG shift"))


    # Age-bin analyses
    lines.append("AGE-BIN SIGNED TESTS")
    lines.append("====================")
    lines.append(f"Bin width: {args.age_bin_width} years")
    lines.append(f"Minimum bin size for Wilcoxon/permutation: {args.min_bin_size}")
    lines.append("")


    pba_bin_lines, pba_bin_df = run_age_bin_signed_tests(
        df=df,
        diff_col="PBA_diff",
        bin_col="Age_bin",
        label="5-year age-bin signed tests for PBA_diff",
        permutations=args.permutations,
        min_bin_size=args.min_bin_size,
    )
    lines.extend(pba_bin_lines)


    bad_bin_lines, bad_bin_df = run_age_bin_signed_tests(
        df=df,
        diff_col="BAD_diff",
        bin_col="Age_bin",
        label="5-year age-bin signed tests for BAD_diff",
        permutations=args.permutations,
        min_bin_size=args.min_bin_size,
    )
    lines.extend(bad_bin_lines)


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


    # Save age-bin tables
    if not pba_bin_df.empty:
        pba_bin_df.to_csv(args.output_dir / "age_bin_signed_tests_pba_diff.csv", index=False)
    if not bad_bin_df.empty:
        bad_bin_df.to_csv(args.output_dir / "age_bin_signed_tests_bad_diff.csv", index=False)


    # Save trend summary
    trend_rows = []


    x = df["Age"].to_numpy(dtype=float)
    for diff_col, label in [("PBA_diff", "PBA_diff"), ("BAD_diff", "BAD_diff")]:
        y = df[diff_col].to_numpy(dtype=float)


        rho, spearman_p = stats.spearmanr(x, y) if len(df) >= 3 else (np.nan, np.nan)
        lin = linear_regression_test(x, y) if len(df) >= 3 else {
            "slope": np.nan, "intercept": np.nan, "r": np.nan, "r2": np.nan, "p": np.nan, "stderr": np.nan
        }
        quad = quadratic_regression_test(x, y)


        trend_rows.append(
            {
                "metric": label,
                "spearman_rho": rho,
                "spearman_p": spearman_p,
                "linear_slope": lin["slope"],
                "linear_intercept": lin["intercept"],
                "linear_r": lin["r"],
                "linear_r2": lin["r2"],
                "linear_p": lin["p"],
                "quadratic_beta0": quad["beta0"],
                "quadratic_beta1": quad["beta1"],
                "quadratic_beta2": quad["beta2"],
                "quadratic_r2": quad["r2"],
                "quadratic_f": quad["f_stat"],
                "quadratic_p": quad["p"],
            }
        )


    pd.DataFrame(trend_rows).to_csv(args.output_dir / "age_trend_tests.csv", index=False)


    print(f"Saved paired comparison table to: {paired_csv}")
    print(f"Saved statistical report to: {out_txt}")
    print(f"Saved summary table to: {out_csv}")
    print(f"Saved age-bin PBA table to: {args.output_dir / 'age_bin_signed_tests_pba_diff.csv'}")
    print(f"Saved age-bin BAG table to: {args.output_dir / 'age_bin_signed_tests_bad_diff.csv'}")
    print(f"Saved age-trend summary to: {args.output_dir / 'age_trend_tests.csv'}")




if __name__ == "__main__":
    main()

