#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Statistical testing + corrected BAG plotting for tumor-agnostic brain-age analysis.


Primary analysis:
  - raw BAG / ΔBAG
  - age as the only covariate


Sensitivity / corrected analysis:
  - age-residualized BAG
  - recompute ΔBAG and recovery statistics
  - plot corrected BAG outputs


Important:
  - This script does NOT compute corrected predicted age.
  - This script does NOT claim corrected MAE/R² improves model performance.
  - Corrected/covariate BAG is used only for downstream lesion-bias analysis.
"""


from __future__ import annotations


import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats




PRED_CANDIDATES = [
    "Predicted_Brain_Age",
    "brainagenext_predictions_run_001_PBA",
]


BAG_CANDIDATES = [
    "Brain_Age_Difference",
    "brainagenext_predictions_run_001_BAD",
    "BAG",
    "BAD",
]


AGE_CANDIDATES = [
    "Age",
    "AGE",
    "Patient's Age",
    "age",
]


ID_CANDIDATES = [
    "IXI_ID",
    "Subject_ID",
    "subject_id",
    "ID",
]




def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None




def parse_named_path(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected LABEL=CSV, got: {raw}")
    label, path = raw.split("=", 1)
    return label.strip(), Path(path.strip())




def sanitize(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text)).strip("_")




def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)


    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]


    if len(x) < 2:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan


    return float(np.corrcoef(x, y)[0, 1])




def normalize_prediction_csv(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]


    id_col = first_existing(df, ID_CANDIDATES)
    age_col = first_existing(df, AGE_CANDIDATES)
    pred_col = first_existing(df, PRED_CANDIDATES)
    bag_col = first_existing(df, BAG_CANDIDATES)


    if id_col is None:
        raise ValueError(f"{label}: missing subject ID column. Found columns: {df.columns.tolist()}")
    if age_col is None:
        raise ValueError(f"{label}: missing age column. Found columns: {df.columns.tolist()}")
    if pred_col is None:
        raise ValueError(f"{label}: missing predicted brain age column. Found columns: {df.columns.tolist()}")


    out = pd.DataFrame()
    out["IXI_ID"] = df[id_col].astype(str).str.strip()
    out["Age"] = pd.to_numeric(df[age_col], errors="coerce")
    out["Predicted_Brain_Age"] = pd.to_numeric(df[pred_col], errors="coerce")


    if bag_col is not None:
        out["BAG"] = pd.to_numeric(df[bag_col], errors="coerce")
    else:
        out["BAG"] = out["Predicted_Brain_Age"] - out["Age"]


    out["Brain_Age_Difference"] = out["BAG"]
    out["Model"] = label
    out["Set_Label"] = label


    out = out.dropna(subset=["IXI_ID", "Age", "Predicted_Brain_Age", "BAG"])
    return out.reset_index(drop=True)




def compute_metrics(df: pd.DataFrame, model: str) -> Dict:
    age = df["Age"].to_numpy(float)
    pred = df["Predicted_Brain_Age"].to_numpy(float)
    bag = df["BAG"].to_numpy(float)


    return {
        "Model": model,
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
        "MAE": float(np.mean(np.abs(bag))),
        "RMSE": float(np.sqrt(np.mean(bag ** 2))),
        "Correlation_Age_vs_Pred": safe_corr(age, pred),
        "Correlation_Age_vs_BAG": safe_corr(age, bag),
    }




def fit_age_correction_from_baseline(baseline_df: pd.DataFrame) -> Tuple[float, float]:
    age = baseline_df["Age"].to_numpy(float)
    bag = baseline_df["BAG"].to_numpy(float)


    valid = np.isfinite(age) & np.isfinite(bag)
    age = age[valid]
    bag = bag[valid]


    X = np.column_stack([np.ones(len(age)), age])
    alpha, beta = np.linalg.lstsq(X, bag, rcond=None)[0]


    return float(alpha), float(beta)




def apply_age_residualized_bag(df: pd.DataFrame, alpha: float, beta: float) -> pd.DataFrame:
    out = df.copy()


    out["BAG_raw"] = out["BAG"]
    out["Brain_Age_Difference_raw"] = out["BAG"]
    out["BAG_age_fit"] = alpha + beta * out["Age"]


    out["BAG"] = out["BAG"] - out["BAG_age_fit"]
    out["Brain_Age_Difference"] = out["BAG"]


    out["Age_Correction_Alpha"] = alpha
    out["Age_Correction_Beta"] = beta


    return out




def mean_ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]


    n = len(values)
    mean = float(np.mean(values))


    if n < 2:
        return mean, mean


    se = stats.sem(values, nan_policy="omit")
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)


    return float(mean - tcrit * se), float(mean + tcrit * se)




def cohens_d_paired(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]


    if len(diff) < 2:
        return np.nan


    sd = diff.std(ddof=1)
    if sd < 1e-12:
        return np.nan


    return float(diff.mean() / sd)




def rank_biserial_from_wilcoxon(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]


    n = len(diff)
    if n == 0:
        return np.nan


    ranks = stats.rankdata(np.abs(diff))
    w_pos = ranks[diff > 0].sum()
    w_neg = ranks[diff < 0].sum()


    return float((w_pos - w_neg) / (n * (n + 1) / 2.0))




def sign_flip_permutation_test(
    diff: np.ndarray,
    permutations: int,
    alternative: str = "two-sided",
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)


    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]


    if len(diff) == 0:
        return np.nan


    observed = float(np.mean(diff))
    signs = rng.choice([-1.0, 1.0], size=(permutations, len(diff)))
    perm_means = np.mean(signs * diff[None, :], axis=1)


    if alternative == "greater":
        p = (np.sum(perm_means >= observed) + 1) / (permutations + 1)
    elif alternative == "less":
        p = (np.sum(perm_means <= observed) + 1) / (permutations + 1)
    else:
        p = (np.sum(np.abs(perm_means) >= abs(observed)) + 1) / (permutations + 1)


    return float(p)




def one_sample_stat_tests(
    values: np.ndarray,
    test_name: str,
    permutations: int,
    alternative: str = "two-sided",
) -> Dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]


    ci_low, ci_high = mean_ci(values)


    row = {
        "test_name": test_name,
        "n": int(len(values)),
        "mean": float(np.mean(values)) if len(values) else np.nan,
        "median": float(np.median(values)) if len(values) else np.nan,
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "alternative": alternative,
        "cohens_d_paired": cohens_d_paired(values),
        "rank_biserial": rank_biserial_from_wilcoxon(values),
    }


    if len(values) >= 2:
        t_res = stats.ttest_1samp(values, popmean=0.0, alternative=alternative)
        row["t_stat"] = float(t_res.statistic)
        row["t_p"] = float(t_res.pvalue)


        try:
            w_res = stats.wilcoxon(values, zero_method="wilcox", alternative=alternative)
            row["wilcoxon_stat"] = float(w_res.statistic)
            row["wilcoxon_p"] = float(w_res.pvalue)
        except ValueError:
            row["wilcoxon_stat"] = np.nan
            row["wilcoxon_p"] = np.nan


        row["permutation_p"] = sign_flip_permutation_test(
            values,
            permutations=permutations,
            alternative=alternative,
        )
    else:
        row["t_stat"] = np.nan
        row["t_p"] = np.nan
        row["wilcoxon_stat"] = np.nan
        row["wilcoxon_p"] = np.nan
        row["permutation_p"] = np.nan


    return row




def age_covariate_intercept_test(values: np.ndarray, age: np.ndarray) -> Dict:
    y = np.asarray(values, dtype=float)
    age = np.asarray(age, dtype=float)


    valid = np.isfinite(y) & np.isfinite(age)
    y = y[valid]
    age = age[valid]


    if len(y) < 3:
        return {
            "age_cov_n": int(len(y)),
            "age_adjusted_intercept": np.nan,
            "age_adjusted_intercept_t": np.nan,
            "age_adjusted_intercept_p": np.nan,
            "age_beta": np.nan,
            "age_beta_t": np.nan,
            "age_beta_p": np.nan,
        }


    age_centered = age - np.mean(age)
    X = np.column_stack([np.ones(len(y)), age_centered])


    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat


    n = len(y)
    p = X.shape[1]
    dof = n - p


    if dof <= 0:
        return {
            "age_cov_n": int(n),
            "age_adjusted_intercept": np.nan,
            "age_adjusted_intercept_t": np.nan,
            "age_adjusted_intercept_p": np.nan,
            "age_beta": np.nan,
            "age_beta_t": np.nan,
            "age_beta_p": np.nan,
        }


    sigma2 = float((resid @ resid) / dof)
    cov_beta = sigma2 * np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(np.diag(cov_beta))


    intercept_t = beta[0] / se_beta[0]
    age_t = beta[1] / se_beta[1]


    intercept_p = 2.0 * (1.0 - stats.t.cdf(abs(intercept_t), df=dof))
    age_p = 2.0 * (1.0 - stats.t.cdf(abs(age_t), df=dof))


    return {
        "age_cov_n": int(n),
        "age_adjusted_intercept": float(beta[0]),
        "age_adjusted_intercept_t": float(intercept_t),
        "age_adjusted_intercept_p": float(intercept_p),
        "age_beta": float(beta[1]),
        "age_beta_t": float(age_t),
        "age_beta_p": float(age_p),
    }




def build_paired_df(base_df: pd.DataFrame, synth_df: pd.DataFrame) -> pd.DataFrame:
    base = base_df.rename(
        columns={
            "Age": "Age_base",
            "Predicted_Brain_Age": "PBA_base",
            "BAG": "BAD_base",
        }
    )


    synth = synth_df.rename(
        columns={
            "Age": "Age_synth",
            "Predicted_Brain_Age": "PBA_synth",
            "BAG": "BAD_synth",
        }
    )


    merged = pd.merge(
        base[["IXI_ID", "Age_base", "PBA_base", "BAD_base"]],
        synth[["IXI_ID", "Age_synth", "PBA_synth", "BAD_synth"]],
        on="IXI_ID",
        how="inner",
    )


    if merged.empty:
        raise RuntimeError("No overlapping IXI_IDs between baseline and comparison set.")


    merged["Age"] = merged["Age_base"]
    merged["PBA_diff"] = merged["PBA_synth"] - merged["PBA_base"]
    merged["PBA_abs_diff"] = np.abs(merged["PBA_diff"])
    merged["BAD_diff"] = merged["BAD_synth"] - merged["BAD_base"]
    merged["BAD_abs_diff"] = np.abs(merged["BAD_diff"])
    merged["ABS_BAD_base"] = np.abs(merged["BAD_base"])
    merged["ABS_BAD_synth"] = np.abs(merged["BAD_synth"])
    merged["BASE_abs_err"] = np.abs(merged["BAD_base"])
    merged["SYNTH_abs_err"] = np.abs(merged["BAD_synth"])
    merged["ABS_ERR_diff"] = merged["SYNTH_abs_err"] - merged["BASE_abs_err"]


    return merged.reset_index(drop=True)




def build_triplet_df(
    baseline_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    inpaint_df: pd.DataFrame,
) -> pd.DataFrame:
    base = baseline_df.rename(
        columns={
            "Age": "Age_base",
            "Predicted_Brain_Age": "PBA_base",
            "BAG": "BAD_base",
        }
    )


    synth = synthetic_df.rename(
        columns={
            "Age": "Age_synth",
            "Predicted_Brain_Age": "PBA_synth",
            "BAG": "BAD_synth",
        }
    )


    inp = inpaint_df.rename(
        columns={
            "Age": "Age_inpaint",
            "Predicted_Brain_Age": "PBA_inpaint",
            "BAG": "BAD_inpaint",
        }
    )


    merged = pd.merge(
        base[["IXI_ID", "Age_base", "PBA_base", "BAD_base"]],
        synth[["IXI_ID", "Age_synth", "PBA_synth", "BAD_synth"]],
        on="IXI_ID",
        how="inner",
    )


    merged = pd.merge(
        merged,
        inp[["IXI_ID", "Age_inpaint", "PBA_inpaint", "BAD_inpaint"]],
        on="IXI_ID",
        how="inner",
    )


    if merged.empty:
        raise RuntimeError("No overlapping IXI_IDs between baseline, synthetic, and inpainted sets.")


    merged["Age"] = merged["Age_base"]


    merged["Delta_BAG_tumor"] = merged["BAD_synth"] - merged["BAD_base"]
    merged["Delta_BAG_inpaint"] = merged["BAD_inpaint"] - merged["BAD_base"]


    merged["Abs_Delta_BAG_tumor"] = np.abs(merged["Delta_BAG_tumor"])
    merged["Abs_Delta_BAG_inpaint"] = np.abs(merged["Delta_BAG_inpaint"])


    merged["Recovery"] = merged["Abs_Delta_BAG_tumor"] - merged["Abs_Delta_BAG_inpaint"]


    merged["synth_distance_from_base"] = merged["Abs_Delta_BAG_tumor"]
    merged["inpaint_distance_from_base"] = merged["Abs_Delta_BAG_inpaint"]


    merged["inpaint_closer_to_base"] = (
        merged["inpaint_distance_from_base"] < merged["synth_distance_from_base"]
    )


    merged["Inpaint_Closer_To_Healthy"] = merged["inpaint_closer_to_base"]


    merged["Percent_Recovery"] = np.where(
        merged["Abs_Delta_BAG_tumor"] > 1e-12,
        100.0 * merged["Recovery"] / merged["Abs_Delta_BAG_tumor"],
        np.nan,
    )


    return merged.reset_index(drop=True)




def bag_paired_shift_plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    d = df.sort_values(["Age", "IXI_ID"]).reset_index(drop=True)
    x = np.arange(len(d))


    plt.figure(figsize=(12, 6))


    for i, row in d.iterrows():
        plt.plot(
            [i, i],
            [row["BAD_base"], row["BAD_synth"]],
            color="gray",
            alpha=0.25,
            linewidth=1,
        )


    plt.scatter(x, d["BAD_base"], color="seagreen", label="Healthy baseline BAG", s=18)
    plt.scatter(x, d["BAD_synth"], color="darkorange", label="Comparison BAG", s=18)


    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Subjects ranked by age")
    plt.ylabel("Age-residualized BAG")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def bag_paired_shift_plot_synthetic_higher_highlight(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    d = df.sort_values(["Age", "IXI_ID"]).reset_index(drop=True)
    x = np.arange(len(d))


    synth_higher = d["BAD_synth"].to_numpy(float) > d["BAD_base"].to_numpy(float)


    plt.figure(figsize=(12, 6))


    for i, row in d.iterrows():
        hi = bool(synth_higher[i])
        plt.plot(
            [i, i],
            [row["BAD_base"], row["BAD_synth"]],
            color="#e85d4c" if hi else "gray",
            alpha=0.55 if hi else 0.22,
            linewidth=1.4 if hi else 1.0,
        )


    plt.scatter(
        x,
        d["BAD_base"],
        color="seagreen",
        label="Healthy baseline BAG",
        s=18,
        zorder=2,
    )


    mask_not_higher = ~synth_higher
    if mask_not_higher.any():
        plt.scatter(
            x[mask_not_higher],
            d.loc[mask_not_higher, "BAD_synth"],
            color="darkorange",
            label="Synthetic BAG",
            s=16,
            alpha=0.55,
            zorder=3,
        )


    if synth_higher.any():
        plt.scatter(
            x[synth_higher],
            d.loc[synth_higher, "BAD_synth"],
            color="darkorange",
            label="Synthetic BAG higher than baseline",
            s=26,
            alpha=0.95,
            edgecolors="#8b0000",
            linewidths=0.7,
            zorder=4,
        )


    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Subjects ranked by age")
    plt.ylabel("Age-residualized BAG")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def bag_paired_shift_plot_baseline_higher_highlight(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    d = df.sort_values(["Age", "IXI_ID"]).reset_index(drop=True)
    x = np.arange(len(d))


    baseline_higher = d["BAD_base"].to_numpy(float) > d["BAD_synth"].to_numpy(float)


    plt.figure(figsize=(12, 6))


    for i, row in d.iterrows():
        hi = bool(baseline_higher[i])
        plt.plot(
            [i, i],
            [row["BAD_base"], row["BAD_synth"]],
            color="#2b6cb0" if hi else "gray",
            alpha=0.55 if hi else 0.22,
            linewidth=1.4 if hi else 1.0,
        )


    mask_not_higher = ~baseline_higher


    if mask_not_higher.any():
        plt.scatter(
            x[mask_not_higher],
            d.loc[mask_not_higher, "BAD_base"],
            color="seagreen",
            label="Healthy baseline BAG",
            s=16,
            alpha=0.55,
            zorder=2,
        )


    if baseline_higher.any():
        plt.scatter(
            x[baseline_higher],
            d.loc[baseline_higher, "BAD_base"],
            color="seagreen",
            label="Baseline BAG higher than synthetic",
            s=26,
            alpha=0.95,
            edgecolors="#0b2f4a",
            linewidths=0.7,
            zorder=4,
        )


    plt.scatter(
        x,
        d["BAD_synth"],
        color="darkorange",
        label="Synthetic BAG",
        s=18,
        zorder=3,
    )


    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Subjects ranked by age")
    plt.ylabel("Age-residualized BAG")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def bag_triplet_shift_plot_closer_to_baseline(df: pd.DataFrame, out_path: Path, title: str) -> None:
    d = df.sort_values(["Age", "IXI_ID"]).reset_index(drop=True)
    x = np.arange(len(d))


    improved = d["inpaint_closer_to_base"].to_numpy(bool)


    good_color = "#2ca25f"
    bad_color = "#de2d26"
    neutral_color = "gray"


    plt.figure(figsize=(13, 6))


    for i, row in d.iterrows():
        color = good_color if improved[i] else bad_color


        plt.plot(
            [i, i],
            [row["BAD_base"], row["BAD_synth"]],
            color=neutral_color,
            alpha=0.22,
            linewidth=1.0,
            zorder=1,
        )


        plt.plot(
            [i, i],
            [row["BAD_synth"], row["BAD_inpaint"]],
            color=color,
            alpha=0.65,
            linewidth=1.4,
            zorder=2,
        )


    plt.scatter(
        x,
        d["BAD_base"],
        marker="o",
        color="black",
        label="Healthy baseline BAG",
        s=20,
        zorder=4,
    )


    plt.scatter(
        x,
        d["BAD_synth"],
        marker="^",
        color="darkorange",
        label="Synthetic tumor BAG",
        s=24,
        alpha=0.85,
        zorder=5,
    )


    plt.scatter(
        x[improved],
        d.loc[improved, "BAD_inpaint"],
        marker="s",
        color=good_color,
        label="Inpainted BAG closer to healthy",
        s=24,
        alpha=0.95,
        zorder=6,
    )


    plt.scatter(
        x[~improved],
        d.loc[~improved, "BAD_inpaint"],
        marker="s",
        color=bad_color,
        label="Inpainted BAG not closer",
        s=24,
        alpha=0.95,
        zorder=6,
    )


    plt.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    plt.xlabel("Subjects ranked by age")
    plt.ylabel("Age-residualized BAG")
    plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def save_global_bag_hist(all_df: pd.DataFrame, out_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))


    for model, g in all_df.groupby("Model"):
        plt.hist(g["BAG"].to_numpy(float), bins=30, alpha=0.45, label=model)


    plt.axvline(0, linestyle="--", color="black")
    plt.xlabel("Age-residualized BAG")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def save_heatmap(summary_df: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    pivot = summary_df.set_index("Model")[[metric]]
    arr = pivot.to_numpy(float)


    plt.figure(figsize=(6, max(4, 0.45 * len(pivot) + 2)))
    im = plt.imshow(arr, aspect="auto")
    plt.colorbar(im, label=metric)


    plt.xticks([0], [metric])
    plt.yticks(np.arange(len(pivot)), pivot.index)


    for i in range(arr.shape[0]):
        plt.text(0, i, f"{arr[i, 0]:.2f}", ha="center", va="center")


    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def run_triplet_statistics(
    triplet: pd.DataFrame,
    analysis_name: str,
    inpaint_label: str,
    permutations: int,
) -> pd.DataFrame:
    age = triplet["Age"].to_numpy(float)


    tests = [
        {
            "test_name": "tumor_induced_signed_delta_BAG",
            "values": triplet["Delta_BAG_tumor"].to_numpy(float),
            "alternative": "two-sided",
            "meaning": "Tests whether tumor insertion changes BAG relative to healthy baseline.",
        },
        {
            "test_name": "inpaint_signed_delta_BAG",
            "values": triplet["Delta_BAG_inpaint"].to_numpy(float),
            "alternative": "two-sided",
            "meaning": "Tests whether inpainted image still differs from healthy baseline.",
        },
        {
            "test_name": "tumor_abs_delta_BAG",
            "values": triplet["Abs_Delta_BAG_tumor"].to_numpy(float),
            "alternative": "greater",
            "meaning": "Tests whether tumor causes nonzero absolute BAG perturbation.",
        },
        {
            "test_name": "inpaint_abs_delta_BAG",
            "values": triplet["Abs_Delta_BAG_inpaint"].to_numpy(float),
            "alternative": "greater",
            "meaning": "Tests whether inpainted image has nonzero absolute BAG perturbation.",
        },
        {
            "test_name": "recovery_abs_tumor_minus_abs_inpaint",
            "values": triplet["Recovery"].to_numpy(float),
            "alternative": "greater",
            "meaning": "Primary recovery test. Positive means inpainting is closer to healthy than tumor.",
        },
    ]


    rows = []


    for item in tests:
        row = one_sample_stat_tests(
            values=item["values"],
            test_name=item["test_name"],
            permutations=permutations,
            alternative=item["alternative"],
        )


        row.update(age_covariate_intercept_test(item["values"], age))
        row["analysis"] = analysis_name
        row["inpaint_label"] = inpaint_label
        row["meaning"] = item["meaning"]


        rows.append(row)


    return pd.DataFrame(rows)




def run_generator_statistics(
    paired: pd.DataFrame,
    analysis_name: str,
    generator_label: str,
    permutations: int,
) -> pd.DataFrame:
    age = paired["Age"].to_numpy(float)


    tests = [
        {
            "test_name": "generator_signed_delta_BAG",
            "values": paired["BAD_diff"].to_numpy(float),
            "alternative": "two-sided",
            "meaning": "Tests whether synthetic tumor generator shifts BAG relative to healthy baseline.",
        },
        {
            "test_name": "generator_abs_delta_BAG",
            "values": paired["BAD_abs_diff"].to_numpy(float),
            "alternative": "greater",
            "meaning": "Tests whether synthetic tumor generator causes nonzero absolute BAG perturbation.",
        },
    ]


    rows = []


    for item in tests:
        row = one_sample_stat_tests(
            values=item["values"],
            test_name=item["test_name"],
            permutations=permutations,
            alternative=item["alternative"],
        )


        row.update(age_covariate_intercept_test(item["values"], age))
        row["analysis"] = analysis_name
        row["generator_label"] = generator_label
        row["meaning"] = item["meaning"]


        rows.append(row)


    return pd.DataFrame(rows)




def summarize_triplet_descriptives(triplet: pd.DataFrame, analysis_name: str, inpaint_label: str) -> Dict:
    return {
        "analysis": analysis_name,
        "inpaint_label": inpaint_label,
        "n": int(len(triplet)),
        "mean_BAG_healthy": float(triplet["BAD_base"].mean()),
        "mean_BAG_tumor": float(triplet["BAD_synth"].mean()),
        "mean_BAG_inpaint": float(triplet["BAD_inpaint"].mean()),
        "mean_Delta_BAG_tumor": float(triplet["Delta_BAG_tumor"].mean()),
        "mean_Delta_BAG_inpaint": float(triplet["Delta_BAG_inpaint"].mean()),
        "mean_abs_Delta_BAG_tumor": float(triplet["Abs_Delta_BAG_tumor"].mean()),
        "mean_abs_Delta_BAG_inpaint": float(triplet["Abs_Delta_BAG_inpaint"].mean()),
        "mean_recovery": float(triplet["Recovery"].mean()),
        "median_recovery": float(triplet["Recovery"].median()),
        "mean_percent_recovery": float(np.nanmean(triplet["Percent_Recovery"])),
        "proportion_inpaint_closer": float(triplet["Inpaint_Closer_To_Healthy"].mean()),
        "count_inpaint_closer": int(triplet["Inpaint_Closer_To_Healthy"].sum()),
    }




def summarize_generator_descriptives(paired: pd.DataFrame, analysis_name: str, generator_label: str) -> Dict:
    return {
        "analysis": analysis_name,
        "generator_label": generator_label,
        "n": int(len(paired)),
        "mean_BAG_healthy": float(paired["BAD_base"].mean()),
        "mean_BAG_synthetic": float(paired["BAD_synth"].mean()),
        "mean_delta_BAG": float(paired["BAD_diff"].mean()),
        "median_delta_BAG": float(paired["BAD_diff"].median()),
        "mean_abs_delta_BAG": float(paired["BAD_abs_diff"].mean()),
        "median_abs_delta_BAG": float(paired["BAD_abs_diff"].median()),
        "count_synth_bag_higher": int((paired["BAD_synth"] > paired["BAD_base"]).sum()),
        "count_baseline_bag_higher": int((paired["BAD_base"] > paired["BAD_synth"]).sum()),
        "count_equal_bag": int((paired["BAD_base"] == paired["BAD_synth"]).sum()),
    }




def write_txt_summary(
    stats_df: pd.DataFrame,
    desc_df: pd.DataFrame,
    out_path: Path,
) -> None:
    lines = []


    lines.append("SUMMARY OF ALL STATISTICAL TESTS")
    lines.append("================================")
    lines.append("")
    lines.append("Interpretation note:")
    lines.append("  Primary analysis uses raw BAG / ΔBAG with age as the only covariate.")
    lines.append("  Corrected analysis uses age-residualized BAG as sensitivity analysis.")
    lines.append("  No corrected predicted ages are computed.")
    lines.append("  No post-hoc corrected MAE/R² model-improvement claims should be made.")
    lines.append("")


    lines.append("DESCRIPTIVE SUMMARIES")
    lines.append("=====================")
    lines.append(desc_df.to_string(index=False))
    lines.append("")


    lines.append("STATISTICAL TESTS")
    lines.append("=================")


    for _, row in stats_df.iterrows():
        label = row.get("inpaint_label", np.nan)
        if pd.isna(label):
            label = row.get("generator_label", "NA")


        lines.append("")
        lines.append(f"Analysis: {row.get('analysis', 'NA')}")
        lines.append(f"Label: {label}")
        lines.append(f"Test: {row.get('test_name', 'NA')}")
        lines.append(f"Meaning: {row.get('meaning', 'NA')}")
        lines.append(f"n = {row.get('n', np.nan)}")
        lines.append(f"mean = {row.get('mean', np.nan):.6f}")
        lines.append(f"median = {row.get('median', np.nan):.6f}")
        lines.append(f"std = {row.get('std', np.nan):.6f}")
        lines.append(f"95% CI = [{row.get('ci95_low', np.nan):.6f}, {row.get('ci95_high', np.nan):.6f}]")
        lines.append(f"alternative = {row.get('alternative', 'NA')}")
        lines.append(f"t = {row.get('t_stat', np.nan):.6f}, p = {row.get('t_p', np.nan):.6g}")
        lines.append(f"Wilcoxon W = {row.get('wilcoxon_stat', np.nan):.6f}, p = {row.get('wilcoxon_p', np.nan):.6g}")
        lines.append(f"Permutation p = {row.get('permutation_p', np.nan):.6g}")
        lines.append(f"Cohen's d paired = {row.get('cohens_d_paired', np.nan):.6f}")
        lines.append(f"Rank-biserial = {row.get('rank_biserial', np.nan):.6f}")
        lines.append(
            "Age-covariate intercept test: "
            f"intercept = {row.get('age_adjusted_intercept', np.nan):.6f}, "
            f"t = {row.get('age_adjusted_intercept_t', np.nan):.6f}, "
            f"p = {row.get('age_adjusted_intercept_p', np.nan):.6g}"
        )
        lines.append(
            "Age slope: "
            f"beta = {row.get('age_beta', np.nan):.6f}, "
            f"t = {row.get('age_beta_t', np.nan):.6f}, "
            f"p = {row.get('age_beta_p', np.nan):.6g}"
        )


    out_path.write_text("\n".join(lines), encoding="utf-8")




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()


    parser.add_argument("--baseline", required=True, help="Format: LABEL=CSV")


    parser.add_argument(
        "--synthetic",
        default=None,
        help="Synthetic tumor reference for inpainting analysis. Format: LABEL=CSV",
    )


    parser.add_argument(
        "--inpaint",
        action="append",
        default=None,
        help="Inpainted set. Format: LABEL=CSV. Can be repeated.",
    )


    parser.add_argument(
        "--synthetic-generator",
        action="append",
        default=None,
        help="Synthetic tumor generator set. Format: LABEL=CSV. Can be repeated.",
    )


    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10000)


    return parser.parse_args()




def main() -> None:
    args = parse_args()


    args.output_dir.mkdir(parents=True, exist_ok=True)


    baseline_label, baseline_path = parse_named_path(args.baseline)


    print(f"Loading baseline: {baseline_label} -> {baseline_path}")
    baseline_df = normalize_prediction_csv(baseline_path, baseline_label)


    alpha, beta = fit_age_correction_from_baseline(baseline_df)
    baseline_corr = apply_age_residualized_bag(baseline_df, alpha, beta)


    all_stats = []
    all_desc = []


    corrected_global_dfs = [baseline_corr]
    corrected_global_metrics = [compute_metrics(baseline_corr, baseline_label)]


    if args.synthetic_generator:
        for raw_spec in args.synthetic_generator:
            gen_label, gen_path = parse_named_path(raw_spec)


            print(f"Processing synthetic generator: {gen_label} -> {gen_path}")


            gen_df = normalize_prediction_csv(gen_path, gen_label)
            gen_corr = apply_age_residualized_bag(gen_df, alpha, beta)


            gen_out = args.output_dir / sanitize(gen_label)
            gen_out.mkdir(parents=True, exist_ok=True)


            paired_corr = build_paired_df(baseline_corr, gen_corr)
            paired_corr.to_csv(gen_out / "paired_age_residualized_BAG.csv", index=False)


            bag_paired_shift_plot(
                paired_corr,
                gen_out / "corrected_bag_per_subject_shift.png",
                title=f"{gen_label}: corrected BAG per-subject shift",
            )


            bag_paired_shift_plot_synthetic_higher_highlight(
                paired_corr,
                gen_out / "corrected_bag_per_subject_shift_synthetic_higher_highlight.png",
                title=f"{gen_label}: corrected BAG shift, synthetic higher highlighted",
            )


            bag_paired_shift_plot_baseline_higher_highlight(
                paired_corr,
                gen_out / "corrected_bag_per_subject_shift_baseline_higher_highlight.png",
                title=f"{gen_label}: corrected BAG shift, baseline higher highlighted",
            )


            gen_stats = run_generator_statistics(
                paired=paired_corr,
                analysis_name="synthetic_generator_age_residualized_BAG",
                generator_label=gen_label,
                permutations=args.permutations,
            )


            gen_desc = pd.DataFrame(
                [
                    summarize_generator_descriptives(
                        paired_corr,
                        analysis_name="synthetic_generator_age_residualized_BAG",
                        generator_label=gen_label,
                    )
                ]
            )


            gen_stats["age_correction_alpha"] = alpha
            gen_stats["age_correction_beta"] = beta
            gen_desc["age_correction_alpha"] = alpha
            gen_desc["age_correction_beta"] = beta


            gen_stats.to_csv(gen_out / "statistical_tests.csv", index=False)
            gen_desc.to_csv(gen_out / "descriptive_summary.csv", index=False)


            all_stats.append(gen_stats)
            all_desc.append(gen_desc)


            corrected_global_dfs.append(gen_corr)
            corrected_global_metrics.append(compute_metrics(gen_corr, gen_label))


    if args.synthetic is not None and args.inpaint:
        synthetic_label, synthetic_path = parse_named_path(args.synthetic)


        print(f"Loading synthetic reference: {synthetic_label} -> {synthetic_path}")


        synthetic_df = normalize_prediction_csv(synthetic_path, synthetic_label)
        synthetic_corr = apply_age_residualized_bag(synthetic_df, alpha, beta)


        corrected_global_dfs.append(synthetic_corr)
        corrected_global_metrics.append(compute_metrics(synthetic_corr, synthetic_label))


        for raw_spec in args.inpaint:
            inp_label, inp_path = parse_named_path(raw_spec)


            print(f"Processing inpainter: {inp_label} -> {inp_path}")


            inp_df = normalize_prediction_csv(inp_path, inp_label)
            inp_corr = apply_age_residualized_bag(inp_df, alpha, beta)


            inp_out = args.output_dir / sanitize(inp_label)
            inp_out.mkdir(parents=True, exist_ok=True)


            triplet_corr = build_triplet_df(
                baseline_df=baseline_corr,
                synthetic_df=synthetic_corr,
                inpaint_df=inp_corr,
            )


            triplet_corr.to_csv(inp_out / "triplet_age_residualized_BAG.csv", index=False)


            bag_triplet_shift_plot_closer_to_baseline(
                triplet_corr,
                inp_out / f"corrected_triplet_BAG_shift_vs_{sanitize(synthetic_label)}.png",
                title=f"{inp_label}: corrected BAG triplet shift",
            )


            inp_stats = run_triplet_statistics(
                triplet=triplet_corr,
                analysis_name="inpainting_age_residualized_BAG",
                inpaint_label=inp_label,
                permutations=args.permutations,
            )


            inp_desc = pd.DataFrame(
                [
                    summarize_triplet_descriptives(
                        triplet_corr,
                        analysis_name="inpainting_age_residualized_BAG",
                        inpaint_label=inp_label,
                    )
                ]
            )


            inp_stats["age_correction_alpha"] = alpha
            inp_stats["age_correction_beta"] = beta
            inp_desc["age_correction_alpha"] = alpha
            inp_desc["age_correction_beta"] = beta


            inp_stats.to_csv(inp_out / "statistical_tests.csv", index=False)
            inp_desc.to_csv(inp_out / "descriptive_summary.csv", index=False)


            all_stats.append(inp_stats)
            all_desc.append(inp_desc)


            corrected_global_dfs.append(inp_corr)
            corrected_global_metrics.append(compute_metrics(inp_corr, inp_label))


    if not all_stats:
        raise RuntimeError(
            "No analysis was run. Provide either --synthetic + --inpaint, or --synthetic-generator."
        )


    final_stats = pd.concat(all_stats, ignore_index=True)
    final_desc = pd.concat(all_desc, ignore_index=True)


    final_stats.to_csv(args.output_dir / "all_statistical_tests.csv", index=False)
    final_desc.to_csv(args.output_dir / "all_descriptive_summaries.csv", index=False)


    corrected_all_df = pd.concat(corrected_global_dfs, ignore_index=True)
    corrected_summary_df = pd.DataFrame(corrected_global_metrics)


    corrected_all_df.to_csv(args.output_dir / "corrected_all_sets_combined_predictions.csv", index=False)
    corrected_summary_df.to_csv(args.output_dir / "corrected_global_model_summary.csv", index=False)


    save_global_bag_hist(
        corrected_all_df,
        args.output_dir / "corrected_global_bag_hist.png",
        title="Corrected global BAG distribution",
    )


    save_heatmap(
        corrected_summary_df,
        metric="MAE",
        out_path=args.output_dir / "corrected_global_heatmap_MAE.png",
        title="Corrected global heatmap: MAE / mean absolute residualized BAG",
    )


    write_txt_summary(
        stats_df=final_stats,
        desc_df=final_desc,
        out_path=args.output_dir / "all_statistical_tests_summary.txt",
    )


    readme = f"""
STATISTICAL STRATEGY
====================


Age correction fit:
  BAG = alpha + beta * Age


Fitted on healthy baseline only.


alpha = {alpha:.8f}
beta  = {beta:.8f}


Corrected BAG:
  BAG_age_residualized = BAG - (alpha + beta * Age)


Primary plotted corrected quantity:
  age-residualized BAG


For inpainters:
  - corrected triplet BAG plot:
      healthy -> synthetic tumor -> inpainted


For synthetic tumor generators:
  - corrected paired BAG shift plot:
      healthy -> synthetic tumor generator
  - corrected paired BAG shift plot with synthetic-higher subjects highlighted
  - corrected paired BAG shift plot with baseline-higher subjects highlighted


Global corrected outputs:
  - corrected_global_bag_hist.png
  - corrected_global_heatmap_MAE.png


Important:
  This script does NOT compute corrected predicted age.
  This script does NOT claim improved model performance.
  Corrected BAG is used only as a sensitivity/error-correction analysis.
""".strip()


    (args.output_dir / "README_statistical_strategy.txt").write_text(readme, encoding="utf-8")


    print("\nDONE")
    print(f"Saved outputs to: {args.output_dir}")
    print(f"Main stats CSV: {args.output_dir / 'all_statistical_tests.csv'}")
    print(f"Main stats TXT: {args.output_dir / 'all_statistical_tests_summary.txt'}")
    print(f"Corrected BAG histogram: {args.output_dir / 'corrected_global_bag_hist.png'}")
    print(f"Corrected MAE heatmap: {args.output_dir / 'corrected_global_heatmap_MAE.png'}")




if __name__ == "__main__":
    main()


