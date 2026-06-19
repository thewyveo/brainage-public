#!/usr/bin/env python3
# -*- coding: utf-8 -*-


r"""


py -3.10 exp_0\processing\postprocessing\overall\auto_postprocess_predictions_updated.py `
  --pred-root "//vumc.nl/Onderzoek/s4e-gpfs2/rath-research-01/Research/neuroRT/students/KayraOzdemir/rerun/pred" `
  --output-dir data\ALLPOSTPROCESS `
  --permutations 10000




Automatically postprocess all healthy / Exp0 / Exp1 brain-age prediction CSVs.


Expected folder layout:


pred_root/
  healthy/
    IXI_JOOS.csv or JOOS_IXI.csv or JOOS.csv
    IXI_BNX.csv  or BNX_IXI.csv  or BNX.csv


  exp0/
    BNX_IXI_CM.csv
    BNX_IXI_GLI.csv
    BNX_IXI_USB.csv
    JOOS_IXI_CM.csv
    JOOS_IXI_GLI.csv
    JOOS_IXI_USB.csv


  exp1/
    BNX_CM_LIT.csv
    BNX_CM_BID.csv
    BNX_CM_USB.csv
    BNX_GLI_LIT.csv
    BNX_GLI_BID.csv
    BNX_GLI_USB.csv
    JOOS_CM_LIT.csv
    JOOS_CM_BID.csv
    JOOS_CM_USB.csv
    JOOS_GLI_LIT.csv
    JOOS_GLI_BID.csv
    JOOS_GLI_USB.csv


What it does:


Exp0:
  For each MODEL + GENERATOR:
    healthy MODEL predictions vs Exp0 MODEL_GENERATOR predictions


Exp1:
  For each MODEL + GENERATOR + INPAINTER:
    healthy MODEL predictions vs Exp0 MODEL_GENERATOR predictions vs Exp1 MODEL_GENERATOR_INPAINTER predictions


Primary analysis:
  Raw BAG / delta-BAG with age-covariate intercept tests.


Sensitivity analysis:
  Age-residualized BAG fitted from the healthy baseline only.


Age exclusion:
  Subjects with Age <= --min-age-exclusive are excluded everywhere before pairing.
  Default is 25.0, matching the adult-only analysis rule.


QC exclusion matching thesis:
  By default, Exp1 analyses derived from USB synthetic tumors are skipped because
  USB-derived inpainting outputs were excluded entirely after visual QC.
  Use --include-usb-derived-exp1 if you explicitly want to analyze them anyway.


Outputs:
  output_dir/raw/...
  output_dir/age_residualized/...
  output_dir/all_raw_statistical_tests.csv
  output_dir/all_age_residualized_statistical_tests.csv
  output_dir/all_raw_descriptive_summaries.csv
  output_dir/all_age_residualized_descriptive_summaries.csv
  output_dir/discovery_manifest.csv
  output_dir/run_summary.txt
  output_dir/interpretations.txt
  output_dir/thesis_table_raw.csv
  output_dir/thesis_table_age_residualized.csv
  output_dir/thesis_table_combined.csv


Example:
  python auto_postprocess_predictions.py \
    --pred-root "//vumc.nl/Onderzoek/s4e-gpfs2/rath-research-01/Research/neuroRT/students/KayraOzdemir/rerun/pred" \
    --output-dir "./postprocessed_all" \
    --permutations 10000
"""


from __future__ import annotations


import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats




MODELS = {"BNX", "JOOS"}
GENERATORS = {"CM", "GLI", "USB"}
INPAINTERS = {"LIT", "BID", "USB"}


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


DEFAULT_MIN_AGE_EXCLUSIVE = 25.0


# Thesis/QC rule: USB (H2P)-derived inpainting outputs were excluded entirely
# after visual quality control. This means Exp1 combinations where the synthetic
# tumor generator is USB should be skipped unless explicitly requested.
DEFAULT_EXCLUDE_USB_DERIVED_EXP1 = True




# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------




def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None




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




def mean_ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]


    if len(values) == 0:
        return np.nan, np.nan


    mean = float(np.mean(values))


    if len(values) < 2:
        return mean, mean


    se = stats.sem(values, nan_policy="omit")
    tcrit = stats.t.ppf(1 - alpha / 2, df=len(values) - 1)
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




# ---------------------------------------------------------------------
# CSV loading / normalization
# ---------------------------------------------------------------------




def normalize_prediction_csv(path: Path, label: str, min_age_exclusive: Optional[float] = DEFAULT_MIN_AGE_EXCLUSIVE) -> pd.DataFrame:
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
    out["Source_CSV"] = str(path)


    out = out.dropna(subset=["IXI_ID", "Age", "Predicted_Brain_Age", "BAG"])

    if min_age_exclusive is not None:
        before = len(out)
        out = out[out["Age"] > float(min_age_exclusive)].copy()
        out["Age_Filter_Min_Exclusive"] = float(min_age_exclusive)
        out["Rows_Removed_By_Age_Filter_In_Source"] = before - len(out)

    return out.reset_index(drop=True)




def compute_metrics(df: pd.DataFrame, label: str) -> Dict[str, object]:
    age = df["Age"].to_numpy(float)
    pred = df["Predicted_Brain_Age"].to_numpy(float)
    bag = df["BAG"].to_numpy(float)


    return {
        "label": label,
        "n": int(len(df)),
        "age_mean": float(np.mean(age)) if len(age) else np.nan,
        "age_std": float(np.std(age)) if len(age) else np.nan,
        "predicted_brain_age_mean": float(np.mean(pred)) if len(pred) else np.nan,
        "predicted_brain_age_std": float(np.std(pred)) if len(pred) else np.nan,
        "BAG_mean": float(np.mean(bag)) if len(bag) else np.nan,
        "BAG_std": float(np.std(bag)) if len(bag) else np.nan,
        "BAG_median": float(np.median(bag)) if len(bag) else np.nan,
        "mean_absolute_BAG": float(np.mean(np.abs(bag))) if len(bag) else np.nan,
        "MAE": float(np.mean(np.abs(bag))) if len(bag) else np.nan,
        "RMSE": float(np.sqrt(np.mean(bag ** 2))) if len(bag) else np.nan,
        "corr_age_vs_pred": safe_corr(age, pred),
        "corr_age_vs_BAG": safe_corr(age, bag),
    }


def summarize_loaded_cohort(label: str, df: pd.DataFrame) -> Dict[str, object]:
    return {
        "label": label,
        "n_after_age_filter": int(len(df)),
        "age_min": float(df["Age"].min()) if len(df) else np.nan,
        "age_max": float(df["Age"].max()) if len(df) else np.nan,
        "age_mean": float(df["Age"].mean()) if len(df) else np.nan,
        "age_std": float(df["Age"].std(ddof=1)) if len(df) > 1 else np.nan,
        "source_csv": str(df["Source_CSV"].iloc[0]) if len(df) and "Source_CSV" in df.columns else "",
    }




# ---------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------




def strip_known_suffix(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".csv"):
        name = name[:-4]
    return name




def split_tokens(path: Path) -> List[str]:
    stem = strip_known_suffix(path)
    return [t.upper() for t in re.split(r"[^A-Za-z0-9]+", stem) if t]




def discover_healthy(healthy_dir: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}


    if not healthy_dir.exists():
        raise FileNotFoundError(f"Missing healthy directory: {healthy_dir}")


    for csv_path in sorted(healthy_dir.glob("*.csv")):
        toks = split_tokens(csv_path)
        model_hits = [m for m in MODELS if m in toks]


        if len(model_hits) != 1:
            continue


        model = model_hits[0]
        if model in found:
            raise RuntimeError(
                f"Multiple healthy CSVs detected for model {model}:\n"
                f"  {found[model]}\n"
                f"  {csv_path}\n"
                "Rename/remove duplicates or pass a cleaner pred-root."
            )
        found[model] = csv_path


    return found




def discover_exp0(exp0_dir: Path) -> Dict[Tuple[str, str], Path]:
    found: Dict[Tuple[str, str], Path] = {}


    if not exp0_dir.exists():
        raise FileNotFoundError(f"Missing exp0 directory: {exp0_dir}")


    for csv_path in sorted(exp0_dir.glob("*.csv")):
        toks = split_tokens(csv_path)
        model_hits = [m for m in MODELS if m in toks]
        gen_hits = [g for g in GENERATORS if g in toks]


        if len(model_hits) != 1 or len(gen_hits) != 1:
            continue


        key = (model_hits[0], gen_hits[0])
        if key in found:
            raise RuntimeError(
                f"Multiple Exp0 CSVs detected for {key}:\n"
                f"  {found[key]}\n"
                f"  {csv_path}"
            )
        found[key] = csv_path


    return found




def discover_exp1(exp1_dir: Path) -> Dict[Tuple[str, str, str], Path]:
    found: Dict[Tuple[str, str, str], Path] = {}


    if not exp1_dir.exists():
        raise FileNotFoundError(f"Missing exp1 directory: {exp1_dir}")


    for csv_path in sorted(exp1_dir.glob("*.csv")):
        toks = split_tokens(csv_path)
        model_hits = [m for m in MODELS if m in toks]
        gen_hits = [g for g in GENERATORS if g in toks]
        inp_hits = [i for i in INPAINTERS if i in toks]


        # Important ambiguity: USB can be generator and inpainter.
        # Exp1 filenames are MODEL_GENERATOR_INPAINTER, e.g. JOOS_GLI_USB.
        # So infer from the last 3 meaningful tokens if possible.
        if len(toks) >= 3 and toks[-3] in MODELS and toks[-2] in GENERATORS and toks[-1] in INPAINTERS:
            key = (toks[-3], toks[-2], toks[-1])
        elif len(model_hits) == 1 and len(gen_hits) >= 1 and len(inp_hits) >= 1:
            model = model_hits[0]
            inp = toks[-1] if toks[-1] in INPAINTERS else inp_hits[-1]
            gen_candidates = [g for g in gen_hits if g != inp]
            if not gen_candidates and inp == "USB" and "USB" in gen_hits:
                # Only safe if filename pattern made it clear; otherwise skip.
                continue
            if not gen_candidates:
                continue
            gen = gen_candidates[-1]
            key = (model, gen, inp)
        else:
            continue


        if key in found:
            raise RuntimeError(
                f"Multiple Exp1 CSVs detected for {key}:\n"
                f"  {found[key]}\n"
                f"  {csv_path}"
            )
        found[key] = csv_path


    return found




def write_discovery_manifest(
    output_dir: Path,
    healthy: Dict[str, Path],
    exp0: Dict[Tuple[str, str], Path],
    exp1: Dict[Tuple[str, str, str], Path],
) -> None:
    rows: List[Dict[str, object]] = []


    for model, path in sorted(healthy.items()):
        rows.append({"set": "healthy", "model": model, "generator": "", "inpainter": "", "path": str(path)})


    for (model, gen), path in sorted(exp0.items()):
        rows.append({"set": "exp0", "model": model, "generator": gen, "inpainter": "", "path": str(path)})


    for (model, gen, inp), path in sorted(exp1.items()):
        rows.append({"set": "exp1", "model": model, "generator": gen, "inpainter": inp, "path": str(path)})


    pd.DataFrame(rows).to_csv(output_dir / "discovery_manifest.csv", index=False)




# ---------------------------------------------------------------------
# Age correction
# ---------------------------------------------------------------------




def fit_age_correction_from_baseline(baseline_df: pd.DataFrame) -> Tuple[float, float]:
    age = baseline_df["Age"].to_numpy(float)
    bag = baseline_df["BAG"].to_numpy(float)


    valid = np.isfinite(age) & np.isfinite(bag)
    age = age[valid]
    bag = bag[valid]


    if len(age) < 2:
        raise RuntimeError("Need at least two valid baseline rows for age correction.")


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




# ---------------------------------------------------------------------
# Pair/triplet construction
# ---------------------------------------------------------------------




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
    merged["inpaint_closer_to_base"] = merged["inpaint_distance_from_base"] < merged["synth_distance_from_base"]
    merged["Inpaint_Closer_To_Healthy"] = merged["inpaint_closer_to_base"]
    merged["Percent_Recovery"] = np.where(
        merged["Abs_Delta_BAG_tumor"] > 1e-12,
        100.0 * merged["Recovery"] / merged["Abs_Delta_BAG_tumor"],
        np.nan,
    )
    return merged.reset_index(drop=True)




# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------




def one_sample_stat_tests(
    values: np.ndarray,
    test_name: str,
    permutations: int,
    alternative: str = "two-sided",
) -> Dict[str, object]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    ci_low, ci_high = mean_ci(values)


    row: Dict[str, object] = {
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


        row["permutation_p"] = sign_flip_permutation_test(values, permutations=permutations, alternative=alternative)
    else:
        row["t_stat"] = np.nan
        row["t_p"] = np.nan
        row["wilcoxon_stat"] = np.nan
        row["wilcoxon_p"] = np.nan
        row["permutation_p"] = np.nan


    return row




def age_covariate_intercept_test(values: np.ndarray, age: np.ndarray) -> Dict[str, object]:
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




def run_generator_statistics(
    paired: pd.DataFrame,
    analysis_name: str,
    model: str,
    generator: str,
    bag_version: str,
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
        {
            "test_name": "synthetic_minus_healthy_absolute_error",
            "values": paired["ABS_ERR_diff"].to_numpy(float),
            "alternative": "two-sided",
            "meaning": "Tests whether absolute BAG error differs between synthetic tumor and healthy baseline.",
        },
    ]


    rows: List[Dict[str, object]] = []
    for item in tests:
        row = one_sample_stat_tests(
            values=item["values"],
            test_name=item["test_name"],
            permutations=permutations,
            alternative=item["alternative"],
        )
        row.update(age_covariate_intercept_test(item["values"], age))
        row.update(
            {
                "analysis": analysis_name,
                "experiment": "exp0",
                "model": model,
                "generator": generator,
                "inpainter": "",
                "bag_version": bag_version,
                "meaning": item["meaning"],
            }
        )
        rows.append(row)


    return pd.DataFrame(rows)




def run_triplet_statistics(
    triplet: pd.DataFrame,
    analysis_name: str,
    model: str,
    generator: str,
    inpainter: str,
    bag_version: str,
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


    rows: List[Dict[str, object]] = []
    for item in tests:
        row = one_sample_stat_tests(
            values=item["values"],
            test_name=item["test_name"],
            permutations=permutations,
            alternative=item["alternative"],
        )
        row.update(age_covariate_intercept_test(item["values"], age))
        row.update(
            {
                "analysis": analysis_name,
                "experiment": "exp1",
                "model": model,
                "generator": generator,
                "inpainter": inpainter,
                "bag_version": bag_version,
                "meaning": item["meaning"],
            }
        )
        rows.append(row)


    return pd.DataFrame(rows)




def summarize_generator_descriptives(
    paired: pd.DataFrame,
    analysis_name: str,
    model: str,
    generator: str,
    bag_version: str,
) -> Dict[str, object]:
    return {
        "analysis": analysis_name,
        "experiment": "exp0",
        "model": model,
        "generator": generator,
        "inpainter": "",
        "bag_version": bag_version,
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




def summarize_triplet_descriptives(
    triplet: pd.DataFrame,
    analysis_name: str,
    model: str,
    generator: str,
    inpainter: str,
    bag_version: str,
) -> Dict[str, object]:
    return {
        "analysis": analysis_name,
        "experiment": "exp1",
        "model": model,
        "generator": generator,
        "inpainter": inpainter,
        "bag_version": bag_version,
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




# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------




def bag_paired_shift_plot(df: pd.DataFrame, out_path: Path, title: str, ylabel: str) -> None:
    d = df.sort_values(["Age", "IXI_ID"]).reset_index(drop=True)
    x = np.arange(len(d))


    plt.figure(figsize=(12, 6))


    for i, row in d.iterrows():
        plt.plot([i, i], [row["BAD_base"], row["BAD_synth"]], color="gray", alpha=0.25, linewidth=1)


    plt.scatter(x, d["BAD_base"], color="seagreen", label="Healthy baseline BAG", s=18)
    plt.scatter(x, d["BAD_synth"], color="darkorange", label="Synthetic tumor BAG", s=18)
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Subjects ranked by age")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def bag_paired_shift_plot_synthetic_higher_highlight(df: pd.DataFrame, out_path: Path, title: str, ylabel: str) -> None:
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


    plt.scatter(x, d["BAD_base"], color="seagreen", label="Healthy baseline BAG", s=18, zorder=2)


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
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def bag_paired_shift_plot_baseline_higher_highlight(df: pd.DataFrame, out_path: Path, title: str, ylabel: str) -> None:
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


    plt.scatter(x, d["BAD_synth"], color="darkorange", label="Synthetic BAG", s=18, zorder=3)
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Subjects ranked by age")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def bag_triplet_shift_plot_closer_to_baseline(df: pd.DataFrame, out_path: Path, title: str, ylabel: str) -> None:
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


    plt.scatter(x, d["BAD_base"], marker="o", color="black", label="Healthy baseline BAG", s=20, zorder=4)
    plt.scatter(x, d["BAD_synth"], marker="^", color="darkorange", label="Synthetic tumor BAG", s=24, alpha=0.85, zorder=5)
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
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def save_global_bag_hist(all_df: pd.DataFrame, out_path: Path, title: str, xlabel: str) -> None:
    plt.figure(figsize=(8, 5))


    for label, g in all_df.groupby("Set_Label"):
        plt.hist(g["BAG"].to_numpy(float), bins=30, alpha=0.35, label=label)


    plt.axvline(0, linestyle="--", color="black")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def save_heatmap(summary_df: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    if summary_df.empty or metric not in summary_df.columns:
        return


    pivot = summary_df.set_index("label")[[metric]]
    arr = pivot.to_numpy(float)


    plt.figure(figsize=(6, max(4, 0.45 * len(pivot) + 2)))
    im = plt.imshow(arr, aspect="auto")
    plt.colorbar(im, label=metric)


    plt.xticks([0], [metric])
    plt.yticks(np.arange(len(pivot)), pivot.index)


    for i in range(arr.shape[0]):
        val = arr[i, 0]
        text = "nan" if not np.isfinite(val) else f"{val:.2f}"
        plt.text(0, i, text, ha="center", va="center")


    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------




def write_txt_summary(stats_df: pd.DataFrame, desc_df: pd.DataFrame, out_path: Path, bag_version: str) -> None:
    lines: List[str] = []
    lines.append(f"SUMMARY OF STATISTICAL TESTS: {bag_version}")
    lines.append("=" * (31 + len(bag_version)))
    lines.append("")
    lines.append("Interpretation note:")
    if bag_version == "raw":
        lines.append("  This is the primary analysis: raw BAG / delta-BAG.")
        lines.append("  Age is handled through age-covariate intercept tests.")
    else:
        lines.append("  This is the sensitivity analysis: BAG residualized against age.")
        lines.append("  The age fit is learned from the healthy baseline only, separately for each model.")
    lines.append("")


    lines.append("DESCRIPTIVE SUMMARIES")
    lines.append("=====================")
    lines.append(desc_df.to_string(index=False) if not desc_df.empty else "No descriptive summaries.")
    lines.append("")


    lines.append("STATISTICAL TESTS")
    lines.append("=================")


    if stats_df.empty:
        lines.append("No statistical tests.")
    else:
        for _, row in stats_df.iterrows():
            label_parts = [str(row.get("model", "")), str(row.get("generator", ""))]
            if str(row.get("inpainter", "")):
                label_parts.append(str(row.get("inpainter", "")))
            label = "_".join([p for p in label_parts if p])


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




def run_single_bag_version(
    bag_version: str,
    output_dir: Path,
    healthy_dfs: Dict[str, pd.DataFrame],
    exp0_dfs: Dict[Tuple[str, str], pd.DataFrame],
    exp1_dfs: Dict[Tuple[str, str, str], pd.DataFrame],
    permutations: int,
    correction_params: Optional[Dict[str, Tuple[float, float]]] = None,
    exclude_usb_derived_exp1: bool = DEFAULT_EXCLUDE_USB_DERIVED_EXP1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    version_out = output_dir / bag_version
    version_out.mkdir(parents=True, exist_ok=True)


    all_stats: List[pd.DataFrame] = []
    all_desc: List[Dict[str, object]] = []
    all_predictions: List[pd.DataFrame] = []
    all_metrics: List[Dict[str, object]] = []


    ylabel = "BAG" if bag_version == "raw" else "Age-residualized BAG"


    for model, healthy_df in sorted(healthy_dfs.items()):
        if correction_params is not None:
            alpha, beta = correction_params[model]
            healthy_work = apply_age_residualized_bag(healthy_df, alpha, beta)
        else:
            alpha, beta = np.nan, np.nan
            healthy_work = healthy_df.copy()


        healthy_work["Set_Label"] = f"{model}_healthy"
        all_predictions.append(healthy_work)
        all_metrics.append(compute_metrics(healthy_work, f"{model}_healthy"))


        for generator in sorted(GENERATORS):
            exp0_key = (model, generator)
            if exp0_key not in exp0_dfs:
                print(f"[SKIP Exp0] Missing Exp0 CSV for {model}_{generator}")
                continue


            exp0_df = exp0_dfs[exp0_key]
            if correction_params is not None:
                exp0_work = apply_age_residualized_bag(exp0_df, alpha, beta)
            else:
                exp0_work = exp0_df.copy()


            exp0_label = f"{model}_{generator}_exp0"
            exp0_work["Set_Label"] = exp0_label
            all_predictions.append(exp0_work)
            all_metrics.append(compute_metrics(exp0_work, exp0_label))


            exp0_out = version_out / "exp0" / model / generator
            exp0_out.mkdir(parents=True, exist_ok=True)


            paired = build_paired_df(healthy_work, exp0_work)
            paired.to_csv(exp0_out / f"paired_{bag_version}_BAG.csv", index=False)


            bag_paired_shift_plot(
                paired,
                exp0_out / f"paired_{bag_version}_BAG_shift.png",
                title=f"Exp0 {model}-{generator}: healthy vs synthetic tumor ({bag_version})",
                ylabel=ylabel,
            )
            bag_paired_shift_plot_synthetic_higher_highlight(
                paired,
                exp0_out / f"paired_{bag_version}_BAG_shift_synthetic_higher_highlight.png",
                title=f"Exp0 {model}-{generator}: synthetic higher highlighted ({bag_version})",
                ylabel=ylabel,
            )
            bag_paired_shift_plot_baseline_higher_highlight(
                paired,
                exp0_out / f"paired_{bag_version}_BAG_shift_baseline_higher_highlight.png",
                title=f"Exp0 {model}-{generator}: baseline higher highlighted ({bag_version})",
                ylabel=ylabel,
            )


            stats_df = run_generator_statistics(
                paired=paired,
                analysis_name=f"exp0_{bag_version}_BAG",
                model=model,
                generator=generator,
                bag_version=bag_version,
                permutations=permutations,
            )
            stats_df["age_correction_alpha"] = alpha
            stats_df["age_correction_beta"] = beta
            stats_df.to_csv(exp0_out / "statistical_tests.csv", index=False)
            all_stats.append(stats_df)


            desc = summarize_generator_descriptives(
                paired=paired,
                analysis_name=f"exp0_{bag_version}_BAG",
                model=model,
                generator=generator,
                bag_version=bag_version,
            )
            desc["age_correction_alpha"] = alpha
            desc["age_correction_beta"] = beta
            pd.DataFrame([desc]).to_csv(exp0_out / "descriptive_summary.csv", index=False)
            all_desc.append(desc)


            for inpainter in sorted(INPAINTERS):
                if exclude_usb_derived_exp1 and generator == "USB":
                    print(f"[SKIP Exp1 QC] Skipping {model}_{generator}_{inpainter}: USB-derived inpainting excluded by thesis visual QC rule.")
                    continue

                exp1_key = (model, generator, inpainter)
                if exp1_key not in exp1_dfs:
                    print(f"[SKIP Exp1] Missing Exp1 CSV for {model}_{generator}_{inpainter}")
                    continue


                exp1_df = exp1_dfs[exp1_key]
                if correction_params is not None:
                    exp1_work = apply_age_residualized_bag(exp1_df, alpha, beta)
                else:
                    exp1_work = exp1_df.copy()


                exp1_label = f"{model}_{generator}_{inpainter}_exp1"
                exp1_work["Set_Label"] = exp1_label
                all_predictions.append(exp1_work)
                all_metrics.append(compute_metrics(exp1_work, exp1_label))


                exp1_out = version_out / "exp1" / model / generator / inpainter
                exp1_out.mkdir(parents=True, exist_ok=True)


                triplet = build_triplet_df(healthy_work, exp0_work, exp1_work)
                triplet.to_csv(exp1_out / f"triplet_{bag_version}_BAG.csv", index=False)


                bag_triplet_shift_plot_closer_to_baseline(
                    triplet,
                    exp1_out / f"triplet_{bag_version}_BAG_shift_closer_to_healthy.png",
                    title=f"Exp1 {model}-{generator}-{inpainter}: healthy vs tumor vs inpainted ({bag_version})",
                    ylabel=ylabel,
                )


                stats_df = run_triplet_statistics(
                    triplet=triplet,
                    analysis_name=f"exp1_{bag_version}_BAG",
                    model=model,
                    generator=generator,
                    inpainter=inpainter,
                    bag_version=bag_version,
                    permutations=permutations,
                )
                stats_df["age_correction_alpha"] = alpha
                stats_df["age_correction_beta"] = beta
                stats_df.to_csv(exp1_out / "statistical_tests.csv", index=False)
                all_stats.append(stats_df)


                desc = summarize_triplet_descriptives(
                    triplet=triplet,
                    analysis_name=f"exp1_{bag_version}_BAG",
                    model=model,
                    generator=generator,
                    inpainter=inpainter,
                    bag_version=bag_version,
                )
                desc["age_correction_alpha"] = alpha
                desc["age_correction_beta"] = beta
                pd.DataFrame([desc]).to_csv(exp1_out / "descriptive_summary.csv", index=False)
                all_desc.append(desc)


    final_stats = pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()
    final_desc = pd.DataFrame(all_desc)
    final_predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    final_metrics = pd.DataFrame(all_metrics)


    final_stats.to_csv(output_dir / f"all_{bag_version}_statistical_tests.csv", index=False)
    final_desc.to_csv(output_dir / f"all_{bag_version}_descriptive_summaries.csv", index=False)
    final_predictions.to_csv(output_dir / f"all_{bag_version}_combined_predictions.csv", index=False)
    final_metrics.to_csv(output_dir / f"all_{bag_version}_global_model_summary.csv", index=False)


    if not final_predictions.empty:
        save_global_bag_hist(
            final_predictions,
            output_dir / f"all_{bag_version}_global_BAG_hist.png",
            title=f"Global BAG distribution ({bag_version})",
            xlabel=ylabel,
        )


    if not final_metrics.empty:
        save_heatmap(
            final_metrics,
            metric="MAE",
            out_path=output_dir / f"all_{bag_version}_global_heatmap_MAE.png",
            title=f"Global heatmap: MAE / mean absolute BAG ({bag_version})",
        )


    write_txt_summary(
        stats_df=final_stats,
        desc_df=final_desc,
        out_path=output_dir / f"all_{bag_version}_statistical_tests_summary.txt",
        bag_version=bag_version,
    )


    return final_stats, final_desc, final_predictions, final_metrics





def p_to_stars(p: object) -> str:
    try:
        p = float(p)
    except Exception:
        return ""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def fmt_float(x: object, digits: int = 2, signed: bool = False) -> str:
    try:
        x = float(x)
    except Exception:
        return ""
    if not np.isfinite(x):
        return ""
    prefix = "+" if signed and x > 0 else ""
    return f"{prefix}{x:.{digits}f}"


def get_stat_row(stats_df: pd.DataFrame, experiment: str, model: str, generator: str, inpainter: str, test_name: str) -> Optional[pd.Series]:
    if stats_df.empty:
        return None
    mask = (
        (stats_df["experiment"].astype(str) == experiment)
        & (stats_df["model"].astype(str) == model)
        & (stats_df["generator"].astype(str) == generator)
        & (stats_df["inpainter"].fillna("").astype(str) == inpainter)
        & (stats_df["test_name"].astype(str) == test_name)
    )
    sub = stats_df.loc[mask]
    if sub.empty:
        return None
    return sub.iloc[0]


def make_thesis_table(stats_df: pd.DataFrame, desc_df: pd.DataFrame, bag_version: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if desc_df.empty:
        return pd.DataFrame()

    for _, d in desc_df.iterrows():
        experiment = str(d.get("experiment", ""))
        model = str(d.get("model", ""))
        generator = str(d.get("generator", ""))
        inpainter = str(d.get("inpainter", "")) if pd.notna(d.get("inpainter", "")) else ""

        if experiment == "exp0":
            signed = get_stat_row(stats_df, "exp0", model, generator, "", "generator_signed_delta_BAG")
            absrow = get_stat_row(stats_df, "exp0", model, generator, "", "generator_abs_delta_BAG")
            errrow = get_stat_row(stats_df, "exp0", model, generator, "", "synthetic_minus_healthy_absolute_error")
            condition = f"{model}_{generator}"
            wilcoxon_p = signed.get("wilcoxon_p", np.nan) if signed is not None else np.nan
            rows.append({
                "bag_version": bag_version,
                "experiment": "Exp0",
                "model": model,
                "condition": condition,
                "generator": generator,
                "inpainter": "",
                "n": int(d.get("n", 0)),
                "Delta_BAG_mean": d.get("mean_delta_BAG", np.nan),
                "Delta_BAG": fmt_float(d.get("mean_delta_BAG", np.nan), signed=True) + p_to_stars(wilcoxon_p),
                "Abs_Delta_BAG_mean": d.get("mean_abs_delta_BAG", np.nan),
                "Abs_Delta_BAG": fmt_float(d.get("mean_abs_delta_BAG", np.nan)) + p_to_stars(absrow.get("wilcoxon_p", np.nan) if absrow is not None else np.nan),
                "Delta_Abs_BAG_mean": errrow.get("mean", np.nan) if errrow is not None else np.nan,
                "Delta_Abs_BAG": fmt_float(errrow.get("mean", np.nan), signed=True) + p_to_stars(errrow.get("wilcoxon_p", np.nan)) if errrow is not None else "",
                "Recovery_mean": "",
                "Recovery": "",
                "Proportion_Closer": "",
                "Wilcoxon_p_signed_delta": wilcoxon_p,
                "Permutation_p_signed_delta": signed.get("permutation_p", np.nan) if signed is not None else np.nan,
                "Cohens_d_signed_delta": signed.get("cohens_d_paired", np.nan) if signed is not None else np.nan,
                "Age_covariate_p_signed_delta": signed.get("age_adjusted_intercept_p", np.nan) if signed is not None else np.nan,
            })
        elif experiment == "exp1":
            signed_tumor = get_stat_row(stats_df, "exp1", model, generator, inpainter, "tumor_induced_signed_delta_BAG")
            signed_inpaint = get_stat_row(stats_df, "exp1", model, generator, inpainter, "inpaint_signed_delta_BAG")
            abs_tumor = get_stat_row(stats_df, "exp1", model, generator, inpainter, "tumor_abs_delta_BAG")
            abs_inpaint = get_stat_row(stats_df, "exp1", model, generator, inpainter, "inpaint_abs_delta_BAG")
            recovery = get_stat_row(stats_df, "exp1", model, generator, inpainter, "recovery_abs_tumor_minus_abs_inpaint")
            condition = f"{model}_{generator}_{inpainter}"
            rec_p = recovery.get("wilcoxon_p", np.nan) if recovery is not None else np.nan
            rows.append({
                "bag_version": bag_version,
                "experiment": "Exp1",
                "model": model,
                "condition": condition,
                "generator": generator,
                "inpainter": inpainter,
                "n": int(d.get("n", 0)),
                "Delta_BAG_tumor_mean": d.get("mean_Delta_BAG_tumor", np.nan),
                "Delta_BAG_tumor": fmt_float(d.get("mean_Delta_BAG_tumor", np.nan), signed=True) + p_to_stars(signed_tumor.get("wilcoxon_p", np.nan) if signed_tumor is not None else np.nan),
                "Abs_Delta_BAG_tumor_mean": d.get("mean_abs_Delta_BAG_tumor", np.nan),
                "Abs_Delta_BAG_tumor": fmt_float(d.get("mean_abs_Delta_BAG_tumor", np.nan)) + p_to_stars(abs_tumor.get("wilcoxon_p", np.nan) if abs_tumor is not None else np.nan),
                "Delta_BAG_inpaint_mean": d.get("mean_Delta_BAG_inpaint", np.nan),
                "Delta_BAG_inpaint": fmt_float(d.get("mean_Delta_BAG_inpaint", np.nan), signed=True) + p_to_stars(signed_inpaint.get("wilcoxon_p", np.nan) if signed_inpaint is not None else np.nan),
                "Abs_Delta_BAG_inpaint_mean": d.get("mean_abs_Delta_BAG_inpaint", np.nan),
                "Abs_Delta_BAG_inpaint": fmt_float(d.get("mean_abs_Delta_BAG_inpaint", np.nan)) + p_to_stars(abs_inpaint.get("wilcoxon_p", np.nan) if abs_inpaint is not None else np.nan),
                "Recovery_mean": d.get("mean_recovery", np.nan),
                "Recovery": fmt_float(d.get("mean_recovery", np.nan), signed=True) + p_to_stars(rec_p),
                "Proportion_Closer": fmt_float(d.get("proportion_inpaint_closer", np.nan), digits=3),
                "Count_Closer": d.get("count_inpaint_closer", np.nan),
                "Wilcoxon_p_recovery": rec_p,
                "Permutation_p_recovery": recovery.get("permutation_p", np.nan) if recovery is not None else np.nan,
                "Cohens_d_recovery": recovery.get("cohens_d_paired", np.nan) if recovery is not None else np.nan,
                "Age_covariate_p_recovery": recovery.get("age_adjusted_intercept_p", np.nan) if recovery is not None else np.nan,
            })

    return pd.DataFrame(rows)


def interpretation_for_stat(row: pd.Series) -> str:
    experiment = str(row.get("experiment", ""))
    model = str(row.get("model", ""))
    generator = str(row.get("generator", ""))
    inpainter = str(row.get("inpainter", ""))
    bag_version = str(row.get("bag_version", ""))
    test = str(row.get("test_name", ""))
    mean = float(row.get("mean", np.nan))
    wilcoxon_p = float(row.get("wilcoxon_p", np.nan)) if pd.notna(row.get("wilcoxon_p", np.nan)) else np.nan
    perm_p = float(row.get("permutation_p", np.nan)) if pd.notna(row.get("permutation_p", np.nan)) else np.nan
    d = float(row.get("cohens_d_paired", np.nan)) if pd.notna(row.get("cohens_d_paired", np.nan)) else np.nan
    age_p = float(row.get("age_adjusted_intercept_p", np.nan)) if pd.notna(row.get("age_adjusted_intercept_p", np.nan)) else np.nan
    n = int(row.get("n", 0)) if pd.notna(row.get("n", np.nan)) else 0

    label = f"{model}_{generator}" if experiment == "exp0" else f"{model}_{generator}_{inpainter}"
    sig = "statistically significant" if np.isfinite(wilcoxon_p) and wilcoxon_p < 0.05 else "not statistically significant"
    robust = "robust in the sign-flip permutation test" if np.isfinite(perm_p) and perm_p < 0.05 else "not robust in the sign-flip permutation test"
    age_adj = "remains significant after age-covariate intercept testing" if np.isfinite(age_p) and age_p < 0.05 else "does not remain significant after age-covariate intercept testing"
    effect = "small"
    if np.isfinite(d):
        ad = abs(d)
        if ad >= 0.8:
            effect = "large"
        elif ad >= 0.5:
            effect = "medium"
        elif ad >= 0.2:
            effect = "small-to-moderate"

    if test == "generator_signed_delta_BAG":
        direction = "older" if mean > 0 else "younger"
        return f"[{bag_version}] {label}: Exp0 signed ΔBAG mean={mean:.3f} years (n={n}); tumor generation shifts predictions {direction} relative to healthy. Wilcoxon result is {sig} (p={wilcoxon_p:.3g}), {robust}, {age_adj}; paired effect size is {effect} (d={d:.3f})."
    if test == "generator_abs_delta_BAG":
        return f"[{bag_version}] {label}: Exp0 |ΔBAG| mean={mean:.3f} years; this is the non-directional tumor-induced perturbation magnitude. Wilcoxon result is {sig} (p={wilcoxon_p:.3g}), {robust}; effect size is {effect} (d={d:.3f})."
    if test == "synthetic_minus_healthy_absolute_error":
        direction = "increases" if mean > 0 else "decreases"
        return f"[{bag_version}] {label}: Exp0 Δ|BAG| mean={mean:.3f} years; synthetic tumor condition {direction} absolute BAG error relative to healthy. Wilcoxon result is {sig} (p={wilcoxon_p:.3g}), {robust}, {age_adj}."
    if test == "tumor_induced_signed_delta_BAG":
        direction = "older" if mean > 0 else "younger"
        return f"[{bag_version}] {label}: Exp1 tumor reference ΔBAG mean={mean:.3f} years; the underlying synthetic tumor shifts predictions {direction}. Wilcoxon result is {sig} (p={wilcoxon_p:.3g}), {robust}, {age_adj}."
    if test == "inpaint_signed_delta_BAG":
        direction = "older" if mean > 0 else "younger"
        return f"[{bag_version}] {label}: Exp1 inpainted ΔBAG mean={mean:.3f} years relative to healthy; after inpainting, predictions are still shifted {direction}. Wilcoxon result is {sig} (p={wilcoxon_p:.3g}), {robust}, {age_adj}."
    if test == "tumor_abs_delta_BAG":
        return f"[{bag_version}] {label}: Exp1 tumor |ΔBAG| mean={mean:.3f} years; this is the tumor perturbation that inpainting tries to recover from. Wilcoxon result is {sig} (p={wilcoxon_p:.3g})."
    if test == "inpaint_abs_delta_BAG":
        return f"[{bag_version}] {label}: Exp1 inpainted |ΔBAG| mean={mean:.3f} years; this is the remaining distance from healthy after inpainting. Wilcoxon result is {sig} (p={wilcoxon_p:.3g})."
    if test == "recovery_abs_tumor_minus_abs_inpaint":
        if mean > 0:
            conclusion = "inpainting improves recovery by moving predictions closer to healthy"
        elif mean < 0:
            conclusion = "inpainting worsens recovery by moving predictions farther from healthy"
        else:
            conclusion = "inpainting shows no average recovery"
        return f"[{bag_version}] {label}: Recovery mean={mean:.3f} years; {conclusion}. Wilcoxon result is {sig} for the one-sided recovery test (p={wilcoxon_p:.3g}), {robust}; effect size is {effect} (d={d:.3f})."
    return f"[{bag_version}] {label}: {test} mean={mean:.3f}, Wilcoxon p={wilcoxon_p:.3g}."


def write_interpretations(
    raw_stats: pd.DataFrame,
    raw_desc: pd.DataFrame,
    corr_stats: pd.DataFrame,
    corr_desc: pd.DataFrame,
    raw_table: pd.DataFrame,
    corr_table: pd.DataFrame,
    output_path: Path,
    min_age_exclusive: float,
    exclude_usb_derived_exp1: bool,
) -> None:
    lines: List[str] = []
    lines.append("INTERPRETATIONS")
    lines.append("===============")
    lines.append("")
    lines.append(f"Age filter: all subjects with Age <= {min_age_exclusive:g} were excluded before pairing/statistics.")
    if exclude_usb_derived_exp1:
        lines.append("Visual QC rule: Exp1 combinations derived from USB synthetic tumors were skipped, matching the thesis statement that USB-derived inpainting outputs were excluded entirely.")
    else:
        lines.append("Visual QC rule: USB-derived Exp1 combinations were included because --include-usb-derived-exp1 was used.")
    lines.append("")
    lines.append("Primary analysis = raw BAG / ΔBAG with age-covariate intercept tests.")
    lines.append("Sensitivity analysis = age-residualized BAG fitted from healthy baseline only.")
    lines.append("Positive Recovery means |ΔBAG_tumor| > |ΔBAG_inpaint|, i.e. inpainting is closer to healthy than the synthetic tumor condition.")
    lines.append("")

    for name, table in [("RAW THESIS TABLE", raw_table), ("AGE-RESIDUALIZED THESIS TABLE", corr_table)]:
        lines.append(name)
        lines.append("-" * len(name))
        if table.empty:
            lines.append("No rows.")
        else:
            export_cols = [c for c in [
                "experiment", "model", "condition", "n", "Delta_BAG", "Abs_Delta_BAG", "Delta_Abs_BAG",
                "Delta_BAG_tumor", "Abs_Delta_BAG_tumor", "Delta_BAG_inpaint", "Abs_Delta_BAG_inpaint",
                "Recovery", "Proportion_Closer", "Wilcoxon_p_signed_delta", "Wilcoxon_p_recovery"
            ] if c in table.columns]
            lines.append(table[export_cols].to_string(index=False))
        lines.append("")

    for title, stats_df in [("RAW STATISTIC-BY-STATISTIC INTERPRETATIONS", raw_stats), ("AGE-RESIDUALIZED STATISTIC-BY-STATISTIC INTERPRETATIONS", corr_stats)]:
        lines.append(title)
        lines.append("-" * len(title))
        if stats_df.empty:
            lines.append("No statistical tests.")
        else:
            for _, row in stats_df.iterrows():
                lines.append(interpretation_for_stat(row))
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--pred-root",
        type=Path,
        required=True,
        help="Main prediction folder containing healthy/, exp0/, exp1/.",
    )
    parser.add_argument("--healthy-dir", type=Path, default=None, help="Optional override for healthy CSV directory.")
    parser.add_argument("--exp0-dir", type=Path, default=None, help="Optional override for Exp0 CSV directory.")
    parser.add_argument("--exp1-dir", type=Path, default=None, help="Optional override for Exp1 CSV directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument(
        "--only-models",
        nargs="*",
        default=None,
        help="Optional model filter, e.g. --only-models BNX JOOS",
    )
    parser.add_argument(
        "--only-generators",
        nargs="*",
        default=None,
        help="Optional generator filter, e.g. --only-generators CM GLI USB",
    )
    parser.add_argument(
        "--only-inpainters",
        nargs="*",
        default=None,
        help="Optional inpainter filter, e.g. --only-inpainters BID LIT USB",
    )
    parser.add_argument(
        "--min-age-exclusive",
        type=float,
        default=DEFAULT_MIN_AGE_EXCLUSIVE,
        help="Exclude subjects with Age <= this value before all pairing/statistics. Default: 25.0",
    )
    parser.add_argument(
        "--include-usb-derived-exp1",
        action="store_true",
        help="Analyze Exp1 combinations where the synthetic generator is USB. By default these are skipped to match thesis QC exclusions.",
    )


    return parser.parse_args()




def filter_discovered(
    healthy: Dict[str, Path],
    exp0: Dict[Tuple[str, str], Path],
    exp1: Dict[Tuple[str, str, str], Path],
    only_models: Optional[Iterable[str]],
    only_generators: Optional[Iterable[str]],
    only_inpainters: Optional[Iterable[str]],
) -> Tuple[Dict[str, Path], Dict[Tuple[str, str], Path], Dict[Tuple[str, str, str], Path]]:
    model_filter = {m.upper() for m in only_models} if only_models else None
    gen_filter = {g.upper() for g in only_generators} if only_generators else None
    inp_filter = {i.upper() for i in only_inpainters} if only_inpainters else None


    if model_filter is not None and not model_filter <= MODELS:
        raise ValueError(f"Invalid model filter: {model_filter}. Valid: {sorted(MODELS)}")
    if gen_filter is not None and not gen_filter <= GENERATORS:
        raise ValueError(f"Invalid generator filter: {gen_filter}. Valid: {sorted(GENERATORS)}")
    if inp_filter is not None and not inp_filter <= INPAINTERS:
        raise ValueError(f"Invalid inpainter filter: {inp_filter}. Valid: {sorted(INPAINTERS)}")


    healthy_f = {
        model: path
        for model, path in healthy.items()
        if model_filter is None or model in model_filter
    }


    exp0_f = {
        key: path
        for key, path in exp0.items()
        if (model_filter is None or key[0] in model_filter)
        and (gen_filter is None or key[1] in gen_filter)
    }


    exp1_f = {
        key: path
        for key, path in exp1.items()
        if (model_filter is None or key[0] in model_filter)
        and (gen_filter is None or key[1] in gen_filter)
        and (inp_filter is None or key[2] in inp_filter)
    }


    return healthy_f, exp0_f, exp1_f




def main() -> None:
    args = parse_args()


    healthy_dir = args.healthy_dir if args.healthy_dir is not None else args.pred_root / "healthy"
    exp0_dir = args.exp0_dir if args.exp0_dir is not None else args.pred_root / "exp0"
    exp1_dir = args.exp1_dir if args.exp1_dir is not None else args.pred_root / "exp1"


    args.output_dir.mkdir(parents=True, exist_ok=True)


    print("Discovering CSVs...")
    healthy_paths = discover_healthy(healthy_dir)
    exp0_paths = discover_exp0(exp0_dir)
    exp1_paths = discover_exp1(exp1_dir)


    healthy_paths, exp0_paths, exp1_paths = filter_discovered(
        healthy=healthy_paths,
        exp0=exp0_paths,
        exp1=exp1_paths,
        only_models=args.only_models,
        only_generators=args.only_generators,
        only_inpainters=args.only_inpainters,
    )


    write_discovery_manifest(args.output_dir, healthy_paths, exp0_paths, exp1_paths)


    print(f"Healthy CSVs: {len(healthy_paths)}")
    print(f"Exp0 CSVs:    {len(exp0_paths)}")
    print(f"Exp1 CSVs:    {len(exp1_paths)}")


    if not healthy_paths:
        raise RuntimeError("No healthy prediction CSVs found. Expected names containing BNX or JOOS in healthy/.")
    if not exp0_paths:
        raise RuntimeError("No Exp0 prediction CSVs found. Expected names like BNX_IXI_CM.csv in exp0/.")
    if not exp1_paths:
        print("WARNING: No Exp1 prediction CSVs found. Only Exp0 analyses will run.")


    print("Loading normalized CSVs...")
    healthy_dfs = {
        model: normalize_prediction_csv(path, f"{model}_healthy", min_age_exclusive=args.min_age_exclusive)
        for model, path in sorted(healthy_paths.items())
    }
    exp0_dfs = {
        key: normalize_prediction_csv(path, f"{key[0]}_{key[1]}_exp0", min_age_exclusive=args.min_age_exclusive)
        for key, path in sorted(exp0_paths.items())
    }
    exp1_dfs = {
        key: normalize_prediction_csv(path, f"{key[0]}_{key[1]}_{key[2]}_exp1", min_age_exclusive=args.min_age_exclusive)
        for key, path in sorted(exp1_paths.items())
    }


    cohort_rows: List[Dict[str, object]] = []
    for model, df in sorted(healthy_dfs.items()):
        cohort_rows.append(summarize_loaded_cohort(f"{model}_healthy", df))
    for (model, gen), df in sorted(exp0_dfs.items()):
        cohort_rows.append(summarize_loaded_cohort(f"{model}_{gen}_exp0", df))
    for (model, gen, inp), df in sorted(exp1_dfs.items()):
        cohort_rows.append(summarize_loaded_cohort(f"{model}_{gen}_{inp}_exp1", df))
    pd.DataFrame(cohort_rows).to_csv(args.output_dir / "cohort_after_age_filter_summary.csv", index=False)

    print("Fitting age correction from healthy baselines only...")
    correction_params: Dict[str, Tuple[float, float]] = {}
    correction_rows: List[Dict[str, object]] = []


    for model, df in sorted(healthy_dfs.items()):
        alpha, beta = fit_age_correction_from_baseline(df)
        correction_params[model] = (alpha, beta)
        correction_rows.append({"model": model, "alpha": alpha, "beta": beta, "fit_source": "healthy_baseline_only"})
        print(f"  {model}: alpha={alpha:.8f}, beta={beta:.8f}")


    pd.DataFrame(correction_rows).to_csv(args.output_dir / "age_correction_parameters.csv", index=False)


    print("Running raw BAG primary analysis...")
    raw_stats, raw_desc, _, _ = run_single_bag_version(
        bag_version="raw",
        output_dir=args.output_dir,
        healthy_dfs=healthy_dfs,
        exp0_dfs=exp0_dfs,
        exp1_dfs=exp1_dfs,
        permutations=args.permutations,
        correction_params=None,
        exclude_usb_derived_exp1=(not args.include_usb_derived_exp1),
    )


    print("Running age-residualized BAG sensitivity analysis...")
    corr_stats, corr_desc, _, _ = run_single_bag_version(
        bag_version="age_residualized",
        output_dir=args.output_dir,
        healthy_dfs=healthy_dfs,
        exp0_dfs=exp0_dfs,
        exp1_dfs=exp1_dfs,
        permutations=args.permutations,
        correction_params=correction_params,
        exclude_usb_derived_exp1=(not args.include_usb_derived_exp1),
    )


    raw_table = make_thesis_table(raw_stats, raw_desc, "raw")
    corr_table = make_thesis_table(corr_stats, corr_desc, "age_residualized")
    combined_table = pd.concat([raw_table, corr_table], ignore_index=True) if not raw_table.empty or not corr_table.empty else pd.DataFrame()

    raw_table.to_csv(args.output_dir / "thesis_table_raw.csv", index=False)
    corr_table.to_csv(args.output_dir / "thesis_table_age_residualized.csv", index=False)
    combined_table.to_csv(args.output_dir / "thesis_table_combined.csv", index=False)

    write_interpretations(
        raw_stats=raw_stats,
        raw_desc=raw_desc,
        corr_stats=corr_stats,
        corr_desc=corr_desc,
        raw_table=raw_table,
        corr_table=corr_table,
        output_path=args.output_dir / "interpretations.txt",
        min_age_exclusive=args.min_age_exclusive,
        exclude_usb_derived_exp1=(not args.include_usb_derived_exp1),
    )

    summary_lines: List[str] = []
    summary_lines.append("AUTOMATIC POSTPROCESSING COMPLETE")
    summary_lines.append("=================================")
    summary_lines.append("")
    summary_lines.append(f"Prediction root: {args.pred_root}")
    summary_lines.append(f"Healthy dir:     {healthy_dir}")
    summary_lines.append(f"Exp0 dir:        {exp0_dir}")
    summary_lines.append(f"Exp1 dir:        {exp1_dir}")
    summary_lines.append(f"Output dir:      {args.output_dir}")
    summary_lines.append(f"Permutations:    {args.permutations}")
    summary_lines.append(f"Age exclusion:   Age <= {args.min_age_exclusive:g} excluded")
    summary_lines.append(f"USB Exp1 QC:     {'included' if args.include_usb_derived_exp1 else 'excluded/skipped'}")
    summary_lines.append("")
    summary_lines.append(f"Healthy CSVs found: {len(healthy_paths)}")
    summary_lines.append(f"Exp0 CSVs found:    {len(exp0_paths)}")
    summary_lines.append(f"Exp1 CSVs found:    {len(exp1_paths)}")
    summary_lines.append("")
    summary_lines.append("Main output files:")
    summary_lines.append("  discovery_manifest.csv")
    summary_lines.append("  age_correction_parameters.csv")
    summary_lines.append("  all_raw_statistical_tests.csv")
    summary_lines.append("  all_raw_descriptive_summaries.csv")
    summary_lines.append("  all_age_residualized_statistical_tests.csv")
    summary_lines.append("  all_age_residualized_descriptive_summaries.csv")
    summary_lines.append("  all_raw_statistical_tests_summary.txt")
    summary_lines.append("  all_age_residualized_statistical_tests_summary.txt")
    summary_lines.append("  cohort_after_age_filter_summary.csv")
    summary_lines.append("  interpretations.txt")
    summary_lines.append("  thesis_table_raw.csv")
    summary_lines.append("  thesis_table_age_residualized.csv")
    summary_lines.append("  thesis_table_combined.csv")
    summary_lines.append("")
    summary_lines.append("Interpretation:")
    summary_lines.append("  Exp0 compares healthy vs synthetic tumor predictions.")
    summary_lines.append("  Exp1 compares healthy vs synthetic tumor vs inpainted predictions.")
    summary_lines.append("  Positive Exp1 Recovery means the inpainted prediction is closer to healthy than the synthetic tumor prediction.")
    summary_lines.append("  Raw BAG is the primary analysis; age-residualized BAG is the sensitivity analysis.")
    summary_lines.append("  Exportable thesis tables are written as CSV and include formatted values with significance stars.")
    summary_lines.append("")
    summary_lines.append(f"Raw statistical rows: {len(raw_stats)}")
    summary_lines.append(f"Raw descriptive rows: {len(raw_desc)}")
    summary_lines.append(f"Age-residualized statistical rows: {len(corr_stats)}")
    summary_lines.append(f"Age-residualized descriptive rows: {len(corr_desc)}")


    (args.output_dir / "run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


    print("\nDONE")
    print(f"Saved outputs to: {args.output_dir}")
    print(f"Manifest: {args.output_dir / 'discovery_manifest.csv'}")
    print(f"Raw stats: {args.output_dir / 'all_raw_statistical_tests.csv'}")
    print(f"Age-residualized stats: {args.output_dir / 'all_age_residualized_statistical_tests.csv'}")
    print(f"Run summary: {args.output_dir / 'run_summary.txt'}")
    print(f"Interpretations: {args.output_dir / 'interpretations.txt'}")
    print(f"Thesis table: {args.output_dir / 'thesis_table_combined.csv'}")




if __name__ == "__main__":
    main()






