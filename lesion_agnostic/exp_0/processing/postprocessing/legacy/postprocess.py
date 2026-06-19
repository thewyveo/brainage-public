#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations


import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D




PRED_CANDIDATES = [
    "Predicted_Brain_Age",
    "brainagenext_predictions_run_001_PBA",
]


BAD_CANDIDATES = [
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




def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None




def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def build_triplet_df(base_df: pd.DataFrame, synth_df: pd.DataFrame, inpaint_df: pd.DataFrame) -> pd.DataFrame:
    base = base_df.rename(columns={
        "Age": "Age_base",
        "Predicted_Brain_Age": "PBA_base",
        "Brain_Age_Difference": "BAD_base",
    })


    synth = synth_df.rename(columns={
        "Age": "Age_synth",
        "Predicted_Brain_Age": "PBA_synth",
        "Brain_Age_Difference": "BAD_synth",
    })


    inp = inpaint_df.rename(columns={
        "Age": "Age_inpaint",
        "Predicted_Brain_Age": "PBA_inpaint",
        "Brain_Age_Difference": "BAD_inpaint",
    })


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


    merged["Age"] = merged["Age_base"]


    merged["synth_distance_from_base"] = np.abs(merged["BAD_synth"] - merged["BAD_base"])
    merged["inpaint_distance_from_base"] = np.abs(merged["BAD_inpaint"] - merged["BAD_base"])


    merged["inpaint_closer_to_base"] = (
        merged["inpaint_distance_from_base"] < merged["synth_distance_from_base"]
    )


    return merged








def bag_triplet_shift_plot_closer_to_baseline(df: pd.DataFrame, out_path: Path) -> None:
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
    plt.xlabel("Subjects (ranked by age, low to high)")
    plt.ylabel("Brain Age Difference / BAG")
    plt.title("Per-subject BAG shift: healthy → synthetic tumor → inpainted")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def sanitize(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text)).strip("_")




def parse_named_path(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected LABEL=PATH, got: {raw}")
    label, path = raw.split("=", 1)
    return label.strip(), Path(path.strip())




def normalize_prediction_csv(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]


    age_col = first_existing(df, AGE_CANDIDATES)
    pred_col = first_existing(df, PRED_CANDIDATES)
    bad_col = first_existing(df, BAD_CANDIDATES)


    if "IXI_ID" not in df.columns:
        raise ValueError(f"{label}: missing IXI_ID column. Found: {df.columns.tolist()}")
    if age_col is None:
        raise ValueError(f"{label}: missing age column. Found: {df.columns.tolist()}")
    if pred_col is None:
        raise ValueError(f"{label}: missing predicted brain age column. Found: {df.columns.tolist()}")


    out = pd.DataFrame()
    out["IXI_ID"] = df["IXI_ID"].astype(str).str.strip()
    out["Age"] = pd.to_numeric(df[age_col], errors="coerce")
    out["Predicted_Brain_Age"] = pd.to_numeric(df[pred_col], errors="coerce")


    if bad_col is not None:
        out["Brain_Age_Difference"] = pd.to_numeric(df[bad_col], errors="coerce")
    else:
        out["Brain_Age_Difference"] = out["Predicted_Brain_Age"] - out["Age"]


    for c in ["Filename", "Path", "BraTS Subject ID", "Case_Folder"]:
        if c in df.columns:
            out[c] = df[c]


    out["Model"] = label
    out = out.dropna(subset=["IXI_ID", "Age", "Predicted_Brain_Age", "Brain_Age_Difference"])
    return out.reset_index(drop=True)




def compute_metrics(df: pd.DataFrame, model: str) -> Dict:
    age = df["Age"].to_numpy(float)
    pred = df["Predicted_Brain_Age"].to_numpy(float)
    bag = df["Brain_Age_Difference"].to_numpy(float)
    err = pred - age


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
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "Correlation_Age_vs_Pred": safe_corr(age, pred),
        "Correlation_Age_vs_BAG": safe_corr(age, bag),
    }




def save_scatter_age_vs_pred(df: pd.DataFrame, out_path: Path, title: str) -> None:
    x = df["Age"].to_numpy(float)
    y = df["Predicted_Brain_Age"].to_numpy(float)


    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.65)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Chronological Age")
    plt.ylabel("Predicted Brain Age")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def save_bag_hist(df: pd.DataFrame, out_path: Path, title: str) -> None:
    bag = df["Brain_Age_Difference"].to_numpy(float)


    plt.figure(figsize=(7, 5))
    plt.hist(bag, bins=30)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Brain Age Gap / Difference")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def build_paired_df(base_df: pd.DataFrame, synth_df: pd.DataFrame) -> pd.DataFrame:
    base = base_df.rename(columns={
        "Age": "Age_base",
        "Predicted_Brain_Age": "PBA_base",
        "Brain_Age_Difference": "BAD_base",
        "Filename": "Filename_base",
        "Path": "Path_base",
    })


    synth = synth_df.rename(columns={
        "Age": "Age_synth",
        "Predicted_Brain_Age": "PBA_synth",
        "Brain_Age_Difference": "BAD_synth",
        "Filename": "Filename_synth",
        "Path": "Path_synth",
    })


    base_cols = ["IXI_ID", "Age_base", "PBA_base", "BAD_base"]
    synth_cols = ["IXI_ID", "Age_synth", "PBA_synth", "BAD_synth"]


    for c in ["Filename_base", "Path_base"]:
        if c in base.columns:
            base_cols.append(c)
    for c in ["Filename_synth", "Path_synth"]:
        if c in synth.columns:
            synth_cols.append(c)


    merged = pd.merge(base[base_cols], synth[synth_cols], on="IXI_ID", how="inner")


    if merged.empty:
        raise RuntimeError("No overlapping IXI_IDs between baseline and synthetic set.")


    merged["Age"] = merged["Age_base"]
    merged["PBA_diff"] = merged["PBA_synth"] - merged["PBA_base"]
    merged["PBA_abs_diff"] = np.abs(merged["PBA_diff"])
    merged["BAD_diff"] = merged["BAD_synth"] - merged["BAD_base"]
    merged["BAD_abs_diff"] = np.abs(merged["BAD_diff"])
    merged["ABS_BAD_base"] = np.abs(merged["BAD_base"])
    merged["ABS_BAD_synth"] = np.abs(merged["BAD_synth"])
    merged["BASE_abs_err"] = np.abs(merged["PBA_base"] - merged["Age"])
    merged["SYNTH_abs_err"] = np.abs(merged["PBA_synth"] - merged["Age"])
    merged["ABS_ERR_diff"] = merged["SYNTH_abs_err"] - merged["BASE_abs_err"]


    return merged




def paired_shift_plot(df: pd.DataFrame, out_path: Path) -> None:
    d = df.sort_values(["Age", "IXI_ID"]).reset_index(drop=True)
    x = np.arange(len(d))


    plt.figure(figsize=(12, 6))
    for i, row in d.iterrows():
        plt.plot([i, i], [row["PBA_base"], row["PBA_synth"]], color="gray", alpha=0.25, linewidth=1)


    plt.scatter(x, d["PBA_base"], color="royalblue", label="Baseline", s=18)
    plt.scatter(x, d["PBA_synth"], color="crimson", label="Synthetic", s=18)
    plt.xlabel("Subjects (ranked by age, low to high)")
    plt.ylabel("Predicted Brain Age")
    plt.title("Per-subject prediction shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def bag_paired_shift_plot(df: pd.DataFrame, out_path: Path) -> None:
    d = df.sort_values(["Age", "IXI_ID"]).reset_index(drop=True)
    x = np.arange(len(d))


    plt.figure(figsize=(12, 6))
    for i, row in d.iterrows():
        plt.plot([i, i], [row["BAD_base"], row["BAD_synth"]], color="gray", alpha=0.25, linewidth=1)


    plt.scatter(x, d["BAD_base"], color="seagreen", label="Baseline BAG", s=18)
    plt.scatter(x, d["BAD_synth"], color="darkorange", label="Synthetic BAG", s=18)
    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Subjects (ranked by age, low to high)")
    plt.ylabel("Brain Age Difference")
    plt.title("Per-subject BAG shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def bag_paired_shift_plot_synthetic_higher_highlight(df: pd.DataFrame, out_path: Path) -> None:
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


    plt.scatter(x, d["BAD_base"], color="seagreen", label="Baseline BAG", s=18, zorder=2)


    mask_lo = ~synth_higher
    if mask_lo.any():
        plt.scatter(
            x[mask_lo],
            d.loc[mask_lo, "BAD_synth"],
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
            label="Synthetic BAG (synthetic > baseline)",
            s=26,
            alpha=0.95,
            edgecolors="#8b0000",
            linewidths=0.7,
            zorder=4,
        )


    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Subjects (ranked by age, low to high)")
    plt.ylabel("Brain Age Difference")
    plt.title("Per-subject BAG shift (synthetic > baseline highlighted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def bag_paired_shift_plot_baseline_higher_highlight(df: pd.DataFrame, out_path: Path) -> None:
    d = df.sort_values(["Age", "IXI_ID"]).reset_index(drop=True)
    x = np.arange(len(d))
    base_higher = d["BAD_base"].to_numpy(float) > d["BAD_synth"].to_numpy(float)


    plt.figure(figsize=(12, 6))


    for i, row in d.iterrows():
        hi = bool(base_higher[i])
        plt.plot(
            [i, i],
            [row["BAD_base"], row["BAD_synth"]],
            color="#2b6cb0" if hi else "gray",
            alpha=0.55 if hi else 0.22,
            linewidth=1.4 if hi else 1.0,
        )


    mask_lo = ~base_higher


    if mask_lo.any():
        plt.scatter(
            x[mask_lo],
            d.loc[mask_lo, "BAD_base"],
            color="seagreen",
            label="Baseline BAG",
            s=16,
            alpha=0.55,
            zorder=2,
        )


    if base_higher.any():
        plt.scatter(
            x[base_higher],
            d.loc[base_higher, "BAD_base"],
            color="seagreen",
            label="Baseline BAG (baseline > synthetic)",
            s=26,
            alpha=0.95,
            edgecolors="#0b2f4a",
            linewidths=0.7,
            zorder=4,
        )


    plt.scatter(x, d["BAD_synth"], color="darkorange", label="Synthetic BAG", s=18, zorder=3)


    plt.axhline(0, color="black", linewidth=1, linestyle="--")
    plt.xlabel("Subjects (ranked by age, low to high)")
    plt.ylabel("Brain Age Difference")
    plt.title("Per-subject BAG shift (baseline > synthetic highlighted)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def bag_diff_signed_line_unified(df: pd.DataFrame, out_path: Path) -> None:
    d = df.sort_values(["Age", "IXI_ID"]).reset_index(drop=True)


    x = np.arange(len(d), dtype=float)
    y = d["BAD_diff"].to_numpy(dtype=float)


    red, blue, gray = "#c0392b", "#2980b9", "#7f8c8d"


    def point_color(v: float) -> str:
        if v > 0:
            return red
        if v < 0:
            return blue
        return gray


    segments = []
    seg_colors = []


    for i in range(len(d) - 1):
        x0, x1 = x[i], x[i + 1]
        y0, y1 = y[i], y[i + 1]


        if y0 == 0 and y1 == 0:
            segments.append([(x0, y0), (x1, y1)])
            seg_colors.append(gray)
            continue


        if y0 * y1 < 0:
            t = y0 / (y0 - y1)
            xm = x0 + t * (x1 - x0)
            segments.append([(x0, y0), (xm, 0.0)])
            seg_colors.append(blue if y0 < 0 else red)
            segments.append([(xm, 0.0), (x1, y1)])
            seg_colors.append(blue if y1 < 0 else red)
            continue


        if y0 >= 0 and y1 >= 0:
            c = red if (y0 > 0 or y1 > 0) else gray
        elif y0 <= 0 and y1 <= 0:
            c = blue if (y0 < 0 or y1 < 0) else gray
        else:
            c = gray


        segments.append([(x0, y0), (x1, y1)])
        seg_colors.append(c)


    fig, ax = plt.subplots(figsize=(12, 6))


    if segments:
        lc = LineCollection(segments, colors=seg_colors, linewidths=1.4, alpha=0.95)
        ax.add_collection(lc)


    pt_colors = [point_color(v) for v in y]
    ax.scatter(x, y, c=pt_colors, s=22, zorder=3, edgecolors="white", linewidths=0.4)


    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.65)
    ax.set_xlim(-0.5, len(d) - 0.5)
    ax.autoscale_view(scalex=False, scaley=True)
    ax.set_xlabel("Subjects (ranked by age, low to high)")
    ax.set_ylabel("ΔBAG = synthetic BAG − baseline BAG")
    ax.set_title("Signed BAG difference")
    ax.grid(True, alpha=0.25)


    legend_elems = [
        Line2D([0], [0], color=red, lw=2, label="Above 0"),
        Line2D([0], [0], color=blue, lw=2, label="Below 0"),
        Line2D([0], [0], color=gray, lw=2, label="On 0"),
    ]


    ax.legend(handles=legend_elems, loc="upper right", fontsize=9)


    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def abs_pba_diff_per_subject_plot(df: pd.DataFrame, out_path: Path) -> None:
    d = df.sort_values(["Age", "IXI_ID"]).reset_index(drop=True)


    plt.figure(figsize=(12, 6))
    x = np.arange(len(d))
    plt.bar(x, d["PBA_abs_diff"].values, alpha=0.8)
    plt.xlabel("Subjects (sorted by age, low to high)")
    plt.ylabel("|Synthetic - Baseline predicted brain age|")
    plt.title("Per-subject absolute prediction difference")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def abs_bad_diff_per_subject_plot(df: pd.DataFrame, out_path: Path) -> None:
    d = df.sort_values(["Age", "IXI_ID"]).reset_index(drop=True)


    plt.figure(figsize=(12, 6))
    x = np.arange(len(d))
    plt.bar(x, d["BAD_abs_diff"].values, alpha=0.8)
    plt.xlabel("Subjects (sorted by age, low to high)")
    plt.ylabel("|Synthetic - Baseline BAG|")
    plt.title("Per-subject absolute BAG difference")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def direct_comparison_scatter(df: pd.DataFrame, out_path: Path) -> None:
    minv = float(min(df["PBA_base"].min(), df["PBA_synth"].min()))
    maxv = float(max(df["PBA_base"].max(), df["PBA_synth"].max()))


    plt.figure(figsize=(7, 7))
    plt.scatter(df["PBA_base"], df["PBA_synth"], alpha=0.75, s=24)
    plt.plot([minv, maxv], [minv, maxv], color="black", linewidth=1.2)
    plt.xlabel("Baseline predicted brain age")
    plt.ylabel("Synthetic predicted brain age")
    plt.title("Direct prediction comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def bag_direct_comparison_scatter(df: pd.DataFrame, out_path: Path) -> None:
    minv = float(min(df["BAD_base"].min(), df["BAD_synth"].min()))
    maxv = float(max(df["BAD_base"].max(), df["BAD_synth"].max()))


    plt.figure(figsize=(7, 7))
    plt.scatter(df["BAD_base"], df["BAD_synth"], alpha=0.75, s=24)
    plt.plot([minv, maxv], [minv, maxv], color="black", linewidth=1.2)
    plt.axhline(0, color="gray", linewidth=1, linestyle="--")
    plt.axvline(0, color="gray", linewidth=1, linestyle="--")
    plt.xlabel("Baseline BAG")
    plt.ylabel("Synthetic BAG")
    plt.title("Direct BAG comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def abs_bag_direct_comparison_scatter(df: pd.DataFrame, out_path: Path) -> None:
    base = np.abs(df["BAD_base"].values)
    synth = np.abs(df["BAD_synth"].values)
    minv = float(min(base.min(), synth.min()))
    maxv = float(max(base.max(), synth.max()))


    plt.figure(figsize=(7, 7))
    plt.scatter(base, synth, alpha=0.75, s=24)
    plt.plot([minv, maxv], [minv, maxv], color="black", linewidth=1.2)
    plt.xlabel("|Baseline BAG|")
    plt.ylabel("|Synthetic BAG|")
    plt.title("Direct absolute BAG comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def overlaid_prediction_histogram(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(df["PBA_base"], bins=20, alpha=0.6, label="Baseline")
    plt.hist(df["PBA_synth"], bins=20, alpha=0.6, label="Synthetic")
    plt.xlabel("Predicted Brain Age")
    plt.ylabel("Count")
    plt.title("Distribution of predicted brain age")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def side_by_side_boxplots(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.boxplot(
        [df["PBA_base"].values, df["PBA_synth"].values],
        labels=["Baseline", "Synthetic"],
        patch_artist=True,
    )
    plt.ylabel("Predicted Brain Age")
    plt.title("Predicted brain age: baseline vs synthetic")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def bag_distribution_histogram(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(df["BAD_base"], bins=20, alpha=0.6, label="Baseline BAG")
    plt.hist(df["BAD_synth"], bins=20, alpha=0.6, label="Synthetic BAG")
    plt.xlabel("Brain Age Difference")
    plt.ylabel("Count")
    plt.title("Distribution of brain age difference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def abs_bag_distribution_histogram(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(np.abs(df["BAD_base"].values), bins=20, alpha=0.6, label="|Baseline BAG|")
    plt.hist(np.abs(df["BAD_synth"].values), bins=20, alpha=0.6, label="|Synthetic BAG|")
    plt.xlabel("|Brain Age Difference|")
    plt.ylabel("Count")
    plt.title("Distribution of absolute brain age difference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def age_vs_prediction_overlay(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    plt.scatter(df["Age"], df["PBA_base"], alpha=0.7, label="Baseline", s=22)
    plt.scatter(df["Age"], df["PBA_synth"], alpha=0.7, label="Synthetic", s=22)
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




def abs_error_vs_age_plot(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    plt.scatter(df["Age"], np.abs(df["BAD_base"].values), alpha=0.7, label="Baseline |BAG|", s=22)
    plt.scatter(df["Age"], np.abs(df["BAD_synth"].values), alpha=0.7, label="Synthetic |BAG|", s=22)
    plt.xlabel("Chronological Age")
    plt.ylabel("|BAG| = absolute age prediction error")
    plt.title("Absolute age prediction error vs chronological age")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def difference_histogram(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(df["PBA_diff"], bins=20, alpha=0.8)
    plt.axvline(0, color="black", linewidth=1)
    plt.axvline(df["PBA_diff"].mean(), color="red", linewidth=1.2, linestyle="--")
    plt.xlabel("Synthetic - Baseline predicted brain age")
    plt.ylabel("Count")
    plt.title("Distribution of prediction differences")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def abs_difference_histogram(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(df["PBA_abs_diff"], bins=20, alpha=0.8)
    plt.axvline(df["PBA_abs_diff"].mean(), color="red", linewidth=1.2, linestyle="--")
    plt.xlabel("|Synthetic - Baseline predicted brain age|")
    plt.ylabel("Count")
    plt.title("Distribution of absolute prediction differences")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def abs_bad_difference_histogram(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(df["BAD_abs_diff"], bins=20, alpha=0.8)
    plt.axvline(df["BAD_abs_diff"].mean(), color="red", linewidth=1.2, linestyle="--")
    plt.xlabel("|Synthetic - Baseline BAG|")
    plt.ylabel("Count")
    plt.title("Distribution of absolute BAG differences")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def combined_distribution_panel(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))


    axes[0].hist(df["PBA_base"], bins=20, alpha=0.6, label="Baseline")
    axes[0].hist(df["PBA_synth"], bins=20, alpha=0.6, label="Synthetic")
    axes[0].set_xlabel("Predicted Brain Age")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Predicted brain age")
    axes[0].legend()


    axes[1].hist(df["BAD_base"], bins=20, alpha=0.6, label="Baseline BAG")
    axes[1].hist(df["BAD_synth"], bins=20, alpha=0.6, label="Synthetic BAG")
    axes[1].set_xlabel("Brain Age Difference")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Brain age difference")
    axes[1].legend()


    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def combined_absolute_panel(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))


    axes[0].hist(df["PBA_abs_diff"], bins=20, alpha=0.8)
    axes[0].set_xlabel("|Synthetic - Baseline predicted brain age|")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Absolute prediction difference")


    axes[1].hist(df["BAD_abs_diff"], bins=20, alpha=0.8)
    axes[1].set_xlabel("|Synthetic - Baseline BAG|")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Absolute BAG difference")


    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def combined_scatter_panel(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))


    minv = float(min(df["PBA_base"].min(), df["PBA_synth"].min()))
    maxv = float(max(df["PBA_base"].max(), df["PBA_synth"].max()))


    axes[0].scatter(df["PBA_base"], df["PBA_synth"], alpha=0.75, s=24)
    axes[0].plot([minv, maxv], [minv, maxv], color="black", linewidth=1.2)
    axes[0].set_xlabel("Baseline predicted brain age")
    axes[0].set_ylabel("Synthetic predicted brain age")
    axes[0].set_title("Direct comparison")


    axes[1].scatter(df["Age"], df["PBA_base"], alpha=0.7, label="Baseline", s=22)
    axes[1].scatter(df["Age"], df["PBA_synth"], alpha=0.7, label="Synthetic", s=22)
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




def bland_altman_plot(
    baseline: pd.Series,
    synthetic: pd.Series,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    baseline = baseline.to_numpy(dtype=float)
    synthetic = synthetic.to_numpy(dtype=float)


    mean_vals = (baseline + synthetic) / 2.0
    diff_vals = synthetic - baseline


    mean_diff = float(np.mean(diff_vals))
    std_diff = float(np.std(diff_vals, ddof=1)) if len(diff_vals) > 1 else 0.0
    upper = mean_diff + 1.96 * std_diff
    lower = mean_diff - 1.96 * std_diff


    plt.figure(figsize=(8, 6))
    plt.scatter(mean_vals, diff_vals, alpha=0.75, s=24)
    plt.axhline(mean_diff, color="red", linestyle="--", linewidth=1.2, label=f"Mean diff = {mean_diff:.2f}")
    plt.axhline(upper, color="gray", linestyle="--", linewidth=1.0, label=f"+1.96 SD = {upper:.2f}")
    plt.axhline(lower, color="gray", linestyle="--", linewidth=1.0, label=f"-1.96 SD = {lower:.2f}")
    plt.axhline(0, color="black", linewidth=1.0)
    plt.xlabel("Mean of Baseline and Synthetic")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def shift_vs_baseline_plot(
    baseline: pd.Series,
    diff: pd.Series,
    baseline_label: str,
    diff_label: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(baseline.to_numpy(float), diff.to_numpy(float), alpha=0.75, s=24)
    plt.axhline(0, color="black", linewidth=1.0)
    plt.xlabel(baseline_label)
    plt.ylabel(diff_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def abs_shift_vs_abs_baseline_plot(
    baseline_abs: pd.Series,
    diff_abs: pd.Series,
    baseline_label: str,
    diff_label: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(baseline_abs.to_numpy(float), diff_abs.to_numpy(float), alpha=0.75, s=24)
    plt.xlabel(baseline_label)
    plt.ylabel(diff_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def mean_ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())


    if n < 2:
        return mean, mean


    se = stats.sem(values, nan_policy="omit")
    if not np.isfinite(se):
        return mean, mean


    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return float(mean - tcrit * se), float(mean + tcrit * se)




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


    return float((w_pos - w_neg) / (n * (n + 1) / 2.0))




def sign_flip_permutation_test(diff: np.ndarray, n_perm: int, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)


    observed = float(np.mean(diff))


    if len(diff) == 0:
        return observed, 1.0


    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    perm_means = np.mean(signs * diff[None, :], axis=1)
    p = (np.sum(np.abs(perm_means) >= abs(observed)) + 1) / (n_perm + 1)


    return observed, float(p)




def run_signed_tests(diff: np.ndarray, label: str, permutations: int) -> List[str]:
    diff = np.asarray(diff, dtype=float)
    n = len(diff)


    lines = []
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


    if n >= 2:
        t_stat, t_p = stats.ttest_1samp(diff, popmean=0.0)
        lines.append(f"  one-sample t statistic      = {t_stat:.6f}")
        lines.append(f"  one-sample t p-value        = {t_p:.6g}")


    try:
        w_stat, w_p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        lines.append(f"  Wilcoxon statistic          = {w_stat:.6f}")
        lines.append(f"  Wilcoxon p-value            = {w_p:.6g}")
    except ValueError as e:
        lines.append(f"  Wilcoxon                    = unavailable ({e})")


    lines.append(f"  Cohen's d paired            = {cohens_d_paired(diff):.6f}")


    try:
        lines.append(f"  rank-biserial correlation   = {rank_biserial_from_wilcoxon(diff):.6f}")
    except Exception as e:
        lines.append(f"  rank-biserial correlation   = unavailable ({e})")


    perm_mean, perm_p = sign_flip_permutation_test(diff, n_perm=permutations)
    lines.append(f"  permutation mean            = {perm_mean:.6f}")
    lines.append(f"  permutation p-value         = {perm_p:.6g}")
    lines.append("")


    return lines




def write_paired_report(df: pd.DataFrame, out_path: Path, permutations: int) -> None:
    lines = []
    lines.append("PAIRED STATISTICAL TESTS")
    lines.append("========================")
    lines.append(f"Subjects: {len(df)}")
    lines.append("")


    lines.extend(run_signed_tests(df["PBA_diff"].values, "Signed PBA shift: PBA_synth - PBA_base", permutations))
    lines.extend(run_signed_tests(df["BAD_diff"].values, "Signed BAG shift: BAG_synth - BAG_base", permutations))
    lines.extend(run_signed_tests(df["PBA_abs_diff"].values, "Absolute PBA difference", permutations))
    lines.extend(run_signed_tests(df["BAD_abs_diff"].values, "Absolute BAG difference", permutations))
    lines.extend(run_signed_tests(df["ABS_ERR_diff"].values, "Absolute error change: synth - baseline", permutations))


    out_path.write_text("\n".join(lines), encoding="utf-8")




def write_paired_summary(df: pd.DataFrame, out_path: Path) -> None:
    diff = df["PBA_diff"].values
    abs_diff = df["PBA_abs_diff"].values
    bad_diff = df["BAD_diff"].values
    bad_abs_diff = df["BAD_abs_diff"].values


    corr = safe_corr(df["PBA_base"].values, df["PBA_synth"].values)
    bad_corr = safe_corr(df["BAD_base"].values, df["BAD_synth"].values)


    baseline_mae_vs_age = float(np.mean(np.abs(df["PBA_base"] - df["Age"])))
    synthetic_mae_vs_age = float(np.mean(np.abs(df["PBA_synth"] - df["Age"])))


    baseline_mean_abs_bag = float(np.mean(np.abs(df["BAD_base"].values)))
    synthetic_mean_abs_bag = float(np.mean(np.abs(df["BAD_synth"].values)))


    lines = []
    lines.append(f"Subjects compared: {len(df)}")
    lines.append("")
    lines.append("Direct comparison between CSVs")
    lines.append(f"  MAE between predictions        = {np.mean(abs_diff):.6f}")
    lines.append(f"  RMSE between predictions       = {np.sqrt(np.mean(diff ** 2)):.6f}")
    lines.append(f"  Mean signed difference         = {np.mean(diff):.6f}")
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
    lines.append("Brain Age Difference shift")
    lines.append(f"  Mean ΔBAD                      = {np.mean(bad_diff):.6f}")
    lines.append(f"  Median ΔBAD                    = {np.median(bad_diff):.6f}")
    lines.append(f"  Std ΔBAD                       = {np.std(bad_diff, ddof=1):.6f}")
    lines.append(f"  Correlation BAD                = {bad_corr:.6f}")
    lines.append("")
    lines.append("Absolute Brain Age Difference")
    lines.append(f"  Mean |ΔBAD|                    = {np.mean(bad_abs_diff):.6f}")
    lines.append(f"  Median |ΔBAD|                  = {np.median(bad_abs_diff):.6f}")
    lines.append(f"  Max |ΔBAD|                     = {np.max(bad_abs_diff):.6f}")
    lines.append("")
    lines.append("Against chronological age")
    lines.append(f"  Baseline MAE vs Age            = {baseline_mae_vs_age:.6f}")
    lines.append(f"  Synthetic MAE vs Age           = {synthetic_mae_vs_age:.6f}")
    lines.append(f"  ΔMAE Synthetic - Baseline      = {synthetic_mae_vs_age - baseline_mae_vs_age:.6f}")
    lines.append(f"  Mean |Baseline BAG|            = {baseline_mean_abs_bag:.6f}")
    lines.append(f"  Mean |Synthetic BAG|           = {synthetic_mean_abs_bag:.6f}")
    lines.append("")
    lines.append("Counts")
    lines.append(f"  Synthetic > Baseline           = {(diff > 0).sum()}")
    lines.append(f"  Synthetic < Baseline           = {(diff < 0).sum()}")
    lines.append(f"  Synthetic = Baseline           = {(diff == 0).sum()}")


    out_path.write_text("\n".join(lines), encoding="utf-8")




def write_paired_summary_csv(df: pd.DataFrame, out_path: Path) -> None:
    summary = pd.DataFrame({
        "metric": [
            "n",
            "mean_PBA_diff",
            "median_PBA_diff",
            "mean_BAD_diff",
            "median_BAD_diff",
            "mean_abs_PBA_diff",
            "mean_abs_BAD_diff",
            "mean_abs_BAG_baseline",
            "mean_abs_BAG_synthetic",
            "baseline_MAE_vs_age",
            "synthetic_MAE_vs_age",
            "delta_MAE_synthetic_minus_baseline",
            "count_synth_gt_base",
            "count_synth_lt_base",
            "count_synth_eq_base",
        ],
        "value": [
            len(df),
            float(df["PBA_diff"].mean()),
            float(df["PBA_diff"].median()),
            float(df["BAD_diff"].mean()),
            float(df["BAD_diff"].median()),
            float(df["PBA_abs_diff"].mean()),
            float(df["BAD_abs_diff"].mean()),
            float(df["ABS_BAD_base"].mean()),
            float(df["ABS_BAD_synth"].mean()),
            float(df["BASE_abs_err"].mean()),
            float(df["SYNTH_abs_err"].mean()),
            float(df["SYNTH_abs_err"].mean() - df["BASE_abs_err"].mean()),
            int((df["PBA_diff"] > 0).sum()),
            int((df["PBA_diff"] < 0).sum()),
            int((df["PBA_diff"] == 0).sum()),
        ],
    })


    summary.to_csv(out_path, index=False)




def save_age_bin_analysis(df: pd.DataFrame, out_dir: Path, bin_width: int, min_bin_size: int) -> None:
    d = df.copy()


    min_age = int(np.floor(d["Age"].min() / bin_width) * bin_width)
    max_age = int(np.ceil(d["Age"].max() / bin_width) * bin_width)


    bins = np.arange(min_age, max_age + bin_width, bin_width)
    d["Age_Bin"] = pd.cut(d["Age"], bins=bins, include_lowest=True)


    rows = []


    for age_bin, g in d.groupby("Age_Bin", observed=False):
        if len(g) < min_bin_size:
            continue


        rows.append({
            "Age_Bin": str(age_bin),
            "N": len(g),
            "Mean_Age": float(g["Age"].mean()),
            "Mean_PBA_Diff": float(g["PBA_diff"].mean()),
            "Median_PBA_Diff": float(g["PBA_diff"].median()),
            "Mean_BAD_Diff": float(g["BAD_diff"].mean()),
            "Mean_ABS_PBA_Diff": float(g["PBA_abs_diff"].mean()),
            "Mean_ABS_BAD_Diff": float(g["BAD_abs_diff"].mean()),
            "Baseline_MAE": float(g["BASE_abs_err"].mean()),
            "Synthetic_MAE": float(g["SYNTH_abs_err"].mean()),
            "Delta_MAE": float(g["ABS_ERR_diff"].mean()),
        })


    if not rows:
        return


    bin_df = pd.DataFrame(rows)
    bin_df.to_csv(out_dir / "age_bin_analysis.csv", index=False)


    plt.figure(figsize=(10, 5))
    plt.plot(bin_df["Age_Bin"], bin_df["Mean_PBA_Diff"], marker="o")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Age bin")
    plt.ylabel("Mean PBA shift")
    plt.title("Mean prediction shift by age bin")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "age_bin_mean_pba_shift.png", dpi=200)
    plt.close()


    plt.figure(figsize=(10, 5))
    plt.plot(bin_df["Age_Bin"], bin_df["Delta_MAE"], marker="o")
    plt.axhline(0, linestyle="--")
    plt.xlabel("Age bin")
    plt.ylabel("ΔMAE synthetic - baseline")
    plt.title("MAE change by age bin")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "age_bin_delta_mae.png", dpi=200)
    plt.close()




def save_metric_barplot(summary_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    plt.figure(figsize=(max(8, 1.2 * len(summary_df)), 5))
    plt.bar(summary_df["Model"], summary_df[metric])
    plt.ylabel(metric)
    plt.title(f"{metric} by set")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def save_heatmap(summary_df: pd.DataFrame, metric: str, out_path: Path) -> None:
    pivot = summary_df.set_index("Model")[[metric]]
    arr = pivot.to_numpy(float)


    plt.figure(figsize=(6, max(4, 0.45 * len(pivot) + 2)))
    im = plt.imshow(arr, aspect="auto")
    plt.colorbar(im, label=metric)
    plt.xticks([0], [metric])
    plt.yticks(np.arange(len(pivot)), pivot.index)


    for i in range(arr.shape[0]):
        plt.text(0, i, f"{arr[i, 0]:.2f}", ha="center", va="center")


    plt.title(f"{metric} heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def save_global_scatter(all_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7, 7))


    all_vals = []


    for model, g in all_df.groupby("Model"):
        x = g["Age"].to_numpy(float)
        y = g["Predicted_Brain_Age"].to_numpy(float)


        all_vals.extend([x, y])
        plt.scatter(x, y, alpha=0.35, label=model)


    concat = np.concatenate(all_vals)
    lo = float(concat.min())
    hi = float(concat.max())


    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Chronological Age")
    plt.ylabel("Predicted Brain Age")
    plt.title("All sets: Age vs Predicted Brain Age")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def save_global_bag_hist(all_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))


    for model, g in all_df.groupby("Model"):
        plt.hist(g["Brain_Age_Difference"].to_numpy(float), bins=30, alpha=0.45, label=model)


    plt.axvline(0, linestyle="--")
    plt.xlabel("BAG")
    plt.ylabel("Count")
    plt.title("All sets: BAG distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def save_global_bag_boxplot(all_df: pd.DataFrame, out_path: Path) -> None:
    labels = []
    values = []


    for model, g in all_df.groupby("Model"):
        labels.append(model)
        values.append(g["Brain_Age_Difference"].to_numpy(float))


    plt.figure(figsize=(max(8, 1.1 * len(labels)), 5))
    plt.boxplot(values, labels=labels, showfliers=False)
    plt.axhline(0, linestyle="--")
    plt.ylabel("BAG")
    plt.title("BAG by set")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()




def run_per_set_outputs(label: str, df: pd.DataFrame, out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)


    df.to_csv(out_dir / "normalized_predictions.csv", index=False)


    metrics = compute_metrics(df, label)
    pd.DataFrame([metrics]).to_csv(out_dir / "model_summary.csv", index=False)


    save_scatter_age_vs_pred(df, out_dir / "scatter_age_vs_pred.png", f"{label}: Age vs predicted brain age")
    save_bag_hist(df, out_dir / "bag_hist.png", f"{label}: BAG distribution")


    return metrics




def run_paired_outputs(
    baseline_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    out_dir: Path,
    permutations: int,
    age_bin_width: int,
    min_bin_size: int,
) -> None:
    paired = build_paired_df(baseline_df, synth_df)


    paired.to_csv(out_dir / "paired_comparison.csv", index=False)


    paired_shift_plot(paired, out_dir / "per_subject_shift.png")
    bag_paired_shift_plot(paired, out_dir / "bag_per_subject_shift.png")
    bag_paired_shift_plot_synthetic_higher_highlight(paired, out_dir / "bag_per_subject_shift_synthetic_higher_highlight.png")
    bag_paired_shift_plot_baseline_higher_highlight(paired, out_dir / "bag_per_subject_shift_baseline_higher_highlight.png")
    bag_diff_signed_line_unified(paired, out_dir / "bag_difference_signed_line_unified.png")


    abs_pba_diff_per_subject_plot(paired, out_dir / "abs_prediction_diff_per_subject.png")
    abs_bad_diff_per_subject_plot(paired, out_dir / "abs_bag_diff_per_subject.png")


    direct_comparison_scatter(paired, out_dir / "direct_prediction_comparison.png")
    bag_direct_comparison_scatter(paired, out_dir / "direct_bag_comparison.png")
    abs_bag_direct_comparison_scatter(paired, out_dir / "direct_abs_bag_comparison.png")


    overlaid_prediction_histogram(paired, out_dir / "prediction_distribution_overlay.png")
    side_by_side_boxplots(paired, out_dir / "prediction_boxplots.png")
    bag_distribution_histogram(paired, out_dir / "bag_distribution_overlay.png")
    abs_bag_distribution_histogram(paired, out_dir / "abs_bag_distribution_overlay.png")


    age_vs_prediction_overlay(paired, out_dir / "age_vs_prediction_overlay.png")
    abs_error_vs_age_plot(paired, out_dir / "abs_error_vs_age.png")


    difference_histogram(paired, out_dir / "prediction_difference_histogram.png")
    abs_difference_histogram(paired, out_dir / "absolute_prediction_difference_histogram.png")
    abs_bad_difference_histogram(paired, out_dir / "absolute_bag_difference_histogram.png")


    combined_distribution_panel(paired, out_dir / "combined_distribution_panel.png")
    combined_absolute_panel(paired, out_dir / "combined_absolute_panel.png")
    combined_scatter_panel(paired, out_dir / "combined_scatter_panel.png")


    bland_altman_plot(
        paired["PBA_base"],
        paired["PBA_synth"],
        ylabel="Synthetic - Baseline predicted brain age",
        title="Bland-Altman: predicted brain age",
        out_path=out_dir / "bland_altman_prediction.png",
    )


    bland_altman_plot(
        paired["BAD_base"],
        paired["BAD_synth"],
        ylabel="Synthetic - Baseline BAG",
        title="Bland-Altman: BAG",
        out_path=out_dir / "bland_altman_bag.png",
    )


    shift_vs_baseline_plot(
        paired["PBA_base"],
        paired["PBA_diff"],
        baseline_label="Baseline predicted brain age",
        diff_label="Synthetic - Baseline predicted brain age",
        title="Prediction shift vs baseline prediction",
        out_path=out_dir / "prediction_shift_vs_baseline.png",
    )


    shift_vs_baseline_plot(
        paired["BAD_base"],
        paired["BAD_diff"],
        baseline_label="Baseline BAG",
        diff_label="Synthetic - Baseline BAG",
        title="BAG shift vs baseline BAG",
        out_path=out_dir / "bag_shift_vs_baseline.png",
    )


    abs_shift_vs_abs_baseline_plot(
        paired["ABS_BAD_base"],
        paired["BAD_abs_diff"],
        baseline_label="|Baseline BAG|",
        diff_label="|Synthetic - Baseline BAG|",
        title="Absolute BAG shift vs baseline absolute BAG",
        out_path=out_dir / "abs_bag_shift_vs_abs_baseline.png",
    )


    write_paired_summary(paired, out_dir / "summary.txt")
    write_paired_report(paired, out_dir / "paired_stats_report.txt", permutations)
    write_paired_summary_csv(paired, out_dir / "paired_stats_table.csv")


    save_age_bin_analysis(paired, out_dir, age_bin_width, min_bin_size)




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()


    parser.add_argument("--baseline", required=True, help="Format: LABEL=CSV")
    parser.add_argument("--set", action="append", required=True, help="Format: LABEL=CSV. Can be repeated.")
    parser.add_argument("--output-dir", type=Path, required=True)


    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--age-bin-width", type=int, default=5)
    parser.add_argument("--min-bin-size", type=int, default=10)


    return parser.parse_args()




def main() -> None:
    args = parse_args()


    args.output_dir.mkdir(parents=True, exist_ok=True)


    baseline_label, baseline_path = parse_named_path(args.baseline)
    set_specs = [parse_named_path(x) for x in args.set]


    print(f"Loading baseline: {baseline_label} -> {baseline_path}")


    baseline_df = normalize_prediction_csv(baseline_path, baseline_label)


    all_dfs = []
    all_metrics = []


    base_out = args.output_dir / sanitize(baseline_label)
    base_metrics = run_per_set_outputs(baseline_label, baseline_df, base_out)


    all_dfs.append(baseline_df)
    all_metrics.append(base_metrics)


    for label, path in set_specs:
        print(f"Processing set: {label} -> {path}")


        df = normalize_prediction_csv(path, label)
        out_dir = args.output_dir / sanitize(label)


        metrics = run_per_set_outputs(label, df, out_dir)


        run_paired_outputs(
            baseline_df=baseline_df,
            synth_df=df,
            out_dir=out_dir,
            permutations=args.permutations,
            age_bin_width=args.age_bin_width,
            min_bin_size=args.min_bin_size,
        )


        all_dfs.append(df)
        all_metrics.append(metrics)


    all_df = pd.concat(all_dfs, ignore_index=True)
    summary_df = pd.DataFrame(all_metrics)


    all_df.to_csv(args.output_dir / "all_sets_combined_predictions_normalized.csv", index=False)
    summary_df.to_csv(args.output_dir / "global_model_summary.csv", index=False)


    for metric in ["MAE", "RMSE", "BAG_Mean", "BAG_STD", "Mean_Absolute_BAG"]:
        save_metric_barplot(summary_df, metric, args.output_dir / f"global_barplot_{metric}.png")
        save_heatmap(summary_df, metric, args.output_dir / f"global_heatmap_{metric}.png")


    save_global_scatter(all_df, args.output_dir / "global_scatter_age_vs_pred.png")
    save_global_bag_hist(all_df, args.output_dir / "global_bag_hist.png")
    save_global_bag_boxplot(all_df, args.output_dir / "global_bag_boxplot.png")


    print("\nDONE")
    print(f"Saved outputs to: {args.output_dir}")


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




"""
RUN WITH:

py -3.10 .\postprocess.py `
  --baseline "IXI_BNX=data\IXI_BNX.csv" `
  --set "BNX_GLI=data\BNX_GLI.csv" `
  --set "BNX_GLI_LIT=data\BNX_GLI_LIT.csv" `
  --set "BNX_GLI_BID=data\BNX_GLI_BID_FILTERED.csv" `
  --set "BNX_GLI_USB=data\BNX_GLI_USB_FILTERED.csv" `
  --output-dir "outputcmb" `
  --permutations 10000 `
  --age-bin-width 5 `
  --min-bin-size 10
"""