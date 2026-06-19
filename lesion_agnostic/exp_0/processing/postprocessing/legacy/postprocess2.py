#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
from pathlib import Path
from typing import Optional, List


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

r"""
py -3.10 .\postprocess2.py `
  --healthy "data\IXI_BNX.csv" `
  --tumor "data\BNX_GLI.csv" `
  --inpaint "data\BNX_GLI_LIT.csv" `
  --output-dir "output\BNX_GLI_LIT" `
  --healthy-label "IXI Healthy T1" `
  --tumor-label "GLI Tumored T1" `
  --inpaint-label "LIT inpainted T1"
"""


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




def load_prediction_csv(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]


    if "IXI_ID" not in df.columns:
        raise ValueError(f"{label}: missing IXI_ID column. Found: {df.columns.tolist()}")


    age_col = first_existing(df, AGE_CANDIDATES)
    bag_col = first_existing(df, BAD_CANDIDATES)


    if age_col is None:
        raise ValueError(f"{label}: missing age column. Found: {df.columns.tolist()}")


    if bag_col is None:
        raise ValueError(f"{label}: missing BAG/BAD column. Found: {df.columns.tolist()}")


    out = pd.DataFrame()
    out["IXI_ID"] = df["IXI_ID"].astype(str).str.strip()
    out["Age"] = pd.to_numeric(df[age_col], errors="coerce")
    out["BAG"] = pd.to_numeric(df[bag_col], errors="coerce")


    return out.dropna(subset=["IXI_ID", "Age", "BAG"]).reset_index(drop=True)




def build_df(healthy_df, tumor_df, inpaint_df):
    h = healthy_df.rename(columns={"Age": "Age_healthy", "BAG": "BAG_healthy"})
    t = tumor_df.rename(columns={"Age": "Age_tumor", "BAG": "BAG_tumor"})
    i = inpaint_df.rename(columns={"Age": "Age_inpaint", "BAG": "BAG_inpaint"})


    df = h[["IXI_ID", "Age_healthy", "BAG_healthy"]].merge(
        t[["IXI_ID", "Age_tumor", "BAG_tumor"]],
        on="IXI_ID",
        how="inner",
    ).merge(
        i[["IXI_ID", "Age_inpaint", "BAG_inpaint"]],
        on="IXI_ID",
        how="inner",
    )


    if df.empty:
        raise RuntimeError("No overlapping IXI_IDs across all three CSV files.")


    df["Age"] = df["Age_healthy"]


    # Main factual test:
    # positive = inpainted BAG is higher than tumored BAG
    # negative = inpainted BAG is lower than tumored BAG
    df["Inpaint_minus_Tumor"] = df["BAG_inpaint"] - df["BAG_tumor"]


    # Healthy is kept only as reference, not as the decision/comparison.
    df["Tumor_minus_Healthy"] = df["BAG_tumor"] - df["BAG_healthy"]
    df["Inpaint_minus_Healthy"] = df["BAG_inpaint"] - df["BAG_healthy"]


    return df.sort_values(["Age", "IXI_ID"]).reset_index(drop=True)




def shift_color(v: float) -> str:
    if v > 0:
        return "#d62728"
    if v < 0:
        return "#2ca02c"
    return "#7f7f7f"




def plot_gli_vs_inpaint_shift(
    df,
    out_path,
    healthy_label,
    tumor_label,
    inpaint_label,
    title,
    show_subject_ticks,
    dpi,
):
    x = np.arange(len(df))


    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 9),
        gridspec_kw={"height_ratios": [3.2, 1.2]},
        sharex=True,
    )


    ax = axes[0]
    ax_bar = axes[1]


    for idx, row in df.iterrows():
        shift = row["Inpaint_minus_Tumor"]
        color = shift_color(shift)


        # faint healthy reference connection
        ax.plot(
            [idx, idx],
            [row["BAG_healthy"], row["BAG_tumor"]],
            color="gray",
            alpha=0.18,
            linewidth=1.0,
            zorder=1,
        )


        # main comparison: tumor -> inpaint
        ax.plot(
            [idx, idx],
            [row["BAG_tumor"], row["BAG_inpaint"]],
            color=color,
            alpha=0.95,
            linewidth=2.6,
            zorder=2,
        )


    ax.scatter(
        x,
        df["BAG_healthy"],
        color="black",
        marker="o",
        s=22,
        alpha=0.75,
        label=f"{healthy_label} reference",
        zorder=5,
    )


    ax.scatter(
        x,
        df["BAG_tumor"],
        color="white",
        edgecolors="black",
        linewidths=0.9,
        marker="s",
        s=36,
        label=tumor_label,
        zorder=6,
    )


    ax.scatter(
        x,
        df["BAG_inpaint"],
        color=[shift_color(v) for v in df["Inpaint_minus_Tumor"]],
        edgecolors="black",
        linewidths=0.5,
        marker="^",
        s=46,
        label=inpaint_label,
        zorder=7,
    )


    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.65)
    ax.set_ylabel("Brain Age Gap / Difference")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


    legend_items = [
        Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=6, label=f"{healthy_label} reference"),
        Line2D([0], [0], marker="s", color="black", markerfacecolor="white", linestyle="None", markersize=7, label=tumor_label),
        Line2D([0], [0], marker="^", color="black", markerfacecolor="#2ca02c", linestyle="None", markersize=7, label=inpaint_label),
        Line2D([0], [0], color="#2ca02c", lw=2.5, label=f"{inpaint_label} BAG < {tumor_label} BAG"),
        Line2D([0], [0], color="#d62728", lw=2.5, label=f"{inpaint_label} BAG > {tumor_label} BAG"),
        Line2D([0], [0], color="#7f7f7f", lw=2.5, label=f"{inpaint_label} BAG = {tumor_label} BAG"),
    ]


    ax.legend(handles=legend_items, loc="best", fontsize=9)


    bar_colors = [shift_color(v) for v in df["Inpaint_minus_Tumor"]]
    ax_bar.bar(x, df["Inpaint_minus_Tumor"], color=bar_colors, alpha=0.85)
    ax_bar.axhline(0, color="black", linewidth=1)
    ax_bar.set_ylabel(f"ΔBAG\n{inpaint_label} - {tumor_label}")
    ax_bar.set_xlabel("Subjects ranked by chronological age")
    ax_bar.grid(axis="y", alpha=0.25)


    if show_subject_ticks:
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(df["IXI_ID"], rotation=90, fontsize=6)
    else:
        ax_bar.set_xticks([])


    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()




def write_summary(df, out_path, healthy_label, tumor_label, inpaint_label):
    shift = df["Inpaint_minus_Tumor"]


    lines = []
    lines.append("GLI VS INPAINT BAG SHIFT SUMMARY")
    lines.append("================================")
    lines.append(f"Subjects included: {len(df)}")
    lines.append("")
    lines.append(f"Mean {healthy_label} BAG: {df['BAG_healthy'].mean():.6f}")
    lines.append(f"Mean {tumor_label} BAG: {df['BAG_tumor'].mean():.6f}")
    lines.append(f"Mean {inpaint_label} BAG: {df['BAG_inpaint'].mean():.6f}")
    lines.append("")
    lines.append(f"Mean ΔBAG ({inpaint_label} - {tumor_label}): {shift.mean():.6f}")
    lines.append(f"Median ΔBAG ({inpaint_label} - {tumor_label}): {shift.median():.6f}")
    lines.append(f"Std ΔBAG ({inpaint_label} - {tumor_label}): {shift.std(ddof=1):.6f}")
    lines.append(f"Min ΔBAG ({inpaint_label} - {tumor_label}): {shift.min():.6f}")
    lines.append(f"Max ΔBAG ({inpaint_label} - {tumor_label}): {shift.max():.6f}")
    lines.append("")
    lines.append(f"{inpaint_label} BAG < {tumor_label} BAG: {int((shift < 0).sum())}")
    lines.append(f"{inpaint_label} BAG > {tumor_label} BAG: {int((shift > 0).sum())}")
    lines.append(f"{inpaint_label} BAG = {tumor_label} BAG: {int((shift == 0).sum())}")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(f"Negative ΔBAG means {inpaint_label} lowered BAG relative to {tumor_label}.")
    lines.append(f"Positive ΔBAG means {inpaint_label} increased BAG relative to {tumor_label}.")
    lines.append(f"{healthy_label} is shown only as a reference, not used for the color decision.")


    out_path.write_text("\n".join(lines), encoding="utf-8")




def parse_args():
    parser = argparse.ArgumentParser()


    parser.add_argument("--healthy", required=True, type=Path)
    parser.add_argument("--tumor", required=True, type=Path)
    parser.add_argument("--inpaint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)


    parser.add_argument("--healthy-label", default="Healthy")
    parser.add_argument("--tumor-label", default="GLI")
    parser.add_argument("--inpaint-label", default="GLI+USB")


    parser.add_argument(
        "--title",
        default="Per-subject BAG shift: GLI vs GLI+inpainting",
    )


    parser.add_argument("--show-subject-ticks", action="store_true")
    parser.add_argument("--dpi", type=int, default=250)


    return parser.parse_args()




def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)


    healthy_df = load_prediction_csv(args.healthy, args.healthy_label)
    tumor_df = load_prediction_csv(args.tumor, args.tumor_label)
    inpaint_df = load_prediction_csv(args.inpaint, args.inpaint_label)


    df = build_df(healthy_df, tumor_df, inpaint_df)


    paired_csv = args.output_dir / "gli_vs_inpaint_bag_paired_values.csv"
    plot_path = args.output_dir / "gli_vs_inpaint_bag_shift.png"
    summary_path = args.output_dir / "gli_vs_inpaint_bag_summary.txt"


    df.to_csv(paired_csv, index=False)


    plot_gli_vs_inpaint_shift(
        df=df,
        out_path=plot_path,
        healthy_label=args.healthy_label,
        tumor_label=args.tumor_label,
        inpaint_label=args.inpaint_label,
        title=args.title,
        show_subject_ticks=args.show_subject_ticks,
        dpi=args.dpi,
    )


    write_summary(
        df=df,
        out_path=summary_path,
        healthy_label=args.healthy_label,
        tumor_label=args.tumor_label,
        inpaint_label=args.inpaint_label,
    )


    print("\nDONE")
    print(f"Subjects included: {len(df)}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved paired CSV: {paired_csv}")
    print(f"Saved summary: {summary_path}")




if __name__ == "__main__":
    main()


