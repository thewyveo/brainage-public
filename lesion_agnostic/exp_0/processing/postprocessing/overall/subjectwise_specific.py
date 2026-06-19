import os
import re
import pandas as pd
import matplotlib.pyplot as plt




# =========================================================
# INPUT PATHS
# Use raw strings for Windows/UNC paths.
# =========================================================


HEALTHY_CSV = r"\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\rerun\pred\healthy\IXI_BNX.csv"
TUMOR_CSV = r"\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\rerun\pred\exp0\BNX_IXI_GLI.csv"
INPAINT_CSV = r"\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\rerun\pred\exp1\BNX_GLI_LIT.csv"


OUT = "subjectwise_BNX_GLI_LIT_curve"




# =========================================================
# HELPERS
# =========================================================


def read_table(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found:\n{path}")


    ext = os.path.splitext(path)[1].lower()


    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)


    if ext == ".csv":
        return pd.read_csv(path)


    raise ValueError(f"Unsupported file extension: {ext}")




def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df




def find_prediction_column(df):
    candidates = [
        "Predicted_Brain_Age",
        "brainagenext_predictions_run_001_PBA",
        "PBA",
        "prediction",
        "predicted_age",
        "Predicted Age",
    ]


    for c in candidates:
        if c in df.columns:
            return c


    for c in df.columns:
        cl = c.lower()
        if "pba" in cl or "predicted" in cl or "prediction" in cl:
            return c


    raise ValueError(
        "Could not find prediction column. Available columns:\n"
        + "\n".join(df.columns)
    )




def find_age_column(df):
    candidates = ["Age", "age", "Chronological_Age", "chronological_age"]


    for c in candidates:
        if c in df.columns:
            return c


    raise ValueError(
        "Could not find age column. Available columns:\n"
        + "\n".join(df.columns)
    )




def find_id_column(df):
    candidates = ["IXI_ID", "subject_id", "Subject_ID", "ID", "id"]


    for c in candidates:
        if c in df.columns:
            return c


    raise ValueError(
        "Could not find ID column. Available columns:\n"
        + "\n".join(df.columns)
    )




def normalize_ixi_id(x):
    """
    Converts things like:
    IXI002
    IXI002-Guys-0828-T1...
    into IXI002
    """
    x = str(x)
    match = re.search(r"IXI\d+", x)
    if match:
        return match.group(0)
    return x.strip()




def standardize_prediction_file(path, name):
    df = read_table(path)
    df = clean_columns(df)


    id_col = find_id_column(df)
    age_col = find_age_column(df)
    pred_col = find_prediction_column(df)


    out = df[[id_col, age_col, pred_col]].copy()
    out.columns = ["IXI_ID", "Age", f"pred_{name}"]


    out["IXI_ID"] = out["IXI_ID"].apply(normalize_ixi_id)
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out[f"pred_{name}"] = pd.to_numeric(out[f"pred_{name}"], errors="coerce")


    out = out.dropna(subset=["IXI_ID", "Age", f"pred_{name}"])


    return out




# =========================================================
# LOAD FILES
# =========================================================


healthy = standardize_prediction_file(HEALTHY_CSV, "healthy")
tumor = standardize_prediction_file(TUMOR_CSV, "tumor")
inpaint = standardize_prediction_file(INPAINT_CSV, "inpaint")




# =========================================================
# MERGE SUBJECTS
# =========================================================


df = healthy.merge(tumor[["IXI_ID", "pred_tumor"]], on="IXI_ID", how="inner")
df = df.merge(inpaint[["IXI_ID", "pred_inpaint"]], on="IXI_ID", how="inner")


df = df[df["Age"] > 25].copy()


if df.empty:
    raise ValueError("No matched subjects after merging and Age > 25 filtering.")


print(f"Matched subjects after Age > 25 filter: {len(df)}")




# =========================================================
# COMPUTE BAG / PERTURBATION / RECOVERY
# =========================================================


df["BAG_healthy"] = df["pred_healthy"] - df["Age"]
df["BAG_tumor"] = df["pred_tumor"] - df["Age"]
df["BAG_inpaint"] = df["pred_inpaint"] - df["Age"]


df["delta_tumor"] = df["BAG_tumor"] - df["BAG_healthy"]
df["delta_inpaint"] = df["BAG_inpaint"] - df["BAG_healthy"]


df["abs_delta_tumor"] = df["delta_tumor"].abs()
df["abs_delta_inpaint"] = df["delta_inpaint"].abs()


df["recovery"] = df["abs_delta_tumor"] - df["abs_delta_inpaint"]
df["closer_after_inpaint"] = df["recovery"] > 0


print(f"Mean |ΔBAG| tumor:   {df['abs_delta_tumor'].mean():.3f}")
print(f"Mean |ΔBAG| inpaint: {df['abs_delta_inpaint'].mean():.3f}")
print(f"Mean recovery:       {df['recovery'].mean():.3f}")
print(f"Proportion closer:   {df['closer_after_inpaint'].mean():.3f}")




# =========================================================
# SORT SUBJECTS FOR CURVE
# =========================================================


df = df.sort_values("abs_delta_tumor").reset_index(drop=True)
df["rank"] = range(1, len(df) + 1)




# =========================================================
# STYLE
# =========================================================


plt.rcParams.update({
    "figure.facecolor": "black",
    "axes.facecolor": "black",
    "savefig.facecolor": "black",
    "text.color": "white",
    "axes.labelcolor": "white",
    "axes.edgecolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "font.size": 10,
})


WHITE = "white"
GREEN = "#22C55E"
BLUE = "#3B82F6"
GRID = "#444444"




# =========================================================
# PLOT 1: SUBJECT-WISE PERTURBATION CURVE
# =========================================================


fig, ax = plt.subplots(figsize=(8.5, 4.8))


# ---------------------------------------------------------
# MAIN CURVES
# ---------------------------------------------------------


ax.plot(
    df["rank"],
    df["abs_delta_tumor"],
    label="Synthetic tumor perturbation",
    linewidth=2.8,
    color="#60A5FA",   # bright blue
    zorder=3,
)


ax.plot(
    df["rank"],
    df["abs_delta_inpaint"],
    label="Post-inpainting perturbation",
    linewidth=2.8,
    color="#22C55E",   # green
    zorder=4,
)


# ---------------------------------------------------------
# RECOVERED REGIONS
# (inpainting closer to healthy baseline)
# ---------------------------------------------------------


recovered_mask = (
    df["abs_delta_inpaint"] < df["abs_delta_tumor"]
)


ax.fill_between(
    df["rank"],
    df["abs_delta_tumor"],
    df["abs_delta_inpaint"],
    where=recovered_mask,
    interpolate=True,
    alpha=0.55,
    color="#22C55E",
    label="Recovered after inpainting",
    zorder=1,
)


# ---------------------------------------------------------
# WORSENED REGIONS
# (inpainting farther from healthy baseline)
# ---------------------------------------------------------


worsened_mask = (
    df["abs_delta_inpaint"] >= df["abs_delta_tumor"]
)


ax.fill_between(
    df["rank"],
    df["abs_delta_tumor"],
    df["abs_delta_inpaint"],
    where=worsened_mask,
    interpolate=True,
    alpha=0.45,
    color="#EF4444",   # red
    label="Worsened after inpainting",
    zorder=0,
)


# ---------------------------------------------------------
# OPTIONAL: EMPHASIZE CROSSOVER
# ---------------------------------------------------------


ax.axhline(
    0,
    color="white",
    linewidth=1,
    alpha=0.15
)


# ---------------------------------------------------------
# LABELS
# ---------------------------------------------------------


ax.set_title(
    "Subject-wise perturbation trajectory: BNX + GLI + LIT",
    fontweight="bold",
    pad=12,
)


ax.set_xlabel(
    "Subjects sorted by tumor-induced perturbation"
)


ax.set_ylabel(
    r"$|\Delta \mathrm{BAG}|$ relative to healthy baseline (years)"
)


# ---------------------------------------------------------
# GRID / LEGEND
# ---------------------------------------------------------


ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.30,
    color="#666666"
)


legend = ax.legend(
    frameon=False,
    loc="upper left"
)


# ---------------------------------------------------------
# CLEAN SPINES
# ---------------------------------------------------------


for spine in ax.spines.values():
    spine.set_visible(False)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------


plt.tight_layout()


plt.savefig(
    f"{OUT}.png",
    dpi=300,
    bbox_inches="tight"
)


plt.savefig(
    f"{OUT}.pdf",
    bbox_inches="tight"
)


plt.show()






# =========================================================
# PLOT 2: SUBJECT-WISE RECOVERY CURVE
# =========================================================


df_recovery = df.sort_values("recovery").reset_index(drop=True)
df_recovery["rank_recovery"] = range(1, len(df_recovery) + 1)


fig, ax = plt.subplots(figsize=(8, 4.5))


ax.plot(
    df_recovery["rank_recovery"],
    df_recovery["recovery"],
    linewidth=2.2,
    color=GREEN,
)


ax.axhline(0, color=WHITE, linewidth=1)
ax.fill_between(
    df_recovery["rank_recovery"],
    0,
    df_recovery["recovery"],
    where=df_recovery["recovery"] > 0,
    alpha=0.30,
    color=GREEN,
    label="Recovered",
)


ax.fill_between(
    df_recovery["rank_recovery"],
    0,
    df_recovery["recovery"],
    where=df_recovery["recovery"] <= 0,
    alpha=0.25,
    color=BLUE,
    label="Worsened",
)


ax.set_title("Subject-wise inpainting recovery curve: BNX+GLI+LIT", fontweight="bold", pad=12)
ax.set_xlabel("Subjects sorted by recovery")
ax.set_ylabel("Recovery toward healthy baseline (years)")
ax.grid(axis="y", linestyle="--", alpha=0.35, color=GRID)
ax.legend(frameon=False)


for spine in ax.spines.values():
    spine.set_visible(False)


plt.tight_layout()
plt.savefig(f"{OUT}_recovery.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT}_recovery.pdf", bbox_inches="tight")
plt.show()




# =========================================================
# SAVE SUBJECT-LEVEL DATA FOR INSPECTION
# =========================================================


df.to_csv(f"{OUT}_subject_level_values.csv", index=False)


print(f"\nSaved:")
print(f"- {OUT}.png")
print(f"- {OUT}.pdf")
print(f"- {OUT}_recovery.png")
print(f"- {OUT}_recovery.pdf")
print(f"- {OUT}_subject_level_values.csv")


