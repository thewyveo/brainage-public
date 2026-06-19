import pandas as pd


BRATS_FILE = r"data\labels\BraTS_24.xlsx"


df = pd.read_excel(BRATS_FILE)


# Clean column names
df.columns = (
    df.columns.astype(str)
    .str.replace("\n", " ", regex=False)
    .str.replace("\r", " ", regex=False)
    .str.strip()
)


print("Columns found:")
print(df.columns.tolist())


age_col = "Patient's Age"
sex_col = "Patient's Sex"


# Find glioma column robustly
glioma_candidates = [c for c in df.columns if "glioma" in c.lower()]
glioma_col = glioma_candidates[0] if glioma_candidates else None


df[age_col] = pd.to_numeric(df[age_col], errors="coerce")
df[sex_col] = df[sex_col].astype(str).str.upper().str.strip()


df_clean = df[df[age_col].notna()].copy()


n = len(df_clean)
age_min = df_clean[age_col].min()
age_max = df_clean[age_col].max()
age_mean = df_clean[age_col].mean()
age_std = df_clean[age_col].std()


sex_counts = df_clean[sex_col].value_counts(dropna=False)
sex_percent = df_clean[sex_col].value_counts(normalize=True, dropna=False) * 100


print("BraTS 2024 Tumor Cohort")
print(f"N = {n}")
print(f"Age range = {age_min:.2f}--{age_max:.2f}")
print(f"Age mean ± std = {age_mean:.2f} ± {age_std:.2f}")


print("\nSex counts:")
print(sex_counts)


print("\nSex percentages:")
print(sex_percent.round(2))


if glioma_col is not None:
    glioma_counts = df_clean[glioma_col].value_counts(dropna=False)
    print(f"\nGlioma type column used: {glioma_col}")
    print("\nGlioma type counts:")
    print(glioma_counts)
else:
    print("\nNo glioma type column found.")


print("\nLaTeX-ready:")
print(
    f"$N={n}$ subjects "
    f"(age: {age_min:.1f}--{age_max:.1f} years, "
    f"{age_mean:.1f} $\\pm$ {age_std:.1f}; "
    f"sex: {sex_percent.get('M', 0):.1f}\\% male, "
    f"{sex_percent.get('F', 0):.1f}\\% female)."
)
