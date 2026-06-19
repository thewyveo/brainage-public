import pandas as pd


IXI_FILE = r"data\labels\IXI.xls"


df = pd.read_excel(IXI_FILE)


print("Columns found:")
print(df.columns.tolist())


df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
df["SEX_ID (1=m, 2=f)"] = pd.to_numeric(df["SEX_ID (1=m, 2=f)"], errors="coerce")


df_clean = df[
    df["AGE"].notna() &
    (df["AGE"] >= 25)
].copy()


df_clean["SEX"] = df_clean["SEX_ID (1=m, 2=f)"].map({1: "M", 2: "F"})


n = len(df_clean)
age_min = df_clean["AGE"].min()
age_max = df_clean["AGE"].max()
age_mean = df_clean["AGE"].mean()
age_std = df_clean["AGE"].std()


sex_counts = df_clean["SEX"].value_counts(dropna=False)
sex_percent = df_clean["SEX"].value_counts(normalize=True, dropna=False) * 100


print("IXI Healthy Cohort")
print(f"N = {n}")
print(f"Age range = {age_min:.2f}--{age_max:.2f}")
print(f"Age mean ± std = {age_mean:.2f} ± {age_std:.2f}")
print("\nSex counts:")
print(sex_counts)
print("\nSex percentages:")
print(sex_percent.round(2))


print("\nLaTeX-ready:")
print(
    f"$N={n}$ healthy subjects "
    f"(age: {age_min:.1f}--{age_max:.1f} years, "
    f"{age_mean:.1f} $\\pm$ {age_std:.1f}; "
    f"sex: {sex_percent.get('M', 0):.1f}\\% male, "
    f"{sex_percent.get('F', 0):.1f}\\% female)."
)
