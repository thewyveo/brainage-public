import pandas as pd
from pathlib import Path
import shutil

MRI_DIR = Path("exp0_gligan_brats24_ixi_t1_only1tumor")
DEMOGRAPHICS = Path("IXI_clean.xls")
OUT_DIR = Path("exp0_gligan_brats24_ixi_t1_only1tumor_filtered")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DEMOGRAPHICS, sep="\t")

def normalize_ixi_id(x):
    s = str(x).strip()

    # If already like IXI002
    if s.upper().startswith("IXI"):
        num = s[3:]
    else:
        num = s

    # remove accidental decimals like 2.0
    try:
        num = str(int(float(num)))
    except ValueError:
        pass

    return f"IXI{int(num):03d}"

valid_ids = set(df["IXI_ID"].apply(normalize_ixi_id))

print(f"Valid IDs: {len(valid_ids)}")
print("Example valid IDs:", sorted(list(valid_ids))[:10])

kept = 0
skipped = 0

for case_folder in MRI_DIR.iterdir():
    if not case_folder.is_dir():
        continue

    # Example folder:
    # IXI002-Guys-0828-T1__from__BraTS-...
    subject_id = case_folder.name.split("-")[0].strip()

    if subject_id in valid_ids:
        shutil.copytree(case_folder, OUT_DIR / case_folder.name, dirs_exist_ok=True)
        kept += 1
    else:
        skipped += 1

print("\nDone.")
print(f"Kept: {kept}")
print(f"Skipped: {skipped}")

