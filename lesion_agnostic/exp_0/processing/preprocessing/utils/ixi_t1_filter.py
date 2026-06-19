#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import shutil


INPUT_DIR = Path("IXI")        # folder with mixed T1/T2
OUTPUT_DIR = Path("ixi_t1_only")        # where T1 files go


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


kept = 0
skipped = 0


# supports both .nii and .nii.gz
for file in INPUT_DIR.glob("*"):
    if not file.is_file():
        continue

    name = file.name

    # check for T1 ending
    if name.endswith("-T1.nii") or name.endswith("-T1.nii.gz"):
        shutil.copy2(file, OUTPUT_DIR / name)
        kept += 1
    else:
        skipped += 1


print("\nDone.")
print(f"Kept (T1): {kept}")
print(f"Skipped: {skipped}")