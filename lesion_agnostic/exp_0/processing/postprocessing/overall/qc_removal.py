# remove_subjects_from_predictions.py


import argparse
import re
from pathlib import Path


import pandas as pd




def parse_subject_list(txt_path):
    """
    Accepts lines like:
    i 101
    i 503
    IXI002
    IXI 12


    Returns:
    {"IXI101", "IXI503", "IXI002", "IXI012"}
    """
    subjects = set()


    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue


            match = re.search(r"(?:ixi|i)\s*0*(\d+)", line, flags=re.IGNORECASE)
            if match:
                num = int(match.group(1))
                subjects.add(f"IXI{num:03d}")


    return subjects




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input predictions CSV")
    parser.add_argument("--remove_txt", required=True, help="TXT file with subjects to remove")
    parser.add_argument("--output_csv", required=True, help="Output cleaned CSV")
    parser.add_argument("--subject_col", default="IXI_ID", help="Column containing subject IDs")


    args = parser.parse_args()


    csv_path = Path(args.csv)
    txt_path = Path(args.remove_txt)
    out_path = Path(args.output_csv)


    df = pd.read_csv(csv_path)
    remove_subjects = parse_subject_list(txt_path)


    before = len(df)


    df_clean = df[~df[args.subject_col].astype(str).isin(remove_subjects)].copy()


    after = len(df_clean)
    removed = before - after


    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False)


    print(f"Subjects requested for removal: {len(remove_subjects)}")
    print(f"Rows before: {before}")
    print(f"Rows after:  {after}")
    print(f"Rows removed: {removed}")
    print(f"Saved cleaned CSV to: {out_path}")


    actually_found = set(df[args.subject_col].astype(str)) & remove_subjects
    missing = remove_subjects - actually_found


    if missing:
        print("\nWarning: these subjects were in the txt file but not found in the CSV:")
        for s in sorted(missing):
            print(s)




if __name__ == "__main__":
    main()


