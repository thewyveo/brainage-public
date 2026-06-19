#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import shutil
import argparse


r"""
py -3.10 exp_0\processing\preprocessing\utils\brats_flatten.py `
    --input-root data\raw\training_data1_v2 `
    --output-dir data\library\CM_BraTS_Masks
"""




def is_seg_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith("-seg.nii") or name.endswith("-seg.nii.gz")




def is_t1n_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith("-t1n.nii") or name.endswith("-t1n.nii.gz")




def make_unique_destination(dst_dir: Path, filename: str) -> Path:
    dst = dst_dir / filename
    if not dst.exists():
        return dst


    if filename.endswith(".nii.gz"):
        stem = filename[:-7]
        suffix = ".nii.gz"
    else:
        stem = Path(filename).stem
        suffix = Path(filename).suffix


    counter = 1
    while True:
        candidate = dst_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1




def find_case_dirs(input_root: Path) -> list[Path]:
    """
    Find BraTS case directories directly under input_root.
    Example:
        data/training_data1_v2/BraTS-GLI-00005-100
    """
    case_dirs = []
    for p in input_root.rglob("*"):
        if p.is_dir() and p.name.startswith("BraTS-GLI-"):
            case_dirs.append(p)
    return sorted(case_dirs)




def find_single_file(case_dir: Path, predicate, label: str) -> Path | None:
    matches = [p for p in case_dir.rglob("*") if p.is_file() and predicate(p)]


    if len(matches) == 0:
        print(f"  [WARN] No {label} found in case folder: {case_dir}")
        return None


    if len(matches) > 1:
        print(f"  [WARN] Multiple {label} files found in case folder: {case_dir}")
        for m in matches:
            print(f"         - {m}")
        print(f"         Using first match: {matches[0]}")


    return sorted(matches)[0]




def extract_seg_and_t1n(input_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


    case_dirs = find_case_dirs(input_root)


    if len(case_dirs) == 0:
        print(f"[WARN] No BraTS case directories found under: {input_root}")
        return


    copied_seg = 0
    copied_t1n = 0
    missing_seg = 0
    missing_t1n = 0


    for i, case_dir in enumerate(case_dirs, start=1):
        print(f"\n[{i}/{len(case_dirs)}] Case: {case_dir.name}")


        seg_path = find_single_file(case_dir, is_seg_file, "seg")
        t1n_path = find_single_file(case_dir, is_t1n_file, "t1n")


        if seg_path is None:
            missing_seg += 1
        else:
            seg_dst = make_unique_destination(output_dir, seg_path.name)
            shutil.copy2(seg_path, seg_dst)
            copied_seg += 1
            print(f"  [SEG ] Copied: {seg_path} -> {seg_dst}")


        if t1n_path is None:
            missing_t1n += 1
        else:
            t1n_dst = make_unique_destination(output_dir, t1n_path.name)
            shutil.copy2(t1n_path, t1n_dst)
            copied_t1n += 1
            print(f"  [T1N ] Copied: {t1n_path} -> {t1n_dst}")


    print("\nDone.")
    print(f"Case folders found: {len(case_dirs)}")
    print(f"Seg files copied: {copied_seg}")
    print(f"T1n files copied: {copied_t1n}")
    print(f"Case folders missing seg: {missing_seg}")
    print(f"Case folders missing t1n: {missing_t1n}")
    print(f"Output folder: {output_dir}")




def main():
    parser = argparse.ArgumentParser(
        description="Flatten BraTS nested case folders to only seg and corresponding t1n files."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root folder containing nested BraTS case folders."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Flat output folder where seg and t1n files will be copied."
    )
    args = parser.parse_args()


    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")


    extract_seg_and_t1n(args.input_root, args.output_dir)




if __name__ == "__main__":
    main()

