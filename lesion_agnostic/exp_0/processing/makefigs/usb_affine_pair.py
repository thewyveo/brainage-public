#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path


def find_affine(affine_dir: Path, ixi_id: str):
    matches = []
    for p in affine_dir.rglob("*"):
        if not p.is_file():
            continue
        if ixi_id in p.name and (
            p.name.endswith(".npy")
            or p.name.endswith(".txt")
            or p.name.endswith(".mat")
            or p.name.endswith(".pkl")
            or p.name.endswith(".json")
        ):
            matches.append(p)

    if len(matches) == 0:
        return None

    if len(matches) > 1:
        print(f"[WARN] multiple affine matches for {ixi_id}:")
        for m in matches:
            print(f"       {m}")
        print(f"       using: {matches[0]}")

    return matches[0]


def copy_or_symlink(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(mode)


def strip_nii_gz(name: str):
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--paired-healthy-dir", required=True, type=Path)
    parser.add_argument("--old-affine-dir", required=True, type=Path)
    parser.add_argument("--new-affine-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["copy", "symlink"], default="symlink")

    args = parser.parse_args()

    paired_files = sorted(args.paired_healthy_dir.glob("*_healthy.nii.gz"))

    if len(paired_files) == 0:
        raise RuntimeError(f"No paired healthy files found in {args.paired_healthy_dir}")

    copied = 0
    missing = 0

    for paired_file in paired_files:
        paired_stem = strip_nii_gz(paired_file.name).replace("_healthy", "")

        # Example:
        # IXI131-HH-1527-T1__BraTS-GLI-02708-100
        ixi_id = paired_stem.split("__")[0]

        affine_src = find_affine(args.old_affine_dir, ixi_id)

        if affine_src is None:
            print(f"[MISSING] affine for {ixi_id}")
            missing += 1
            continue

        affine_ext = "".join(affine_src.suffixes)
        affine_dst = args.new_affine_dir / f"{paired_stem}_affine{affine_ext}"

        copy_or_symlink(affine_src, affine_dst, args.mode)

        print(f"[OK] {ixi_id}")
        print(f"     {affine_src}")
        print(f"  -> {affine_dst}")

        copied += 1

    print("\nDONE")
    print(f"Copied/symlinked: {copied}")
    print(f"Missing:          {missing}")


if __name__ == "__main__":
    main()
