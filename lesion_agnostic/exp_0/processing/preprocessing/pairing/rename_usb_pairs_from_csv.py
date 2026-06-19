#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def strip_known_suffix(stem: str) -> str:
    for suf in [
        "_brain_n4_rigid",
        "_preprocessed",
        "_seg_rigid",
        "-seg",
        "_seg",
    ]:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def find_one(folder: Path, patterns: list[str]) -> Path | None:
    hits = []
    for pat in patterns:
        hits.extend(folder.glob(pat))
    hits = sorted(set(hits))
    if len(hits) == 0:
        return None
    if len(hits) > 1:
        raise RuntimeError(f"Multiple matches for {patterns} in {folder}: {hits[:5]}")
    return hits[0]


def copy_or_link(src: Path, dst: Path, mode: str, dry_run: bool) -> None:
    print(f"{mode.upper()}: {src} -> {dst}")
    if dry_run:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        dst.hardlink_to(src)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename/copy IXI T1, T1 affine, and BraTS pathology_probability masks into deterministic IXI__BraTS USB H2P pairs."
    )

    parser.add_argument("--csv", required=True, type=Path)

    parser.add_argument("--t1-dir", required=True, type=Path)
    parser.add_argument("--affine-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)

    parser.add_argument("--out-dir", required=True, type=Path)

    parser.add_argument(
        "--mode",
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="copy is safest; symlink saves space; hardlink works only on same filesystem",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned operations; do not write anything.",
    )

    args = parser.parse_args()

    out_t1 = args.out_dir / "T1"
    out_aff = args.out_dir / "T1_affine"
    out_mask = args.out_dir / "pathology_probability"

    missing = []

    with args.csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} pairs from CSV")
    print(f"Dry run: {args.dry_run}")
    print()

    for i, row in enumerate(rows, start=1):
        healthy_stem = row["healthy_stem"].strip()
        label_stem = row["label_stem"].strip()

        healthy_id = strip_known_suffix(healthy_stem)
        label_id = strip_known_suffix(label_stem)

        case_id = f"{healthy_id}__{label_id}"

        t1_src = find_one(
            args.t1_dir,
            [
                f"{healthy_id}_brain_n4_rigid.nii.gz",
                f"{healthy_id}_preprocessed.nii.gz",
                f"{healthy_id}.nii.gz",
                f"{healthy_id}*.nii.gz",
            ],
        )

        aff_src = find_one(
            args.affine_dir,
            [
                f"{healthy_id}_brain_n4_rigid.affine.npy",
                f"{healthy_id}_preprocessed.affine.npy",
                f"{healthy_id}.affine.npy",
                f"{healthy_id}*.affine.npy",
                f"{healthy_id}*.npy",
            ],
        )

        mask_src = find_one(
            args.mask_dir,
            [
                f"{label_id}_seg_rigid.nii.gz",
                f"{label_id}-seg_rigid.nii.gz",
                f"{label_id}-seg.nii.gz",
                f"{label_id}_seg.nii.gz",
                f"{label_id}.nii.gz",
                f"{label_id}*.nii.gz",
            ],
        )

        if t1_src is None or aff_src is None or mask_src is None:
            missing.append(
                {
                    "row": i,
                    "case_id": case_id,
                    "t1_found": t1_src is not None,
                    "affine_found": aff_src is not None,
                    "mask_found": mask_src is not None,
                }
            )
            continue

        t1_dst = out_t1 / f"{case_id}.nii.gz"
        aff_dst = out_aff / f"{case_id}.affine.npy"
        mask_dst = out_mask / f"{case_id}.nii.gz"

        copy_or_link(t1_src, t1_dst, args.mode, args.dry_run)
        copy_or_link(aff_src, aff_dst, args.mode, args.dry_run)
        copy_or_link(mask_src, mask_dst, args.mode, args.dry_run)
        print()

    print("Done.")
    print(f"Missing/incomplete pairs: {len(missing)}")

    if missing:
        print("\nFirst missing cases:")
        for m in missing[:20]:
            print(m)


if __name__ == "__main__":
    main()
