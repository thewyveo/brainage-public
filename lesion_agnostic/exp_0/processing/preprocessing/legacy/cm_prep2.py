#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import ants


def strip_nii(name: str) -> str:
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def safe_rel_stem(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    parts = list(rel.parts)
    parts[-1] = strip_nii(parts[-1])
    return "__".join(parts)


def matches_filter(path: Path, root: Path, name_filter: str | None, recursive: bool) -> bool:
    if name_filter is None or name_filter.strip() == "":
        return True

    filt = name_filter.lower()

    if recursive:
        searchable = str(path.relative_to(root)).lower()
    else:
        searchable = path.name.lower()

    return filt in searchable


def find_niftis(root: Path, name_filter: str | None, recursive: bool) -> list[Path]:
    files: list[Path] = []

    iterator = root.rglob("*") if recursive else root.glob("*")

    for path in iterator:
        if not path.is_file():
            continue

        lower = path.name.lower()
        if not (lower.endswith(".nii.gz") or lower.endswith(".nii")):
            continue

        if not matches_filter(path, root, name_filter, recursive):
            continue

        files.append(path)

    return sorted(files)


def process_image(
    img_path: str,
    out_path: str,
    mni_path: str,
) -> tuple[str, bool, str]:

    name = Path(img_path).name

    try:
        fixed = ants.image_read(mni_path)
        moving = ants.image_read(img_path)

        moving_n4 = ants.n4_bias_field_correction(moving)

        reg = ants.registration(
            fixed=fixed,
            moving=moving_n4,
            type_of_transform="Rigid",
        )

        ants.image_write(reg["warpedmovout"], out_path)

        return name, True, "OK"

    except Exception as e:
        return name, False, str(e)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="N4 + rigid-register SynthStripped IXI T1 images to MNI."
    )

    parser.add_argument("--ixi-img-dir", type=Path, required=True)
    parser.add_argument("--out-img-dir", type=Path, required=True)
    parser.add_argument("--mni", type=Path, required=True)

    parser.add_argument("--image-filter", type=str, default="T1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Search recursively. Output filenames include relative subfolder names, "
            "e.g. sub/case/T1.nii.gz -> sub__case__T1_n4_rigid.nii.gz"
        ),
    )

    args = parser.parse_args()

    args.out_img_dir.mkdir(parents=True, exist_ok=True)

    img_files = find_niftis(
        root=args.ixi_img_dir,
        name_filter=args.image_filter,
        recursive=args.recursive,
    )

    if args.limit is not None:
        img_files = img_files[: args.limit]

    print(f"Found images: {len(img_files)}")
    print(f"Recursive mode: {args.recursive}")

    done = 0
    failed = 0
    futures = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for img_path in img_files:
            if args.recursive:
                stem = safe_rel_stem(img_path, args.ixi_img_dir)
            else:
                stem = strip_nii(img_path.name)

            out_path = args.out_img_dir / f"{stem}_n4_rigid.nii.gz"

            if out_path.exists():
                print(f"EXISTS {img_path}")
                done += 1
                continue

            futures.append(
                ex.submit(
                    process_image,
                    str(img_path),
                    str(out_path),
                    str(args.mni),
                )
            )

        for fut in as_completed(futures):
            name, ok, msg = fut.result()

            if ok:
                print(f"DONE   {name}")
                done += 1
            else:
                print(f"FAILED {name}: {msg}")
                failed += 1

    print()
    print("Finished.")
    print(f"Done:   {done}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
