from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

r"""
py -3.10 exp_0\processing\preprocessing\utils\gligan_flatten.py `
  --input-dir "exp_0\synth_lesion_generator\GliGAN\data\generated_t1_gligan_FAITHFUL" `
  --output-dir "data\preprocessed\GliGAN\FAITHFULFLAT_generated_t1_gligan" `
  --mega-metadata-dir "data\preprocessed\GliGAN\mega_metadata"
"""

IXI_ID_RE = re.compile(r"(IXI\d+-(?:Guys|HH|IOP)-\d+)", flags=re.IGNORECASE)
GLI_ID_RE = re.compile(r"(GLI-\d{5}-\d+)", flags=re.IGNORECASE)


@dataclass(frozen=True)
class CaseIds:
    ixi_id: str
    gli_id: str

    @property
    def out_stem(self) -> str:
        return f"{self.ixi_id}__{self.gli_id}"


def _iter_strings(obj: Any) -> Iterable[str]:
    if obj is None:
        return
    if isinstance(obj, str):
        yield obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_strings(k)
            yield from _iter_strings(v)
        return
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _iter_strings(v)
        return


def _first_match(regex: re.Pattern[str], strings: Iterable[str]) -> str | None:
    for s in strings:
        m = regex.search(s)
        if m:
            return m.group(1)
    return None


def parse_ids_from_metadata(metadata: dict[str, Any]) -> CaseIds:
    """
    Extract IXI and GLI IDs from metadata.

    Prefers `healthy_path` and `label_path` keys if present, but will fall back
    to scanning all string fields in the JSON.
    """
    preferred_strings: list[str] = []
    for key in ("healthy_path", "label_path", "healthy", "label", "image", "seg"):
        v = metadata.get(key)
        if isinstance(v, str):
            preferred_strings.append(v)

    all_strings = list(_iter_strings(metadata))

    ixi = _first_match(IXI_ID_RE, preferred_strings) or _first_match(IXI_ID_RE, all_strings)
    gli = _first_match(GLI_ID_RE, preferred_strings) or _first_match(GLI_ID_RE, all_strings)

    if not ixi or not gli:
        raise ValueError(
            "Could not parse required IDs from metadata.json "
            f"(ixi_id={ixi!r}, gli_id={gli!r})."
        )

    # Normalize casing to match your canonical formatting.
    ixi_parts = ixi.split("-")
    if len(ixi_parts) >= 3:
        prefix = ixi_parts[0].upper()  # IXI###
        site_raw = ixi_parts[1].upper()
        if site_raw == "GUYS":
            site = "Guys"
        elif site_raw == "HH":
            site = "HH"
        elif site_raw == "IOP":
            site = "IOP"
        else:
            site = ixi_parts[1]
        rest = "-".join(ixi_parts[2:])
        ixi = f"{prefix}-{site}-{rest}"
    else:
        ixi = ixi[:3].upper() + ixi[3:]

    gli = gli.upper()
    return CaseIds(ixi_id=ixi, gli_id=gli)


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = "".join(path.suffixes)
    parent = path.parent
    for i in range(2, 10_000):
        candidate = parent / f"{stem}__dup{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a unique filename near: {path}")


def _copy_case_metadata_folder(case_dir: Path, mega_metadata_dir: Path) -> Path:
    """
    Preserve each case folder's contents (except synthetic_t1) in mega-metadata.
    """
    dest = mega_metadata_dir / case_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    for p in case_dir.iterdir():
        if p.is_dir():
            # Skip nested directories to keep this predictable and fast.
            continue
        if p.name.lower() == "synthetic_t1.nii.gz":
            continue
        shutil.copy2(p, dest / p.name)
    return dest


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at {path}, got {type(data).__name__}.")
    return data


def flatten_gligan_folder(
    input_dir: Path,
    output_dir: Path,
    mega_metadata_dir: Path,
    *,
    dry_run: bool = False,
) -> None:
    input_dir = input_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mega_metadata_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, str]] = []

    for case_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        metadata_path = case_dir / "metadata.json"
        synthetic_path = case_dir / "synthetic_t1.nii.gz"
        if not metadata_path.exists():
            print(f"[skip] {case_dir.name}: missing metadata.json")
            continue
        if not synthetic_path.exists():
            print(f"[skip] {case_dir.name}: missing synthetic_t1.nii.gz")
            continue

        metadata = _load_json(metadata_path)
        ids = parse_ids_from_metadata(metadata)

        out_path = output_dir / f"{ids.out_stem}.nii.gz"
        out_path = _unique_path(out_path)

        print(f"[case] {case_dir.name} -> {out_path.name}")
        if not dry_run:
            # Move/flatten the synthetic volume.
            shutil.move(str(synthetic_path), str(out_path))

            # Preserve metadata and any other files next to it (excluding the volume).
            meta_case_folder = _copy_case_metadata_folder(case_dir, mega_metadata_dir)

            # Also drop a convenient copy of metadata.json named by output stem.
            named_meta = mega_metadata_dir / f"{out_path.stem}.metadata.json"
            shutil.copy2(str(metadata_path), str(_unique_path(named_meta)))

        index_rows.append(
            {
                "case_folder": case_dir.name,
                "ixi_id": ids.ixi_id,
                "gli_id": ids.gli_id,
                "out_file": out_path.name,
            }
        )

    # Write an index for traceability.
    index_path = mega_metadata_dir / "index.csv"
    if index_rows and not dry_run:
        import csv

        with index_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
            w.writeheader()
            w.writerows(index_rows)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Flatten a GliGAN generated folder: for each case subfolder, parse "
            "IXI + GLI IDs from metadata.json, rename synthetic_t1.nii.gz to "
            "IXI...__GLI....nii.gz, move to a single output directory, and "
            "preserve per-case metadata/files in a mega-metadata folder."
        )
    )
    p.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Folder containing per-case subfolders (each with metadata.json + synthetic_t1.nii.gz).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Flattened output directory for renamed synthetic volumes.",
    )
    p.add_argument(
        "--mega-metadata-dir",
        required=True,
        type=Path,
        help="Directory to store preserved metadata/files for all cases plus index.csv.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions without moving/copying.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    flatten_gligan_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mega_metadata_dir=args.mega_metadata_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
