#!/usr/bin/env python3
import argparse
import json
import re
import shutil
from pathlib import Path


GLIGAN_RE = re.compile(
    r"(?P<ixi>IXI\d+-(?:Guys|HH|IOP)-\d+)__GLI-(?P<gli>\d+-\d+)"
)

CM_RE = re.compile(
    r"(?P<ixi>IXI\d+-(?:Guys|HH|IOP)-\d+)-T1_brain_n4_rigid_carvemix\.nii(?:\.gz)?$"
)

META_GLI_RE = re.compile(r"BraTS-GLI-(?P<gli>\d+-\d+)")


def extract_gligan_pairs(gligan_dir: Path):
    pairs = set()
    ixis = set()
    glis = set()

    for path in sorted(gligan_dir.glob("*.nii*")):
        m = GLIGAN_RE.search(path.name)
        if not m:
            print(f"[WARN] Could not parse GliGAN filename: {path.name}")
            continue

        ixi = m.group("ixi")
        gli = f"GLI-{m.group('gli')}"

        pairs.add((ixi, gli))
        ixis.add(ixi)
        glis.add(gli)

    return pairs, ixis, glis


def find_metadata_for_cm(cm_path: Path, metadata_dir: Path):
    base = cm_path.name
    base = base.replace("_carvemix.nii.gz", "_metadata.json")
    base = base.replace("_carvemix.nii", "_metadata.json")
    return metadata_dir / base


def extract_cm_pair(cm_path: Path, metadata_dir: Path):
    m = CM_RE.search(cm_path.name)
    if not m:
        print(f"[WARN] Could not parse CarveMix filename: {cm_path.name}")
        return None

    ixi = m.group("ixi")
    meta_path = find_metadata_for_cm(cm_path, metadata_dir)

    if not meta_path.exists():
        print(f"[WARN] Missing metadata for {cm_path.name}: {meta_path}")
        return None

    with open(meta_path, "r") as f:
        meta = json.load(f)

    library_item_path = meta.get("library_item_path", "")
    m_gli = META_GLI_RE.search(library_item_path)

    if not m_gli:
        print(f"[WARN] Could not extract GLI ID from metadata: {meta_path.name}")
        return None

    gli = f"GLI-{m_gli.group('gli')}"
    return ixi, gli, meta_path


def write_list(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in sorted(values):
            f.write(str(v) + "\n")


def write_pairs_csv(path: Path, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("ixi_id,gli_id\n")
        for ixi, gli in sorted(pairs):
            f.write(f"{ixi},{gli}\n")


def copy_or_move(src: Path, dst_dir: Path, mode: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError("mode must be copy or move")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gligan-dir", required=True)
    parser.add_argument("--carvemix-dir", required=True)
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument(
        "--mode",
        choices=["copy", "move"],
        default="copy",
        help="copy is safer; move actually reorganizes files.",
    )

    args = parser.parse_args()

    gligan_dir = Path(args.gligan_dir)
    carvemix_dir = Path(args.carvemix_dir)
    metadata_dir = Path(args.metadata_dir)
    out_dir = Path(args.out_dir)

    gligan_pairs, gligan_ixis, gligan_glis = extract_gligan_pairs(gligan_dir)

    write_pairs_csv(out_dir / "gligan_pairs.csv", gligan_pairs)
    write_list(out_dir / "gligan_ixi_ids.txt", gligan_ixis)
    write_list(out_dir / "gligan_gli_ids.txt", gligan_glis)

    matched = []
    conflicts = []
    extra_unmatched = []
    parse_failed = []

    for cm_path in sorted(carvemix_dir.glob("*.nii*")):
        parsed = extract_cm_pair(cm_path, metadata_dir)

        if parsed is None:
            parse_failed.append(cm_path.name)
            continue

        cm_ixi, cm_gli, meta_path = parsed
        cm_pair = (cm_ixi, cm_gli)

        if cm_pair in gligan_pairs:
            category = "matched"
            matched.append(cm_pair)
        elif cm_ixi in gligan_ixis or cm_gli in gligan_glis:
            category = "conflict"
            conflicts.append(cm_pair)
        else:
            category = "extra_unmatched"
            extra_unmatched.append(cm_pair)

        copy_or_move(cm_path, out_dir / category / "carvemix", args.mode)
        copy_or_move(meta_path, out_dir / category / "metadata", args.mode)

    write_pairs_csv(out_dir / "carvemix_matched_pairs.csv", matched)
    write_pairs_csv(out_dir / "carvemix_conflict_pairs.csv", conflicts)
    write_pairs_csv(out_dir / "carvemix_extra_unmatched_pairs.csv", extra_unmatched)
    write_list(out_dir / "parse_failed.txt", parse_failed)

    print()
    print("Done.")
    print(f"GliGAN canonical pairs:       {len(gligan_pairs)}")
    print(f"GliGAN IXI IDs:               {len(gligan_ixis)}")
    print(f"GliGAN GLI IDs:               {len(gligan_glis)}")
    print(f"CarveMix exact matched:       {len(matched)}")
    print(f"CarveMix conflicts:           {len(conflicts)}")
    print(f"CarveMix extra unmatched:     {len(extra_unmatched)}")
    print(f"Parse failed:                 {len(parse_failed)}")
    print()
    print(f"Output written to: {out_dir}")


if __name__ == "__main__":
    main()
