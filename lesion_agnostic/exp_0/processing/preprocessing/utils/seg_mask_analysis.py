#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


r"""
py -3.10 exp_0\processing\preprocessing\utils\seg_mask_analysis.py `
    --seg-root data\library\BraTS_Masks `
    --output-dir exp_0\processing\preprocessing\utils\insights\BraTS_Mask_Analysis
"""

def is_seg_file(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith("-seg.nii") or name.endswith("-seg.nii.gz")




def strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem




def get_bbox(mask: np.ndarray):
    """
    Returns bounding box for nonzero voxels as:
    (x_min, x_max, y_min, y_max, z_min, z_max)
    If mask is empty, returns None.
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None


    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)


    return (
        int(mins[0]), int(maxs[0]),
        int(mins[1]), int(maxs[1]),
        int(mins[2]), int(maxs[2]),
    )




def bbox_sizes_from_bbox(bbox):
    if bbox is None:
        return 0, 0, 0
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    return (
        x_max - x_min + 1,
        y_max - y_min + 1,
        z_max - z_min + 1,
    )




def analyze_seg(seg_path: Path) -> dict:
    img = nib.load(str(seg_path))
    data = img.get_fdata()
    mask = data > 0


    tumor_voxels = int(np.count_nonzero(mask))
    total_voxels = int(mask.size)
    tumor_fraction = float(tumor_voxels / total_voxels) if total_voxels > 0 else 0.0


    zooms = img.header.get_zooms()[:3]
    voxel_volume_mm3 = float(zooms[0] * zooms[1] * zooms[2])
    tumor_volume_mm3 = float(tumor_voxels * voxel_volume_mm3)


    bbox = get_bbox(mask)
    bbox_x, bbox_y, bbox_z = bbox_sizes_from_bbox(bbox)
    bbox_volume_voxels = int(bbox_x * bbox_y * bbox_z)


    return {
        "case_id": strip_nii_suffix(seg_path.name).replace("-seg", ""),
        "filename": seg_path.name,
        "filepath": str(seg_path),
        "tumor_voxels": tumor_voxels,
        "tumor_volume_mm3": tumor_volume_mm3,
        "tumor_fraction": tumor_fraction,
        "voxel_size_x_mm": float(zooms[0]),
        "voxel_size_y_mm": float(zooms[1]),
        "voxel_size_z_mm": float(zooms[2]),
        "bbox_x_voxels": bbox_x,
        "bbox_y_voxels": bbox_y,
        "bbox_z_voxels": bbox_z,
        "bbox_volume_voxels": bbox_volume_voxels,
    }




def save_distribution_plots(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


    # Histogram: tumor voxel counts
    plt.figure(figsize=(8, 5))
    plt.hist(df["tumor_voxels"], bins=40)
    plt.xlabel("Tumor voxels")
    plt.ylabel("Number of cases")
    plt.title("Distribution of tumor sizes (voxel count)")
    plt.tight_layout()
    plt.savefig(output_dir / "tumor_voxel_distribution.png", dpi=200)
    plt.close()


    # Histogram: tumor volume mm3
    plt.figure(figsize=(8, 5))
    plt.hist(df["tumor_volume_mm3"], bins=40)
    plt.xlabel("Tumor volume (mm³)")
    plt.ylabel("Number of cases")
    plt.title("Distribution of tumor sizes (mm³)")
    plt.tight_layout()
    plt.savefig(output_dir / "tumor_volume_mm3_distribution.png", dpi=200)
    plt.close()


    # Histogram: tumor fraction of whole image
    plt.figure(figsize=(8, 5))
    plt.hist(df["tumor_fraction"], bins=40)
    plt.xlabel("Tumor fraction of image")
    plt.ylabel("Number of cases")
    plt.title("Distribution of tumor fraction")
    plt.tight_layout()
    plt.savefig(output_dir / "tumor_fraction_distribution.png", dpi=200)
    plt.close()


    # Boxplot: tumor volume
    plt.figure(figsize=(6, 5))
    plt.boxplot(df["tumor_volume_mm3"].values, vert=True)
    plt.ylabel("Tumor volume (mm³)")
    plt.title("Tumor volume boxplot")
    plt.tight_layout()
    plt.savefig(output_dir / "tumor_volume_boxplot.png", dpi=200)
    plt.close()


    # Optional: log-scale histogram if sizes are very skewed
    positive = df[df["tumor_voxels"] > 0]["tumor_voxels"]
    if len(positive) > 0:
        plt.figure(figsize=(8, 5))
        plt.hist(np.log10(positive), bins=40)
        plt.xlabel("log10(Tumor voxels)")
        plt.ylabel("Number of cases")
        plt.title("Distribution of tumor sizes (log10 voxel count)")
        plt.tight_layout()
        plt.savefig(output_dir / "tumor_voxel_distribution_log10.png", dpi=200)
        plt.close()




def save_summary(df: pd.DataFrame, output_dir: Path) -> None:
    summary_lines = []


    def line(text=""):
        summary_lines.append(text)


    line(f"Number of cases: {len(df)}")
    line()


    for col in ["tumor_voxels", "tumor_volume_mm3", "tumor_fraction", "bbox_volume_voxels"]:
        line(f"{col}:")
        line(f"  min    = {df[col].min()}")
        line(f"  max    = {df[col].max()}")
        line(f"  mean   = {df[col].mean()}")
        line(f"  median = {df[col].median()}")
        line(f"  std    = {df[col].std()}")
        line()


    biggest = df.sort_values("tumor_voxels", ascending=False).iloc[0]
    line("Biggest tumor case by voxel count:")
    line(f"  case_id          = {biggest['case_id']}")
    line(f"  filename         = {biggest['filename']}")
    line(f"  tumor_voxels     = {biggest['tumor_voxels']}")
    line(f"  tumor_volume_mm3 = {biggest['tumor_volume_mm3']}")
    line(f"  tumor_fraction   = {biggest['tumor_fraction']}")
    line(f"  bbox (x,y,z)     = ({biggest['bbox_x_voxels']}, {biggest['bbox_y_voxels']}, {biggest['bbox_z_voxels']})")


    with open(output_dir / "dataset_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))




def main():
    parser = argparse.ArgumentParser(
        description="Analyze BraTS tumor masks and plot tumor scale distributions."
    )
    parser.add_argument(
        "--seg-root",
        type=Path,
        required=True,
        help="Folder containing BraTS segmentation masks."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Folder where CSV, summary, and plots will be saved."
    )
    args = parser.parse_args()


    if not args.seg_root.exists():
        raise FileNotFoundError(f"Seg root not found: {args.seg_root}")


    args.output_dir.mkdir(parents=True, exist_ok=True)


    seg_files = sorted([p for p in args.seg_root.rglob("*") if p.is_file() and is_seg_file(p)])
    if not seg_files:
        raise FileNotFoundError(f"No seg files found in: {args.seg_root}")


    rows = []
    for i, seg_path in enumerate(seg_files, start=1):
        row = analyze_seg(seg_path)
        rows.append(row)
        print(f"[{i}/{len(seg_files)}] {seg_path.name} -> tumor_voxels={row['tumor_voxels']}")


    df = pd.DataFrame(rows)
    df = df.sort_values("tumor_voxels", ascending=False).reset_index(drop=True)


    csv_path = args.output_dir / "brats_tumor_mask_stats.csv"
    df.to_csv(csv_path, index=False)


    save_distribution_plots(df, args.output_dir)
    save_summary(df, args.output_dir)


    biggest = df.iloc[0]
    print("\nDone.")
    print(f"CSV saved to: {csv_path}")
    print(f"Plots saved to: {args.output_dir}")
    print("Biggest tumor case:")
    print(f"  case_id: {biggest['case_id']}")
    print(f"  filename: {biggest['filename']}")
    print(f"  tumor_voxels: {biggest['tumor_voxels']}")
    print(f"  tumor_volume_mm3: {biggest['tumor_volume_mm3']}")
    print(f"  tumor_fraction: {biggest['tumor_fraction']}")




if __name__ == "__main__":
    main()

