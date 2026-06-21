import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt


# ============================================================
# EDIT CSV PATHS
# ============================================================

CSV_PATHS = {
    "GLI": r"C:\Users\P102179\OneDrive - Amsterdam UMC\Bureaublad\repo\brainage\lesion_agnostic\data\results\ALLPOSTPROCESS\raw\exp0\BNX\GLI\paired_raw_BAG.csv",
    "CM": r"C:\Users\P102179\OneDrive - Amsterdam UMC\Bureaublad\repo\brainage\lesion_agnostic\data\results\ALLPOSTPROCESS\raw\exp0\BNX\CM\paired_raw_BAG.csv",
    "GLI_LIT": r"C:\Users\P102179\OneDrive - Amsterdam UMC\Bureaublad\repo\brainage\lesion_agnostic\data\results\ALLPOSTPROCESS\raw\exp1\BNX\GLI\LIT\triplet_raw_BAG.csv",
    "GLI_BID": r"C:\Users\P102179\OneDrive - Amsterdam UMC\Bureaublad\repo\brainage\lesion_agnostic\data\results\ALLPOSTPROCESS\raw\exp1\BNX\GLI\BID\triplet_raw_BAG.csv",
    "CM_LIT": r"C:\Users\P102179\OneDrive - Amsterdam UMC\Bureaublad\repo\brainage\lesion_agnostic\data\results\ALLPOSTPROCESS\raw\exp1\BNX\CM\LIT\triplet_raw_BAG.csv",
    "CM_BID": r"C:\Users\P102179\OneDrive - Amsterdam UMC\Bureaublad\repo\brainage\lesion_agnostic\data\results\ALLPOSTPROCESS\raw\exp1\BNX\CM\BID\triplet_raw_BAG.csv",
}


# ============================================================
# EDIT ROOT PATHS
# ============================================================


ROOTS = {
    "GLI": r"\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\gen\IXI_GLI",
    "CM_MASKS": r"\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\gen\IXI_CM\tumor_masks",
}


OUT_DIR = "weighted_atlases_aggregated"
MIN_COUNT = 3
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# GROUP DEFINITIONS
# ============================================================


GROUPS = {
    "EXP0_ALL": ["GLI", "CM"],
    "EXP1_ALL": ["GLI_LIT", "GLI_BID", "CM_LIT", "CM_BID"],
    "LIT_ALL": ["GLI_LIT", "CM_LIT"],
    "BID_ALL": ["GLI_BID", "CM_BID"],
}


ALL_CONDITIONS = ["GLI", "CM", "GLI_LIT", "GLI_BID", "CM_LIT", "CM_BID"]


# ============================================================
# PATH HELPERS
# ============================================================


def first_match(pattern):
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def find_case_dir(root, ixi_id):
    return first_match(os.path.join(root, f"{ixi_id}*"))


def get_gli_mask(ixi_id):
    case_dir = find_case_dir(ROOTS["GLI"], ixi_id)
    if case_dir is None:
        return None
    return os.path.join(case_dir, "synthetic_seg.nii.gz")


def get_cm_mask(ixi_id):
    return first_match(os.path.join(ROOTS["CM_MASKS"], f"{ixi_id}*_tumor_mask.nii.gz"))


def get_mask_path(condition, ixi_id):
    # all inpainting conditions use original synthetic tumor-generation masks
    if condition.startswith("GLI"):
        return get_gli_mask(ixi_id)
    if condition.startswith("CM"):
        return get_cm_mask(ixi_id)
    raise ValueError(f"Unknown condition: {condition}")


# ============================================================
# IMAGE HELPERS
# ============================================================


def load_mask(path):
    img = nib.load(path)
    data = np.squeeze(img.get_fdata())
    mask = (data > 0).astype(np.float32)
    return img, mask


def save_nifti(arr, ref_img, path):
    nib.save(
        nib.Nifti1Image(arr.astype(np.float32), ref_img.affine, ref_img.header),
        path,
    )
    print("Saved:", path)


def pick_best_slice(arr):
    return int(np.argmax(np.sum(np.abs(arr), axis=(0, 1))))


def plot_atlas(arr, valid, title, path, cmap="seismic", symmetric=True):
    z = pick_best_slice(arr)


    if symmetric:
        vals = np.abs(arr[valid])
        vals = vals[vals > 0]
        vmax = np.percentile(vals, 99) if len(vals) else 1
        vmin = -vmax
    else:
        vals = arr[arr > 0]
        vmin = 0
        vmax = np.percentile(vals, 99) if len(vals) else 1


    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    im = ax.imshow(np.rot90(arr[:, :, z]), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f"{title}\nSlice {z}", fontsize=11)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved:", path)


# ============================================================
# CSV CLEANING
# ============================================================


def make_clean_condition_csv(condition):
    df = pd.read_csv(CSV_PATHS[condition])


    if "IXI_ID" not in df.columns:
        raise ValueError(f"{condition}: missing IXI_ID")


    clean = pd.DataFrame()
    clean["condition"] = condition
    clean["case_id"] = df["IXI_ID"].astype(str)


    if condition in ["GLI", "CM"]:
        if "BAD_diff" not in df.columns:
            raise ValueError(f"{condition}: missing BAD_diff")
        clean["delta_bag"] = df["BAD_diff"]
        clean["recovery"] = np.nan
    else:
        if "Delta_BAG_tumor" not in df.columns:
            raise ValueError(f"{condition}: missing Delta_BAG_tumor")
        if "Recovery" not in df.columns:
            raise ValueError(f"{condition}: missing Recovery")
        clean["delta_bag"] = df["Delta_BAG_tumor"]
        clean["recovery"] = df["Recovery"]


    clean["mask_path"] = clean["case_id"].apply(lambda x: get_mask_path(condition, x))
    clean["mask_exists"] = clean["mask_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))


    condition_dir = os.path.join(OUT_DIR, "conditions", condition)
    os.makedirs(condition_dir, exist_ok=True)


    valid = clean[clean["mask_exists"]].drop(columns=["mask_exists"])
    missing = clean[~clean["mask_exists"]]


    valid.to_csv(os.path.join(condition_dir, f"atlas_inputs_{condition}.csv"), index=False)
    missing.to_csv(os.path.join(condition_dir, f"missing_masks_{condition}.csv"), index=False)


    print(f"\n{condition}")
    print("Valid:", len(valid))
    print("Missing:", len(missing))


    return valid


# ============================================================
# ATLAS CORE
# ============================================================


def build_atlas(df, name, include_recovery=True):
    group_dir = os.path.join(OUT_DIR, name)
    os.makedirs(group_dir, exist_ok=True)


    if len(df) == 0:
        print(f"Skipping {name}: no rows.")
        return


    ref_img, first_mask = load_mask(df.iloc[0]["mask_path"])
    shape = first_mask.shape


    count_map = np.zeros(shape, dtype=np.float32)
    mask_sum = np.zeros(shape, dtype=np.float32)
    bias_sum = np.zeros(shape, dtype=np.float32)
    recovery_sum = np.zeros(shape, dtype=np.float32)
    recovery_count = np.zeros(shape, dtype=np.float32)


    skipped = []


    for _, row in df.iterrows():
        try:
            img, mask = load_mask(row["mask_path"])


            if mask.shape != shape:
                skipped.append((row["condition"], row["case_id"], row["mask_path"], f"shape mismatch {mask.shape}"))
                continue


            if not np.allclose(img.affine, ref_img.affine, atol=1e-3):
                skipped.append((row["condition"], row["case_id"], row["mask_path"], "affine mismatch"))
                continue


            delta_bag = float(row["delta_bag"])


            mask_sum += mask
            count_map += mask
            bias_sum += mask * delta_bag


            if include_recovery and not pd.isna(row["recovery"]):
                recovery = float(row["recovery"])
                recovery_sum += mask * recovery
                recovery_count += mask


        except Exception as e:
            skipped.append((row["condition"], row["case_id"], row["mask_path"], str(e)))


    pd.DataFrame(
        skipped,
        columns=["condition", "case_id", "mask_path", "reason"],
    ).to_csv(os.path.join(group_dir, "skipped_cases.csv"), index=False)


    eps = 1e-8
    valid_voxels = count_map >= MIN_COUNT


    tumor_frequency = mask_sum / max(len(df), 1)
    bias_atlas = np.where(valid_voxels, bias_sum / (count_map + eps), 0)


    save_nifti(tumor_frequency, ref_img, os.path.join(group_dir, "tumor_frequency_atlas.nii.gz"))
    save_nifti(count_map, ref_img, os.path.join(group_dir, "tumor_count_map.nii.gz"))
    save_nifti(bias_atlas, ref_img, os.path.join(group_dir, "bias_weighted_delta_bag_atlas.nii.gz"))


    plot_atlas(
        tumor_frequency,
        valid_voxels,
        f"{name}: Tumor Frequency",
        os.path.join(group_dir, "tumor_frequency_atlas.png"),
        cmap="magma",
        symmetric=False,
    )


    plot_atlas(
        bias_atlas,
        valid_voxels,
        f"{name}: Bias-Weighted Atlas\nmean ΔBAG",
        os.path.join(group_dir, "bias_weighted_delta_bag_atlas.png"),
        cmap="seismic",
        symmetric=True,
    )


    recovery_atlas = None


    if include_recovery:
        valid_recovery_voxels = recovery_count >= MIN_COUNT
        recovery_atlas = np.where(valid_recovery_voxels, recovery_sum / (recovery_count + eps), 0)


        save_nifti(
            recovery_atlas,
            ref_img,
            os.path.join(group_dir, "recovery_weighted_atlas.nii.gz"),
        )


        plot_atlas(
            recovery_atlas,
            valid_recovery_voxels,
            f"{name}: Recovery-Weighted Atlas\nmean recovery",
            os.path.join(group_dir, "recovery_weighted_atlas.png"),
            cmap="seismic",
            symmetric=True,
        )


    make_combined(name, group_dir, tumor_frequency, bias_atlas, recovery_atlas, valid_voxels)


def make_combined(name, group_dir, tumor_frequency, bias_atlas, recovery_atlas, valid_voxels):
    if recovery_atlas is None:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
        items = [
            (tumor_frequency, "Tumor Frequency", "magma", False),
            (bias_atlas, "Bias-Weighted Atlas\nmean ΔBAG", "seismic", True),
        ]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
        items = [
            (tumor_frequency, "Tumor Frequency", "magma", False),
            (bias_atlas, "Bias-Weighted Atlas\nmean ΔBAG", "seismic", True),
            (recovery_atlas, "Recovery-Weighted Atlas\nmean recovery", "seismic", True),
        ]


    for ax, (arr, title, cmap, symmetric) in zip(axes, items):
        z = pick_best_slice(arr)


        if symmetric:
            vals = np.abs(arr[valid_voxels])
            vals = vals[vals > 0]
            vmax = np.percentile(vals, 99) if len(vals) else 1
            vmin = -vmax
        else:
            vals = arr[arr > 0]
            vmin = 0
            vmax = np.percentile(vals, 99) if len(vals) else 1


        im = ax.imshow(np.rot90(arr[:, :, z]), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10, pad=3)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


    plt.suptitle(name, fontsize=13, fontweight="bold")
    plt.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.02, wspace=0.05)


    out = os.path.join(group_dir, f"combined_weighted_atlases_{name}.png")
    plt.savefig(out, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved:", out)


# ============================================================
# MAIN
# ============================================================


clean_dfs = {}


for condition in ALL_CONDITIONS:
    clean_dfs[condition] = make_clean_condition_csv(condition)


all_clean = pd.concat(clean_dfs.values(), ignore_index=True)
all_clean.to_csv(os.path.join(OUT_DIR, "all_clean_atlas_inputs.csv"), index=False)


# Individual condition atlases
for condition in ALL_CONDITIONS:
    include_recovery = condition not in ["GLI", "CM"]
    build_atlas(clean_dfs[condition], condition, include_recovery=include_recovery)


# Aggregated group atlases
for group_name, conditions in GROUPS.items():
    group_df = pd.concat([clean_dfs[c] for c in conditions], ignore_index=True)
    include_recovery = group_name != "EXP0_ALL"
    build_atlas(group_df, group_name, include_recovery=include_recovery)


print("\nDONE.")
print("Output:", OUT_DIR)
