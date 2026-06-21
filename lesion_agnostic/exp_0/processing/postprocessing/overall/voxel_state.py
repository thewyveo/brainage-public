import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt


from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from nibabel.processing import resample_from_to



CSV_PATHS = {
    "GLI_LIT": r"C:\Users\P102179\OneDrive - Amsterdam UMC\Bureaublad\repo\brainage\lesion_agnostic\data\results\ALLPOSTPROCESS\raw\exp1\BNX\GLI\LIT\triplet_raw_BAG.csv",
    "GLI_BID": r"C:\Users\P102179\OneDrive - Amsterdam UMC\Bureaublad\repo\brainage\lesion_agnostic\data\results\ALLPOSTPROCESS\raw\exp1\BNX\GLI\BID\triplet_raw_BAG.csv",
    "CM_LIT": r"C:\Users\P102179\OneDrive - Amsterdam UMC\Bureaublad\repo\brainage\lesion_agnostic\data\results\ALLPOSTPROCESS\raw\exp1\BNX\CM\LIT\triplet_raw_BAG.csv",
    "CM_BID": r"C:\Users\P102179\OneDrive - Amsterdam UMC\Bureaublad\repo\brainage\lesion_agnostic\data\results\ALLPOSTPROCESS\raw\exp1\BNX\CM\BID\triplet_raw_BAG.csv",
}




# ============================================================
# ROOT PATHS
# ============================================================


ROOTS = {
    "GLI": r"\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\gen\IXI_GLI",
    "GLI_LIT": r"\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\gen\GLI_LIT",
    "GLI_BID": r"\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\gen\GLI_BID",


    "CM_SYNTH": r"\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\gen\IXI_CM\synthetic",
    "CM_LIT": r"\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\gen\CM_LIT",
    "CM_BID": r"\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\gen\CM_BID",
}


OUT_DIR = "voxel_state_transition_atlases"
os.makedirs(OUT_DIR, exist_ok=True)




# ============================================================
# PARAMETERS
# ============================================================


# Defines "tumor meaningfully changed this voxel" using |T-H|
TUMOR_CHANGE_PERCENTILE = 99.0


# Defines off-target inpainting change using |I-H|
OFFTARGET_THRESHOLD_SCALE = 0.50


# Minimum fractional reduction required to count as recovered
# 0.25 means |I-H| must be at least 25% lower than |T-H|
MIN_RECOVERY_FRACTION = 0.25


# Ignore tiny sign flips near zero
OVERSHOOT_MIN_SCALE = 0.20


# Minimum number of cases covering a voxel in group maps
MIN_CASE_COUNT = 3


# Resample tumored/inpainted image to healthy grid if needed
RESAMPLE_TO_HEALTHY = True




# ============================================================
# STATE LABELS
# ============================================================


STATE_UNAFFECTED = 0
STATE_RECOVERED = 1
STATE_PERSISTENT = 2
STATE_OVERSHOOT = 3
STATE_OFFTARGET = 4


STATE_NAMES = {
    STATE_UNAFFECTED: "Unaffected",
    STATE_RECOVERED: "Recovered",
    STATE_PERSISTENT: "Persistent residual",
    STATE_OVERSHOOT: "Overshoot",
    STATE_OFFTARGET: "Off-target hallucination",
}


STATE_COLORS = {
    STATE_UNAFFECTED: "#000000",
    STATE_RECOVERED: "#2ecc71",
    STATE_PERSISTENT: "#f39c12",
    STATE_OVERSHOOT: "#9b59b6",
    STATE_OFFTARGET: "#e74c3c",
}




# ============================================================
# PATH HELPERS
# ============================================================


def first_match(pattern):
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None




def find_case_dir(root, ixi_id):
    return first_match(os.path.join(root, f"{ixi_id}*"))




def get_gli_case_dir(ixi_id):
    return find_case_dir(ROOTS["GLI"], ixi_id)




def get_healthy_path(ixi_id):
    gli_case_dir = get_gli_case_dir(ixi_id)


    if gli_case_dir is None:
        return None


    healthy_path = os.path.join(gli_case_dir, "healthy_t1.nii.gz")


    if os.path.exists(healthy_path):
        return healthy_path


    return None




def get_gli_synth_path(ixi_id):
    gli_case_dir = get_gli_case_dir(ixi_id)


    if gli_case_dir is None:
        return None


    synth_path = os.path.join(gli_case_dir, "synthetic_t1.nii.gz")


    if os.path.exists(synth_path):
        return synth_path


    return None




def get_gli_lit_path(ixi_id):
    case_dir = find_case_dir(ROOTS["GLI_LIT"], ixi_id)


    if case_dir is None:
        return None


    path = os.path.join(case_dir, "inpainting_volumes", "inpainting_result.nii.gz")


    if os.path.exists(path):
        return path


    return None




def get_gli_bid_path(ixi_id):
    return first_match(
        os.path.join(ROOTS["GLI_BID"], f"{ixi_id}*brainid_recon.nii.gz")
    )




def get_cm_synth_path(ixi_id):
    return first_match(
        os.path.join(ROOTS["CM_SYNTH"], f"{ixi_id}*_carvemix.nii.gz")
    )




def get_cm_lit_path(ixi_id):
    case_dir = find_case_dir(ROOTS["CM_LIT"], ixi_id)


    if case_dir is None:
        return None


    path = os.path.join(case_dir, "inpainting_volumes", "inpainting_result.nii.gz")


    if os.path.exists(path):
        return path


    return None




def get_cm_bid_path(ixi_id):
    return first_match(
        os.path.join(ROOTS["CM_BID"], f"{ixi_id}*brainid_recon.nii.gz")
    )




def get_paths_for_condition(condition, ixi_id):
    healthy_path = get_healthy_path(ixi_id)


    if condition == "GLI_LIT":
        tumored_path = get_gli_synth_path(ixi_id)
        inpainted_path = get_gli_lit_path(ixi_id)


    elif condition == "GLI_BID":
        tumored_path = get_gli_synth_path(ixi_id)
        inpainted_path = get_gli_bid_path(ixi_id)


    elif condition == "CM_LIT":
        tumored_path = get_cm_synth_path(ixi_id)
        inpainted_path = get_cm_lit_path(ixi_id)


    elif condition == "CM_BID":
        tumored_path = get_cm_synth_path(ixi_id)
        inpainted_path = get_cm_bid_path(ixi_id)


    else:
        raise ValueError(f"Unknown condition: {condition}")


    return healthy_path, tumored_path, inpainted_path




# ============================================================
# IMAGE HELPERS
# ============================================================


def load_img(path):
    img = nib.load(path)
    data = np.squeeze(img.get_fdata()).astype(np.float32)
    return img, data




def same_space(img_a, data_a, img_b, data_b):
    return (
        data_a.shape == data_b.shape
        and np.allclose(img_a.affine, img_b.affine, atol=1e-3)
    )




def resample_to_ref(source_img, ref_img):
    return resample_from_to(source_img, ref_img, order=1)




def robust_norm(x):
    mask = x > 0


    if np.sum(mask) == 0:
        return x.astype(np.float32)


    lo, hi = np.percentile(x[mask], [1, 99])
    x = np.clip(x, lo, hi)


    return ((x - lo) / (hi - lo + 1e-8)).astype(np.float32)




def save_nifti(arr, ref_img, path):
    nib.save(
        nib.Nifti1Image(arr.astype(np.float32), ref_img.affine, ref_img.header),
        path,
    )
    print("Saved:", path)




def pick_best_slice(arr):
    scores = np.sum(arr > 0, axis=(0, 1))


    if np.max(scores) == 0:
        scores = np.sum(np.abs(arr), axis=(0, 1))


    return int(np.argmax(scores))




# ============================================================
# VOXEL STATE CLASSIFICATION
# ============================================================


def classify_voxel_states(healthy, tumored, inpainted):
    healthy_n = robust_norm(healthy)
    tumored_n = robust_norm(tumored)
    inpainted_n = robust_norm(inpainted)


    brain_mask = healthy_n > 0.05


    d_t = tumored_n - healthy_n
    d_i = inpainted_n - healthy_n


    abs_t = np.abs(d_t)
    abs_i = np.abs(d_i)


    if np.sum(brain_mask) == 0:
        raise ValueError("Empty brain mask after normalization.")


    tumor_thr = np.percentile(abs_t[brain_mask], TUMOR_CHANGE_PERCENTILE)
    tumor_thr = max(tumor_thr, 1e-4)


    off_thr = tumor_thr * OFFTARGET_THRESHOLD_SCALE
    over_thr = tumor_thr * OVERSHOOT_MIN_SCALE


    tumor_changed = brain_mask & (abs_t >= tumor_thr)
    inpaint_changed = brain_mask & (abs_i >= off_thr)


    sign_flip = (d_t * d_i) < 0


    overshoot = (
        tumor_changed
        & sign_flip
        & (abs_i >= over_thr)
    )


    recovered = (
        tumor_changed
        & (~overshoot)
        & (abs_i < abs_t * (1.0 - MIN_RECOVERY_FRACTION))
    )


    persistent = (
        tumor_changed
        & (~overshoot)
        & (~recovered)
    )


    off_target = (
        (~tumor_changed)
        & inpaint_changed
    )


    state = np.zeros(healthy.shape, dtype=np.uint8)


    state[recovered] = STATE_RECOVERED
    state[persistent] = STATE_PERSISTENT
    state[overshoot] = STATE_OVERSHOOT
    state[off_target] = STATE_OFFTARGET


    return state, brain_mask.astype(np.float32), healthy_n




# ============================================================
# PLOTTING
# ============================================================


def plot_state_overlay(dominant_state, background, title, out_path):
    z = pick_best_slice(dominant_state)


    color_list = [
        STATE_COLORS[STATE_UNAFFECTED],
        STATE_COLORS[STATE_RECOVERED],
        STATE_COLORS[STATE_PERSISTENT],
        STATE_COLORS[STATE_OVERSHOOT],
        STATE_COLORS[STATE_OFFTARGET],
    ]


    cmap = ListedColormap(color_list)


    overlay = np.ma.masked_where(dominant_state == 0, dominant_state)
    alpha = np.where(dominant_state[:, :, z] == 0, 0.0, 0.82)


    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)


    ax.imshow(np.rot90(background[:, :, z]), cmap="gray")
    ax.imshow(
        np.rot90(overlay[:, :, z]),
        cmap=cmap,
        vmin=0,
        vmax=4,
        alpha=np.rot90(alpha),
        interpolation="nearest",
    )


    ax.set_title(f"{title}\nSlice {z}", fontsize=11)
    ax.axis("off")


    legend_handles = [
        Patch(color=STATE_COLORS[STATE_RECOVERED], label="Recovered"),
        Patch(color=STATE_COLORS[STATE_PERSISTENT], label="Persistent"),
        Patch(color=STATE_COLORS[STATE_OVERSHOOT], label="Overshoot"),
        Patch(color=STATE_COLORS[STATE_OFFTARGET], label="Off-target"),
    ]


    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=False,
        fontsize=8,
    )


    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()


    print("Saved:", out_path)




def plot_probability_map(prob_map, background, title, out_path, cmap="magma"):
    z = pick_best_slice(prob_map)


    vals = prob_map[prob_map > 0]
    vmax = np.percentile(vals, 99) if len(vals) else 1.0


    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)


    ax.imshow(np.rot90(background[:, :, z]), cmap="gray")
    im = ax.imshow(
        np.rot90(prob_map[:, :, z]),
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        alpha=0.75,
    )


    ax.set_title(f"{title}\nSlice {z}", fontsize=11)
    ax.axis("off")


    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()


    print("Saved:", out_path)




# ============================================================
# CLEAN INPUT CSV CREATION
# ============================================================


def make_clean_condition_csv(condition):
    df = pd.read_csv(CSV_PATHS[condition])


    if "IXI_ID" not in df.columns:
        raise ValueError(f"{condition}: CSV missing IXI_ID")


    clean_rows = []


    for _, row in df.iterrows():
        ixi_id = str(row["IXI_ID"])


        healthy_path, tumored_path, inpainted_path = get_paths_for_condition(
            condition,
            ixi_id,
        )


        clean_rows.append({
            "condition": condition,
            "case_id": ixi_id,
            "healthy_path": healthy_path,
            "tumored_path": tumored_path,
            "inpainted_path": inpainted_path,
            "healthy_exists": isinstance(healthy_path, str) and os.path.exists(healthy_path),
            "tumored_exists": isinstance(tumored_path, str) and os.path.exists(tumored_path),
            "inpainted_exists": isinstance(inpainted_path, str) and os.path.exists(inpainted_path),
        })


    clean = pd.DataFrame(clean_rows)


    condition_dir = os.path.join(OUT_DIR, "conditions", condition)
    os.makedirs(condition_dir, exist_ok=True)


    valid = clean[
        clean["healthy_exists"]
        & clean["tumored_exists"]
        & clean["inpainted_exists"]
    ].drop(columns=[
        "healthy_exists",
        "tumored_exists",
        "inpainted_exists",
    ])


    missing = clean[
        ~(
            clean["healthy_exists"]
            & clean["tumored_exists"]
            & clean["inpainted_exists"]
        )
    ]


    valid_csv = os.path.join(condition_dir, f"voxel_state_inputs_{condition}.csv")
    missing_csv = os.path.join(condition_dir, f"missing_voxel_state_inputs_{condition}.csv")


    valid.to_csv(valid_csv, index=False)
    missing.to_csv(missing_csv, index=False)


    print(f"\n{condition}")
    print("Valid:", len(valid))
    print("Missing:", len(missing))
    print("Input CSV:", valid_csv)
    print("Missing log:", missing_csv)


    return valid




# ============================================================
# ATLAS BUILDING
# ============================================================


def build_state_atlas(df, name):
    out_dir = os.path.join(OUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)


    if len(df) == 0:
        print(f"Skipping {name}: no valid rows.")
        return


    state_counts = None
    brain_count = None
    mean_healthy_sum = None
    ref_img = None
    shape = None


    skipped = []


    for idx, row in df.iterrows():
        case_id = row["case_id"]
        condition = row["condition"]


        try:
            h_img, h = load_img(row["healthy_path"])
            t_img, t = load_img(row["tumored_path"])
            i_img, i = load_img(row["inpainted_path"])


            if RESAMPLE_TO_HEALTHY:
                if not same_space(h_img, h, t_img, t):
                    t_img = resample_to_ref(t_img, h_img)
                    t = np.squeeze(t_img.get_fdata()).astype(np.float32)


                if not same_space(h_img, h, i_img, i):
                    i_img = resample_to_ref(i_img, h_img)
                    i = np.squeeze(i_img.get_fdata()).astype(np.float32)


            if h.shape != t.shape or h.shape != i.shape:
                skipped.append((condition, case_id, "shape mismatch after resampling"))
                continue


            if ref_img is None:
                ref_img = h_img
                shape = h.shape


                state_counts = np.zeros((5,) + shape, dtype=np.float32)
                brain_count = np.zeros(shape, dtype=np.float32)
                mean_healthy_sum = np.zeros(shape, dtype=np.float32)


            if h.shape != shape:
                skipped.append((condition, case_id, f"shape mismatch {h.shape} vs {shape}"))
                continue


            if not np.allclose(h_img.affine, ref_img.affine, atol=1e-3):
                skipped.append((condition, case_id, "healthy affine mismatch"))
                continue


            state, brain_mask, healthy_n = classify_voxel_states(h, t, i)


            for s in range(5):
                state_counts[s] += (state == s).astype(np.float32)


            brain_count += brain_mask
            mean_healthy_sum += healthy_n


        except Exception as e:
            skipped.append((condition, case_id, str(e)))
            continue


        if (idx + 1) % 25 == 0:
            print(f"{name}: processed {idx + 1}/{len(df)}")


    skipped_df = pd.DataFrame(
        skipped,
        columns=["condition", "case_id", "reason"],
    )


    skipped_df.to_csv(
        os.path.join(out_dir, "skipped_cases.csv"),
        index=False,
    )


    if ref_img is None:
        print(f"Skipping {name}: no usable images.")
        return


    eps = 1e-8
    usable = brain_count >= MIN_CASE_COUNT


    mean_healthy = mean_healthy_sum / (brain_count + eps)


    prob_maps = {}


    for s in range(1, 5):
        prob_maps[s] = np.where(
            usable,
            state_counts[s] / (brain_count + eps),
            0,
        )


    stacked = np.stack(
        [prob_maps[s] for s in range(1, 5)],
        axis=0,
    )


    max_prob = np.max(stacked, axis=0)
    dominant_state = np.argmax(stacked, axis=0).astype(np.uint8) + 1
    dominant_state = np.where(
        usable & (max_prob > 0),
        dominant_state,
        0,
    ).astype(np.uint8)


    save_nifti(
        mean_healthy,
        ref_img,
        os.path.join(out_dir, "mean_healthy_background.nii.gz"),
    )


    save_nifti(
        dominant_state,
        ref_img,
        os.path.join(out_dir, "dominant_voxel_state_atlas.nii.gz"),
    )


    save_nifti(
        max_prob,
        ref_img,
        os.path.join(out_dir, "dominant_state_probability.nii.gz"),
    )


    save_nifti(
        prob_maps[STATE_RECOVERED],
        ref_img,
        os.path.join(out_dir, "recovered_probability.nii.gz"),
    )


    save_nifti(
        prob_maps[STATE_PERSISTENT],
        ref_img,
        os.path.join(out_dir, "persistent_residual_probability.nii.gz"),
    )


    save_nifti(
        prob_maps[STATE_OVERSHOOT],
        ref_img,
        os.path.join(out_dir, "overshoot_probability.nii.gz"),
    )


    save_nifti(
        prob_maps[STATE_OFFTARGET],
        ref_img,
        os.path.join(out_dir, "offtarget_hallucination_probability.nii.gz"),
    )


    plot_state_overlay(
        dominant_state,
        mean_healthy,
        f"{name}: Dominant Voxel-State Transition",
        os.path.join(out_dir, "dominant_voxel_state_atlas.png"),
    )


    plot_probability_map(
        prob_maps[STATE_RECOVERED],
        mean_healthy,
        f"{name}: Recovery Probability",
        os.path.join(out_dir, "recovered_probability.png"),
        cmap="Greens",
    )


    plot_probability_map(
        prob_maps[STATE_PERSISTENT],
        mean_healthy,
        f"{name}: Persistent Residual Probability",
        os.path.join(out_dir, "persistent_residual_probability.png"),
        cmap="Oranges",
    )


    plot_probability_map(
        prob_maps[STATE_OVERSHOOT],
        mean_healthy,
        f"{name}: Overshoot Probability",
        os.path.join(out_dir, "overshoot_probability.png"),
        cmap="Purples",
    )


    plot_probability_map(
        prob_maps[STATE_OFFTARGET],
        mean_healthy,
        f"{name}: Off-target Hallucination Probability",
        os.path.join(out_dir, "offtarget_hallucination_probability.png"),
        cmap="Reds",
    )


    make_combined_panel(
        name,
        out_dir,
        dominant_state,
        prob_maps,
        mean_healthy,
    )


    print(f"\n{name}: done")
    print("Skipped:", len(skipped_df))




def make_combined_panel(name, out_dir, dominant_state, prob_maps, background):
    z = pick_best_slice(dominant_state)


    color_list = [
        STATE_COLORS[STATE_UNAFFECTED],
        STATE_COLORS[STATE_RECOVERED],
        STATE_COLORS[STATE_PERSISTENT],
        STATE_COLORS[STATE_OVERSHOOT],
        STATE_COLORS[STATE_OFFTARGET],
    ]


    state_cmap = ListedColormap(color_list)


    fig, axes = plt.subplots(1, 5, figsize=(17, 4), dpi=300)


    overlay = np.ma.masked_where(dominant_state == 0, dominant_state)
    alpha = np.where(dominant_state[:, :, z] == 0, 0.0, 0.82)


    axes[0].imshow(np.rot90(background[:, :, z]), cmap="gray")
    axes[0].imshow(
        np.rot90(overlay[:, :, z]),
        cmap=state_cmap,
        vmin=0,
        vmax=4,
        alpha=np.rot90(alpha),
        interpolation="nearest",
    )
    axes[0].set_title("Dominant state", fontsize=10)
    axes[0].axis("off")


    panels = [
        (STATE_RECOVERED, "Recovered", "Greens"),
        (STATE_PERSISTENT, "Persistent", "Oranges"),
        (STATE_OVERSHOOT, "Overshoot", "Purples"),
        (STATE_OFFTARGET, "Off-target", "Reds"),
    ]


    for ax, (state_id, title, cmap) in zip(axes[1:], panels):
        p = prob_maps[state_id]
        vals = p[p > 0]
        vmax = np.percentile(vals, 99) if len(vals) else 1.0


        ax.imshow(np.rot90(background[:, :, z]), cmap="gray")
        im = ax.imshow(
            np.rot90(p[:, :, z]),
            cmap=cmap,
            vmin=0,
            vmax=vmax,
            alpha=0.75,
        )


        ax.set_title(title, fontsize=10)
        ax.axis("off")


        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


    legend_handles = [
        Patch(color=STATE_COLORS[STATE_RECOVERED], label="Recovered"),
        Patch(color=STATE_COLORS[STATE_PERSISTENT], label="Persistent"),
        Patch(color=STATE_COLORS[STATE_OVERSHOOT], label="Overshoot"),
        Patch(color=STATE_COLORS[STATE_OFFTARGET], label="Off-target"),
    ]


    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
    )


    plt.suptitle(
        f"{name}: Voxel-State Transition Atlas",
        fontsize=14,
        fontweight="bold",
    )


    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.82,
        bottom=0.18,
        wspace=0.04,
    )


    out_path = os.path.join(
        out_dir,
        f"combined_voxel_state_transition_{name}.png",
    )


    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()


    print("Saved:", out_path)




# ============================================================
# MAIN
# ============================================================


CONDITIONS = [
    "GLI_LIT",
    "GLI_BID",
    "CM_LIT",
    "CM_BID",
]


GROUPS = {
    "EXP1_ALL": ["GLI_LIT", "GLI_BID", "CM_LIT", "CM_BID"],
    "LIT_ALL": ["GLI_LIT", "CM_LIT"],
    "BID_ALL": ["GLI_BID", "CM_BID"],
    "GLI_INPAINT_ALL": ["GLI_LIT", "GLI_BID"],
    "CM_INPAINT_ALL": ["CM_LIT", "CM_BID"],
}


clean_dfs = {}


for condition in CONDITIONS:
    clean_dfs[condition] = make_clean_condition_csv(condition)


all_clean = pd.concat(clean_dfs.values(), ignore_index=True)


all_clean.to_csv(
    os.path.join(OUT_DIR, "all_voxel_state_inputs.csv"),
    index=False,
)


for condition in CONDITIONS:
    build_state_atlas(clean_dfs[condition], condition)


for group_name, condition_list in GROUPS.items():
    group_df = pd.concat(
        [clean_dfs[c] for c in condition_list],
        ignore_index=True,
    )


    build_state_atlas(group_df, group_name)


print("\nDONE.")
print("Output directory:", OUT_DIR)
