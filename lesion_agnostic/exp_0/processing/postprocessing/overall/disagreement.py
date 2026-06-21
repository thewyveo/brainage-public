import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt




# ============================================================
# EDIT THIS IF NEEDED
# ============================================================


OUT_DIR = "weighted_atlases_aggregated"


# folders produced by previous script
GLI_DIR = os.path.join(OUT_DIR, "GLI")
CM_DIR = os.path.join(OUT_DIR, "CM")
GLI_LIT_DIR = os.path.join(OUT_DIR, "GLI_LIT")
GLI_BID_DIR = os.path.join(OUT_DIR, "GLI_BID")
CM_LIT_DIR = os.path.join(OUT_DIR, "CM_LIT")
CM_BID_DIR = os.path.join(OUT_DIR, "CM_BID")


DISAGREE_DIR = os.path.join(OUT_DIR, "disagreement_maps")
os.makedirs(DISAGREE_DIR, exist_ok=True)




# ============================================================
# HELPERS
# ============================================================


def load_nii(path):
    img = nib.load(path)
    return img, img.get_fdata().astype(np.float32)




def save_nii(arr, ref_img, path):
    nib.save(
        nib.Nifti1Image(arr.astype(np.float32), ref_img.affine, ref_img.header),
        path,
    )
    print("Saved:", path)




def check_same_space(name_a, img_a, arr_a, name_b, img_b, arr_b):
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"Shape mismatch: {name_a} {arr_a.shape} vs {name_b} {arr_b.shape}")


    if not np.allclose(img_a.affine, img_b.affine, atol=1e-3):
        raise ValueError(f"Affine mismatch: {name_a} vs {name_b}")




def pick_best_slice(arr):
    return int(np.argmax(np.sum(np.abs(arr), axis=(0, 1))))




def plot_disagreement(arr, title, out_path, cmap="seismic"):
    z = pick_best_slice(arr)


    nonzero = np.abs(arr[arr != 0])
    vmax = np.percentile(nonzero, 99) if len(nonzero) else 1.0


    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)


    im = ax.imshow(
        np.rot90(arr[:, :, z]),
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )


    ax.set_title(f"{title}\nSlice {z}", fontsize=11)
    ax.axis("off")


    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)


    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()


    print("Saved:", out_path)




# ============================================================
# 1) INPAINTER DISAGREEMENT: LIT - BID
# ============================================================


# GLI: LIT vs BID
gli_lit_img, gli_lit_rec = load_nii(os.path.join(GLI_LIT_DIR, "recovery_weighted_atlas.nii.gz"))
gli_bid_img, gli_bid_rec = load_nii(os.path.join(GLI_BID_DIR, "recovery_weighted_atlas.nii.gz"))


check_same_space("GLI_LIT", gli_lit_img, gli_lit_rec, "GLI_BID", gli_bid_img, gli_bid_rec)


gli_lit_minus_bid = gli_lit_rec - gli_bid_rec


save_nii(
    gli_lit_minus_bid,
    gli_lit_img,
    os.path.join(DISAGREE_DIR, "gli_lit_minus_bid_recovery_disagreement.nii.gz"),
)


plot_disagreement(
    gli_lit_minus_bid,
    "GLI Inpainter Disagreement\nRecovery LIT − Recovery BID",
    os.path.join(DISAGREE_DIR, "gli_lit_minus_bid_recovery_disagreement.png"),
)


# CM: LIT vs BID
cm_lit_img, cm_lit_rec = load_nii(os.path.join(CM_LIT_DIR, "recovery_weighted_atlas.nii.gz"))
cm_bid_img, cm_bid_rec = load_nii(os.path.join(CM_BID_DIR, "recovery_weighted_atlas.nii.gz"))


check_same_space("CM_LIT", cm_lit_img, cm_lit_rec, "CM_BID", cm_bid_img, cm_bid_rec)


cm_lit_minus_bid = cm_lit_rec - cm_bid_rec


save_nii(
    cm_lit_minus_bid,
    cm_lit_img,
    os.path.join(DISAGREE_DIR, "cm_lit_minus_bid_recovery_disagreement.nii.gz"),
)


plot_disagreement(
    cm_lit_minus_bid,
    "CM Inpainter Disagreement\nRecovery LIT − Recovery BID",
    os.path.join(DISAGREE_DIR, "cm_lit_minus_bid_recovery_disagreement.png"),
)




# ============================================================
# 2) GENERATOR DISAGREEMENT: GLI - CM
# ============================================================


gli_img, gli_bias = load_nii(os.path.join(GLI_DIR, "bias_weighted_delta_bag_atlas.nii.gz"))
cm_img, cm_bias = load_nii(os.path.join(CM_DIR, "bias_weighted_delta_bag_atlas.nii.gz"))


check_same_space("GLI", gli_img, gli_bias, "CM", cm_img, cm_bias)


gli_minus_cm_bias = gli_bias - cm_bias


save_nii(
    gli_minus_cm_bias,
    gli_img,
    os.path.join(DISAGREE_DIR, "gli_minus_cm_bias_disagreement.nii.gz"),
)


plot_disagreement(
    gli_minus_cm_bias,
    "Generator Disagreement\nΔBAG GLI − ΔBAG CM",
    os.path.join(DISAGREE_DIR, "gli_minus_cm_bias_disagreement.png"),
)




# ============================================================
# 3) COMBINED POSTER FIGURE
# ============================================================


maps = [
    (
        gli_lit_minus_bid,
        "GLI: LIT − BID\nRecovery",
        "Red = LIT better",
    ),
    (
        cm_lit_minus_bid,
        "CM: LIT − BID\nRecovery",
        "Red = LIT better",
    ),
    (
        gli_minus_cm_bias,
        "GLI − CM\nΔBAG",
        "Red = GLI stronger aging shift",
    ),
]


fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)


for ax, (arr, title, subtitle) in zip(axes, maps):
    z = pick_best_slice(arr)


    nonzero = np.abs(arr[arr != 0])
    vmax = np.percentile(nonzero, 99) if len(nonzero) else 1.0


    im = ax.imshow(
        np.rot90(arr[:, :, z]),
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
    )


    ax.set_title(f"{title}\n{subtitle}", fontsize=9, pad=3)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


plt.subplots_adjust(
    left=0.02,
    right=0.98,
    top=0.86,
    bottom=0.02,
    wspace=0.06,
)


combined_path = os.path.join(DISAGREE_DIR, "combined_disagreement_maps.png")
plt.savefig(combined_path, bbox_inches="tight", dpi=300)
plt.close()


print("Saved:", combined_path)
print("\nDONE.")
