import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# =========================
# EDIT THESE PATHS
# =========================
healthy_path = "healthy.nii.gz"
tumored_path = "tumored.nii.gz"
inpainted_path = "inpainted_aligned_to_healthy.nii.gz"

out_dir = "anatomical_change_map"
os.makedirs(out_dir, exist_ok=True)


# =========================
# LOAD NIFTI
# =========================
healthy_img = nib.load(healthy_path)
tumored_img = nib.load(tumored_path)
inpainted_img = nib.load(inpainted_path)

healthy = healthy_img.get_fdata().astype(np.float32)
tumored = tumored_img.get_fdata().astype(np.float32)
inpainted = inpainted_img.get_fdata().astype(np.float32)


# =========================
# CHECK ALIGNMENT
# =========================
if healthy.shape != tumored.shape or healthy.shape != inpainted.shape:
    raise ValueError(
        f"Shape mismatch:\n"
        f"Healthy: {healthy.shape}\n"
        f"Tumored: {tumored.shape}\n"
        f"Inpainted: {inpainted.shape}"
    )

if not np.allclose(healthy_img.affine, tumored_img.affine, atol=1e-3):
    raise ValueError("Healthy and tumored affines do not match.")

if not np.allclose(healthy_img.affine, inpainted_img.affine, atol=1e-3):
    raise ValueError("Healthy and inpainted affines do not match.")


# =========================
# NORMALIZE EACH MRI
# robust percentile normalization
# =========================
def robust_norm(x):
    mask = x > 0
    lo, hi = np.percentile(x[mask], [1, 99])
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-8)


healthy_n = robust_norm(healthy)
tumored_n = robust_norm(tumored)
inpainted_n = robust_norm(inpainted)


# =========================
# DIFFERENCE MAPS
# =========================
tumor_effect = tumored_n - healthy_n
inpainting_effect = inpainted_n - tumored_n
residual_effect = inpainted_n - healthy_n


# =========================
# PICK SLICE
# choose slice with strongest tumor effect
# =========================
slice_scores = np.sum(np.abs(tumor_effect), axis=(0, 1))
z = int(np.argmax(slice_scores))

print(f"Selected axial slice: {z}")


# =========================
# PLOT HELPERS
# =========================
def show_mri(ax, img, title):
    ax.imshow(np.rot90(img[:, :, z]), cmap="gray")
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def show_overlay(ax, base, diff, title, vmax=None):
    if vmax is None:
        vmax = np.percentile(np.abs(diff), 99)

    ax.imshow(np.rot90(base[:, :, z]), cmap="gray")
    ax.imshow(
        np.rot90(diff[:, :, z]),
        cmap="seismic",
        alpha=0.65,
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=11)
    ax.axis("off")


# same color scale for all difference maps
global_vmax = np.percentile(
    np.abs(np.concatenate([
        tumor_effect.flatten(),
        inpainting_effect.flatten(),
        residual_effect.flatten()
    ])),
    99
)


# =========================
# MAKE FIGURE
# =========================
fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=300)

show_mri(axes[0, 0], healthy_n, "Healthy MRI")
show_mri(axes[0, 1], tumored_n, "Synthetic Tumor MRI")
show_mri(axes[0, 2], inpainted_n, "Inpainted MRI")

show_overlay(
    axes[1, 0],
    healthy_n,
    tumor_effect,
    "Tumor insertion\nTumored − Healthy",
    vmax=global_vmax,
)

show_overlay(
    axes[1, 1],
    tumored_n,
    inpainting_effect,
    "Inpainting correction\nInpainted − Tumored",
    vmax=global_vmax,
)

show_overlay(
    axes[1, 2],
    healthy_n,
    residual_effect,
    "Residual deviation\nInpainted − Healthy",
    vmax=global_vmax,
)

plt.subplots_adjust(
    left=0.02,
    right=0.98,
    top=0.92,
    bottom=0.05,
    wspace=-0.5,
    hspace=0.15
)

out_png = os.path.join(out_dir, "anatomical_change_map.png")
plt.savefig(out_png, bbox_inches="tight", dpi=300)
plt.close()

print(f"Saved figure to: {out_png}")


# =========================
# SAVE DIFFERENCE NIFTIS TOO
# =========================
nib.save(
    nib.Nifti1Image(tumor_effect, healthy_img.affine, healthy_img.header),
    os.path.join(out_dir, "tumor_effect_tumored_minus_healthy.nii.gz")
)

nib.save(
    nib.Nifti1Image(inpainting_effect, healthy_img.affine, healthy_img.header),
    os.path.join(out_dir, "inpainting_effect_inpainted_minus_tumored.nii.gz")
)

nib.save(
    nib.Nifti1Image(residual_effect, healthy_img.affine, healthy_img.header),
    os.path.join(out_dir, "residual_effect_inpainted_minus_healthy.nii.gz")
)

print("Saved NIfTI difference maps.")

# =========================
# RECOVERY MAP
# =========================

# Absolute deviation before inpainting
tumor_abs_deviation = np.abs(tumored_n - healthy_n)

# Absolute deviation after inpainting
inpaint_abs_deviation = np.abs(inpainted_n - healthy_n)

# Positive = inpainting moved closer to healthy
# Negative = inpainting moved farther from healthy
recovery_map = tumor_abs_deviation - inpaint_abs_deviation

# Choose same slice as before, or choose strongest recovery slice
# z = int(np.argmax(np.sum(np.abs(recovery_map), axis=(0, 1))))

recovery_vmax = np.percentile(np.abs(recovery_map), 99)

fig, ax = plt.subplots(figsize=(5, 5), dpi=300)

ax.imshow(np.rot90(healthy_n[:, :, z]), cmap="gray")
im = ax.imshow(
    np.rot90(recovery_map[:, :, z]),
    cmap="seismic",
    alpha=0.65,
    vmin=-recovery_vmax,
    vmax=recovery_vmax,
)

ax.set_title(
    "Inpainting Recovery\n|Tumored − Healthy| − |Inpainted − Healthy|",
    fontsize=11,
)
ax.axis("off")

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Recovery score", fontsize=9)

plt.tight_layout()

out_recovery_png = os.path.join(out_dir, "inpainting_recovery_map.png")
plt.savefig(out_recovery_png, bbox_inches="tight", dpi=300)
plt.close()

print(f"Saved recovery map to: {out_recovery_png}")

# Save NIfTI too
nib.save(
    nib.Nifti1Image(recovery_map, healthy_img.affine, healthy_img.header),
    os.path.join(out_dir, "inpainting_recovery_map.nii.gz")
)

# =========================
# EXTRA POSTER FIGURES
# =========================
from scipy import ndimage


# -------------------------
# 1) AUTOMATIC LESION ROI
# -------------------------
brain_mask = healthy_n > 0.05

# approximate lesion from strongest tumor-induced absolute changes
thr = np.percentile(tumor_abs_deviation[brain_mask], 99.5)
lesion_mask = tumor_abs_deviation > thr

# keep largest connected component
labeled, nlab = ndimage.label(lesion_mask)
if nlab > 0:
    sizes = ndimage.sum(lesion_mask, labeled, range(1, nlab + 1))
    largest = int(np.argmax(sizes) + 1)
    lesion_mask = labeled == largest

# dilate slightly to include surrounding reconstruction region
lesion_roi = ndimage.binary_dilation(lesion_mask, iterations=4)
outside_roi = brain_mask & (~lesion_roi)


# -------------------------
# 2) RECOVERY DECOMPOSITION
# -------------------------
recovered_component = np.minimum(tumor_abs_deviation, np.maximum(recovery_map, 0))
worsened_component = np.maximum(-recovery_map, 0)
remaining_residual = inpaint_abs_deviation

vmax_dec = np.percentile(
    np.concatenate([
        recovered_component[brain_mask].ravel(),
        worsened_component[brain_mask].ravel(),
        remaining_residual[brain_mask].ravel()
    ]),
    99
)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)

maps = [
    (recovered_component, "Recovered change\nmin(|T−H|, positive recovery)"),
    (worsened_component, "New/worsened change\nnegative recovery"),
    (remaining_residual, "Remaining residual\n|I−H|"),
]

for ax, (m, title) in zip(axes, maps):
    ax.imshow(np.rot90(healthy_n[:, :, z]), cmap="gray")
    im = ax.imshow(
        np.rot90(m[:, :, z]),
        cmap="magma",
        alpha=0.70,
        vmin=0,
        vmax=vmax_dec,
    )
    ax.contour(
        np.rot90(lesion_roi[:, :, z]),
        levels=[0.5],
        colors="cyan",
        linewidths=0.8,
    )
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
out_decomp = os.path.join(out_dir, "recovery_decomposition.png")
plt.savefig(out_decomp, bbox_inches="tight", dpi=300)
plt.close()
print("Saved:", out_decomp)


# -------------------------
# 3) OUTSIDE-LESION ARTIFACT MAP
# -------------------------
outside_residual = np.where(outside_roi, np.abs(inpainted_n - healthy_n), 0)

fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
ax.imshow(np.rot90(healthy_n[:, :, z]), cmap="gray")
im = ax.imshow(
    np.rot90(outside_residual[:, :, z]),
    cmap="inferno",
    alpha=0.70,
    vmin=0,
    vmax=np.percentile(outside_residual[outside_roi], 99),
)
ax.contour(
    np.rot90(lesion_roi[:, :, z]),
    levels=[0.5],
    colors="cyan",
    linewidths=0.9,
)
ax.set_title("Off-target reconstruction change\n|Inpainted − Healthy| outside lesion ROI", fontsize=10)
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()

out_artifact = os.path.join(out_dir, "outside_lesion_artifact_map.png")
plt.savefig(out_artifact, bbox_inches="tight", dpi=300)
plt.close()
print("Saved:", out_artifact)


# -------------------------
# 4) INTENSITY PROFILE THROUGH LESION
# -------------------------
coords = np.argwhere(lesion_roi[:, :, z])
if coords.size > 0:
    cx, cy = coords.mean(axis=0).astype(int)

    # horizontal line through lesion center
    x_line = np.arange(healthy_n.shape[0])

    h_prof = healthy_n[x_line, cy, z]
    t_prof = tumored_n[x_line, cy, z]
    i_prof = inpainted_n[x_line, cy, z]

    fig, ax = plt.subplots(figsize=(7, 3), dpi=300)

    ax.plot(h_prof, label="Healthy")
    ax.plot(t_prof, label="Tumored")
    ax.plot(i_prof, label="Inpainted")

    ax.axvline(cx, linestyle="--", linewidth=1)
    ax.set_title("Intensity profile through lesion center", fontsize=11)
    ax.set_xlabel("Voxel position")
    ax.set_ylabel("Normalized intensity")
    ax.legend(frameon=False)
    plt.tight_layout()

    out_profile = os.path.join(out_dir, "lesion_intensity_profile.png")
    plt.savefig(out_profile, bbox_inches="tight", dpi=300)
    plt.close()
    print("Saved:", out_profile)


# -------------------------
# 5) VOXEL RECOVERY SCATTER
# -------------------------
sample_mask = lesion_roi | ndimage.binary_dilation(lesion_roi, iterations=8)
x = tumor_abs_deviation[sample_mask]
y = inpaint_abs_deviation[sample_mask]

# sample if too many points
if len(x) > 8000:
    idx = np.random.choice(len(x), 8000, replace=False)
    x = x[idx]
    y = y[idx]

fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=300)
ax.scatter(x, y, s=3, alpha=0.25)

maxv = max(np.percentile(x, 99), np.percentile(y, 99))
ax.plot([0, maxv], [0, maxv], linestyle="--", linewidth=1)

ax.set_xlim(0, maxv)
ax.set_ylim(0, maxv)
ax.set_xlabel("|Tumored − Healthy|")
ax.set_ylabel("|Inpainted − Healthy|")
ax.set_title("Voxel-wise recovery\nbelow diagonal = closer to healthy", fontsize=10)

plt.tight_layout()
out_scatter = os.path.join(out_dir, "voxel_recovery_scatter.png")
plt.savefig(out_scatter, bbox_inches="tight", dpi=300)
plt.close()
print("Saved:", out_scatter)


# -------------------------
# 6) SIMPLE NUMERIC SUMMARY
# -------------------------
mean_before = np.mean(tumor_abs_deviation[lesion_roi])
mean_after = np.mean(inpaint_abs_deviation[lesion_roi])
mean_recovery = np.mean(recovery_map[lesion_roi])

outside_after = np.mean(np.abs(inpainted_n[outside_roi] - healthy_n[outside_roi]))

print("\n=== ROI SUMMARY ===")
print(f"Mean lesion deviation before inpainting: {mean_before:.4f}")
print(f"Mean lesion deviation after inpainting:  {mean_after:.4f}")
print(f"Mean lesion recovery:                  {mean_recovery:.4f}")
print(f"Mean outside-lesion residual:           {outside_after:.4f}")