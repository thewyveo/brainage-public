import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nibabel.orientations import aff2axcodes
from nibabel.processing import resample_from_to
from scipy.ndimage import sobel


# =========================
# EDIT PATHS
# =========================
healthy_path = r"\\vumc.nl\Onderzoek\s4e-gpfs2\rath-research-01\Research\neuroRT\students\KayraOzdemir\gen\IXI_CM\healthy\IXI013-HH-1212-T1_brain_n4_rigid__BraTS-GLI-02608-102.nii.gz"
usb_path = r"C:\Users\P102179\Downloads\usbbid_aligned_axis0flip.nii.gz"

out_dir = "usb_alignment_diagnostic_synthmorph_leftrightflip"
os.makedirs(out_dir, exist_ok=True)


# =========================
# LOAD
# =========================
h_img = nib.load(healthy_path)
u_img = nib.load(usb_path)


h = h_img.get_fdata().astype(np.float32)
u = u_img.get_fdata().astype(np.float32)


# =========================
# RESAMPLE USB TO HEALTHY GRID
# =========================
u_aligned_img = resample_from_to(u_img, h_img, order=1)
u_aligned = u_aligned_img.get_fdata().astype(np.float32)


nib.save(u_aligned_img, os.path.join(out_dir, "usb_aligned_to_healthy.nii.gz"))


# =========================
# NORMALIZATION
# =========================
def robust_norm(x):
    mask = x > 0
    if mask.sum() == 0:
        return x
    lo, hi = np.percentile(x[mask], [1, 99])
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-8)


h_n = robust_norm(h)
u_n = robust_norm(u)
u_a_n = robust_norm(u_aligned)


# =========================
# CHOOSE SLICE
# =========================
z = h_n.shape[2] // 2


# If tumor/USB issue is lower/higher, manually override:
# z = 90


# For raw USB, choose its own middle slice
z_usb = u_n.shape[2] // 2


# =========================
# EDGE MAPS
# =========================
def edge2d(x2d):
    gx = sobel(x2d, axis=0)
    gy = sobel(x2d, axis=1)
    e = np.sqrt(gx**2 + gy**2)
    if e.max() > 0:
        e = e / e.max()
    return e


h_slice = h_n[:, :, z]
u_a_slice = u_a_n[:, :, z]


h_edge = edge2d(h_slice)
u_edge = edge2d(u_a_slice)


edge_mismatch = u_edge - h_edge
diff = u_a_n - h_n


vmax_diff = np.percentile(np.abs(diff[h_n > 0.05]), 99)


# =========================
# METADATA TEXT
# =========================
meta_text = (
    "Reference / Healthy\n"
    f"shape: {h_img.shape}\n"
    f"orientation: {aff2axcodes(h_img.affine)}\n"
    f"affine:\n{np.array2string(h_img.affine, precision=1, suppress_small=True)}\n\n"
    "USB (SynthMorph + LR flip)\n"
    f"shape: {u_img.shape}\n"
    f"orientation: {aff2axcodes(u_img.affine)}\n"
    f"affine:\n{np.array2string(u_img.affine, precision=1, suppress_small=True)}\n\n"
    "USB resampled to healthy grid\n"
    f"shape: {u_aligned_img.shape}\n"
    f"orientation: {aff2axcodes(u_aligned_img.affine)}"
)


# =========================
# PLOT
# =========================
fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=300)


axes[0, 0].imshow(np.rot90(h_n[:, :, z]), cmap="gray")
axes[0, 0].set_title("Healthy reference")
axes[0, 0].axis("off")


axes[0, 1].imshow(np.rot90(u_n[:, :, z_usb]), cmap="gray")
axes[0, 1].set_title("USB + SynthMorph + LR flip grid")
axes[0, 1].axis("off")


axes[0, 2].imshow(np.rot90(u_a_n[:, :, z]), cmap="gray")
axes[0, 2].set_title("USB resampled to healthy grid")
axes[0, 2].axis("off")


axes[1, 0].imshow(np.rot90(h_slice), cmap="gray")
axes[1, 0].imshow(np.rot90(h_edge), cmap="Greens", alpha=0.45)
axes[1, 0].imshow(np.rot90(u_edge), cmap="Reds", alpha=0.45)
axes[1, 0].set_title("Edge overlay\nGreen=healthy, Red=USB")
axes[1, 0].axis("off")


im1 = axes[1, 1].imshow(
    np.rot90(diff[:, :, z]),
    cmap="seismic",
    vmin=-vmax_diff,
    vmax=vmax_diff,
)
axes[1, 1].set_title("Residual difference\nUSB aligned − Healthy")
axes[1, 1].axis("off")
plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)


axes[1, 2].axis("off")
axes[1, 2].text(
    0,
    1,
    meta_text,
    va="top",
    ha="left",
    fontsize=7,
    family="monospace",
)


plt.suptitle("USB Alignment / Orientation Diagnostic", fontsize=15, fontweight="bold")
plt.tight_layout()


out_png = os.path.join(out_dir, "usb_alignment_diagnostic.png")
plt.savefig(out_png, bbox_inches="tight", dpi=300)
plt.close()


print("Saved:", out_png)
print("Saved aligned USB:", os.path.join(out_dir, "usb_aligned_to_healthy.nii.gz"))
