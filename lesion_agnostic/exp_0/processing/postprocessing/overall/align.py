import nibabel as nib
from nilearn.image import resample_to_img

healthy_path = "healthy.nii.gz"
inpainted_path = "inpainted.nii.gz"
out_path = "inpainted_aligned_to_healthy.nii.gz"

healthy_img = nib.load(healthy_path)
inpainted_img = nib.load(inpainted_path)

aligned = resample_to_img(
    source_img=inpainted_img,
    target_img=healthy_img,
    interpolation="continuous",
    force_resample=True,
    copy_header=True,
)

aligned.to_filename(out_path)

print("Saved:", out_path)
print("Healthy shape:", healthy_img.shape)
print("Aligned inpainted shape:", aligned.shape)