#!/bin/bash
#SBATCH --job-name=usblit
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/kozdemir/logs/usblit_%j.out
#SBATCH --error=/home/kozdemir/logs/usblit_%j.err

set -uo pipefail

cd /home/kozdemir/brainage

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

VENV_DIR="/home/kozdemir/venvs/neurolit"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating neuroLIT venv..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install neuroLIT nibabel numpy
else
    echo "Using existing neuroLIT venv..."
    source "$VENV_DIR/bin/activate"
fi

T1_DIR="/home/kozdemir/USB_IXI_rerun/y_p/mris"
MASK_DIR="/home/kozdemir/IXI_USB_binarymasks"
OUTPUT_DIR="/home/kozdemir/USB_LIT"

T1_SUFFIX=".nii.gz"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Starting LIT inpainting"
echo "T1 dir:     $T1_DIR"
echo "Mask dir:   $MASK_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="

NUM_INPUTS=$(find "$T1_DIR" -type f -name "*${T1_SUFFIX}" | wc -l)
echo "Found $NUM_INPUTS T1 files"

if [ "$NUM_INPUTS" -eq 0 ]; then
    echo "ERROR: no T1 files found"
    exit 1
fi

find "$T1_DIR" -type f -name "*${T1_SUFFIX}" | sort | while read -r input_img; do

    filename=$(basename "$input_img")
    case_id="${filename%$T1_SUFFIX}"

    brats_id=$(echo "$case_id" | grep -o 'BraTS-GLI-[0-9]\{5\}-[0-9]\{3\}' | head -n 1)

    if [ -z "$brats_id" ]; then
        echo "Could not extract BraTS ID from:"
        echo "$case_id"
        continue
    fi

    mask_img="$MASK_DIR/${brats_id}_seg_rigid.nii.gz"
    case_out="$OUTPUT_DIR/$case_id"
    tmp_mask="$case_out/mask_for_lit_input_geometry.nii.gz"

    if [ ! -f "$mask_img" ]; then
        echo "Missing mask:"
        echo "$mask_img"
        continue
    fi

    if [ -d "$case_out" ] && [ "$(find "$case_out" -type f ! -name 'mask_for_lit_input_geometry.nii.gz' | wc -l)" -gt 0 ]; then
        echo "Already processed:"
        echo "$case_out"
        continue
    fi

    rm -rf "$case_out"
    mkdir -p "$case_out"

    echo "=========================================="
    echo "Processing:"
    echo "$case_id"
    echo "Input: $input_img"
    echo "Mask:  $mask_img"
    echo "Out:   $case_out"
    echo "=========================================="

    python - "$input_img" "$mask_img" "$tmp_mask" <<'PY'
import sys
import nibabel as nib
import numpy as np

input_path, mask_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

inp = nib.load(input_path)
msk = nib.load(mask_path)

inp_data = inp.get_fdata()
msk_data = msk.get_fdata()

mask = (msk_data > 0).astype(np.uint8)

print("Input shape:", inp_data.shape)
print("Mask shape: ", mask.shape)
print("Raw mask nonzero:", int(mask.sum()), "total:", int(mask.size))

if mask.shape != inp_data.shape:
    raise RuntimeError(
        f"Shape mismatch: input shape {inp_data.shape}, mask shape {mask.shape}. "
        "This mask is not in the USB image grid."
    )

nz = int(mask.sum())
total = int(mask.size)

if nz == 0:
    raise RuntimeError("BAD MASK: all zero before LIT")

if nz == total:
    raise RuntimeError("BAD MASK: all nonzero before LIT")

out = nib.Nifti1Image(mask, inp.affine, inp.header.copy())
out.header.set_data_dtype(np.uint8)
nib.save(out, out_path)

check = nib.load(out_path).get_fdata()
check_nz = int(np.count_nonzero(check))
check_total = int(check.size)

print("Saved LIT mask:", out_path)
print("Saved mask nonzero:", check_nz, "zero:", check_total - check_nz, "total:", check_total)

if check_nz == 0 or check_nz == check_total:
    raise RuntimeError("BAD SAVED MASK")
PY

    prep_status=$?

    if [ "$prep_status" -ne 0 ]; then
        echo "Mask preparation failed, skipping:"
        echo "$case_id"
        rm -rf "$case_out"
        continue
    fi

    lit-inpainting \
        --input_image "$input_img" \
        --mask_image "$tmp_mask" \
        --output_directory "$case_out" \
        --dilate 2

    lit_status=$?

    if [ "$lit_status" -ne 0 ]; then
        echo "LIT failed, skipping:"
        echo "$case_id"
        continue
    fi

    echo "Finished:"
    echo "$case_id"

done

echo "=========================================="
echo "All done."
echo "=========================================="
