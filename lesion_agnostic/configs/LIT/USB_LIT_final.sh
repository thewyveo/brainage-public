#!/bin/bash
#SBATCH --job-name=usb_lit
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/kozdemir/logs/usb_lit_%j.out
#SBATCH --error=/home/kozdemir/logs/usb_lit_%j.err

set -euo pipefail

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
    pip install neuroLIT
else
    echo "Using existing neuroLIT venv..."
    source "$VENV_DIR/bin/activate"
fi

T1_DIR="/home/kozdemir/IXI_USB_t1/mris"
MASK_DIR="/home/kozdemir/IXI_USB_masks"
OUTPUT_DIR="/home/kozdemir/USB_LIT"

T1_SUFFIX=".nii.gz"
MASK_SUFFIX="_synthetic_seg.nii.gz"

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

    mask_img="$MASK_DIR/${case_id}${MASK_SUFFIX}"
    case_out="$OUTPUT_DIR/$case_id"

    if [ ! -f "$mask_img" ]; then
        echo "Missing mask:"
        echo "$mask_img"
        continue
    fi

    if [ -d "$case_out" ] && [ "$(find "$case_out" -type f | wc -l)" -gt 0 ]; then
        echo "Already processed:"
        echo "$case_out"
        continue
    fi

    rm -rf "$case_out"
    mkdir -p "$case_out"

    echo "=========================================="
    echo "Processing:"
    echo "$case_id"
    echo "Input:"
    echo "$input_img"
    echo "Mask:"
    echo "$mask_img"
    echo "Output:"
    echo "$case_out"
    echo "=========================================="

    lit-inpainting \
        --input_image "$input_img" \
        --mask_image "$mask_img" \
        --output_directory "$case_out" \
        --dilate 2

done

echo "=========================================="
echo "All done."
echo "=========================================="
