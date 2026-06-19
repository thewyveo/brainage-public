#!/bin/bash
#SBATCH --job-name=lit_gligan
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=lit_gligan_%j.out
#SBATCH --error=lit_gligan_%j.err

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

INPUT_DIR="/home/kozdemir/gligan_n4_rigid_after585"
MASK_DIR="/home/kozdemir/BraTS_seg_n4_rigid"
OUTPUT_DIR="/home/kozdemir/GLI_LIT"

mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "Starting LIT inpainting"
echo "========================================="
echo "Input dir:  $INPUT_DIR"
echo "Mask dir:   $MASK_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "========================================="

NUM_INPUTS=$(find "$INPUT_DIR" -maxdepth 1 -name "*.nii.gz" | wc -l)

echo "Found $NUM_INPUTS input images"

if [ "$NUM_INPUTS" -eq 0 ]; then
    echo "ERROR: no input images found"
    exit 1
fi

for input_img in "$INPUT_DIR"/*.nii.gz; do

    base=$(basename "$input_img")

    # Example:
    # IXI002-Guys-0828__GLI-02196-105.nii.gz
    #
    # Extract:
    # GLI-02196-105

    gli_id=$(echo "$base" | grep -o 'GLI-[0-9]\+-[0-9]\+')

    if [ -z "$gli_id" ]; then
        echo "Could not extract GLI ID from:"
        echo "$base"
        continue
    fi

    mask_img="$MASK_DIR/BraTS-${gli_id}_seg_rigid.nii.gz"

    if [ ! -f "$mask_img" ]; then
        echo "Missing mask:"
        echo "$mask_img"
        continue
    fi

    case_id="${base%.nii.gz}"
    case_out="$OUTPUT_DIR/$case_id"

    # Resume support
    if [ -d "$case_out" ]; then
        echo "Already processed:"
        echo "$case_out"
        continue
    fi

    echo "========================================="
    echo "Processing:"
    echo "$base"
    echo "Using mask:"
    echo "$(basename "$mask_img")"
    echo "========================================="

    lit-inpainting \
      --input_image "$input_img" \
      --mask_image "$mask_img" \
      --output_directory "$case_out" \
      --dilate 2

done

echo "========================================="
echo "All done."
echo "========================================="
