#!/bin/bash
#SBATCH --job-name=brainid_gligan
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=brainid_gligan_%j.out
#SBATCH --error=brainid_gligan_%j.err

set -euo pipefail

cd /home/kozdemir/brainage/lesion_agnostic/exp_1/inpainting/brain_id/Brain-ID

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source /home/kozdemir/venvs/brainid/bin/activate

export PYTHONPATH=.

INPUT_DIR="/home/kozdemir/gligan_generated_mris/gligan_generated_mris"
OUTPUT_DIR="/home/kozdemir/brainid_gligan"
CHECKPOINT="/home/kozdemir/brainage/lesion_agnostic/exp_1/inpainting/brain_id/Brain-ID/assets/brain_id_pretrained.pth"

mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "Starting Brain-ID reconstruction batch"
echo "========================================="
echo "Input dir:      $INPUT_DIR"
echo "Output dir:     $OUTPUT_DIR"
echo "Checkpoint:     $CHECKPOINT"
echo "========================================="

for input_img in "$INPUT_DIR"/*.nii.gz; do

    base=$(basename "$input_img")
    case_id="${base%.nii.gz}"

    expected_out="${OUTPUT_DIR}/${case_id}_brainid_recon.nii.gz"

    # Resume support
    if [ -f "$expected_out" ]; then
        echo "Already processed, skipping:"
        echo "$expected_out"
        continue
    fi

    echo "========================================="
    echo "Processing:"
    echo "$base"
    echo "========================================="

    python scripts/infer.py \
      --input "$input_img" \
      --checkpoint "$CHECKPOINT" \
      --out_dir "$OUTPUT_DIR" \
      --device cuda:0

done

echo "========================================="
echo "All Brain-ID reconstructions completed."
echo "========================================="
