#!/bin/bash
#SBATCH --job-name=gli_bid_rerun
#SBATCH --partition=gpu_a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/kozdemir/logs/gli_bid_rerun_%j.out
#SBATCH --error=/home/kozdemir/logs/gli_bid_rerun_%j.err

set -euo pipefail

echo "=========================================="
echo "Brain-ID batch inference"
echo "=========================================="

cd /home/kozdemir/brainage/lesion_agnostic/exp_1/inpainting/brain_id/Brain-ID

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source /home/kozdemir/venvs/brainid/bin/activate

export PYTHONPATH=.

INPUT_DIR="/home/kozdemir/IXI_GLI_finalale"
OUTPUT_DIR="/home/kozdemir/GLI_BID_finalale"

CHECKPOINT="/home/kozdemir/brainage/lesion_agnostic/exp_1/inpainting/brain_id/Brain-ID/assets/brain_id_pretrained.pth"

echo "Input dir:      $INPUT_DIR"
echo "Output dir:     $OUTPUT_DIR"
echo "Checkpoint:     $CHECKPOINT"

mkdir -p "$OUTPUT_DIR"
mkdir -p /home/kozdemir/logs

python scripts/infer2.py \
    --input-dir "$INPUT_DIR" \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --image-filter synthetic_t1 \
    --recursive

echo "=========================================="
echo "Finished."
echo "=========================================="
