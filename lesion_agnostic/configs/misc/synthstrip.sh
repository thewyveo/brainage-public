#!/bin/bash
#SBATCH --job-name=synthstrip
#SBATCH --output=/home/kozdemir/logs/synthstrip_%j.out
#SBATCH --error=/home/kozdemir/logs/synthstrip_%j.err
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00

set -eo pipefail
set -x

mkdir -p /home/kozdemir/logs

export FREESURFER_HOME=/home/kozdemir/freesurfer
export SUBJECTS_DIR=/home/kozdemir/freesurfer/subjects
export PATH="$FREESURFER_HOME/bin:$PATH"

INPUT_DIR="/home/kozdemir/data/raw/IXI-T1"
OUTPUT_DIR="/home/kozdemir/IXI-Stripped"
SYNTHSTRIP="$FREESURFER_HOME/bin/mri_synthstrip"

mkdir -p "$OUTPUT_DIR"

for f in "$INPUT_DIR"/*.nii.gz; do
    base=$(basename "$f" .nii.gz)

    "$SYNTHSTRIP" \
        -i "$f" \
        -o "$OUTPUT_DIR/${base}_brain.nii.gz" \
        -g
done
