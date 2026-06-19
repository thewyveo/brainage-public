#!/bin/bash
#SBATCH --job-name=bnx_all_prep
#SBATCH --output=/home/kozdemir/logs/bnx_all_prep_%j.out
#SBATCH --error=/home/kozdemir/logs/bnx_all_prep_%j.err
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail
set -x

cd /home/kozdemir/brainage/lesion_agnostic
mkdir -p /home/kozdemir/logs

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source /home/kozdemir/venvs/brainage/bin/activate

export OMP_NUM_THREADS=4
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

RUNS=(
  CM_USB
  GLI_USB
  USB_USB
)

for RUN in "${RUNS[@]}"; do
    echo "=========================================="
    echo "Preprocessing BNX run: $RUN"
    echo "=========================================="

    INPUT_DIR="/home/kozdemir/${RUN}_input/y_h"
    OUTPUT_DIR="/home/kozdemir/BNX_${RUN}_PREP"

    mkdir -p "$OUTPUT_DIR"

    python exp_0/processing/preprocessing/overall/preprocess2.py \
      --profile custom \
      --input-dir "$INPUT_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --mni /home/kozdemir/brainage/data/MNI152_T1_1mm_Brain.nii \
      --workers 8 \
      --refresh-cache \
      --enable-n4 \
      --registration-backend ants_rigid \
      --disable-synthstrip \
      --disable-bet \
      --disable-fast \
      --disable-reorient \
      --disable-robustfov \
      --disable-denoise \
      --disable-resample \
      --disable-crop \
      --disable-normalize-1-99 \
      --disable-zscore-nonzero \
      --disable-save-npy

    echo "Finished: $RUN"
done

echo "All BNX preprocessing runs finished."
