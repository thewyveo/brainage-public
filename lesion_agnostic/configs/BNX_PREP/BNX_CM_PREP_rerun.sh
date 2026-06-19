#!/bin/bash
#SBATCH --job-name=bnx_cm_prep
#SBATCH --output=/home/kozdemir/logs/bnx_cm_prep_%j.out
#SBATCH --error=/home/kozdemir/logs/bnx_cm_prep_%j.err

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
set -x

cd /home/kozdemir/brainage/lesion_agnostic

mkdir -p /home/kozdemir/logs

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source /home/kozdemir/venvs/brainage/bin/activate

# 8 parallel images × 4 threads each = 32 cores total
export OMP_NUM_THREADS=4
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

python exp_0/processing/preprocessing/overall/preprocess2.py \
    --profile custom \
    --input-dir /home/kozdemir/IXI_CM_t1 \
    --output-dir /home/kozdemir/BNX_CM_PREP \
    --mni /home/kozdemir/brainage/data/MNI152_T1_1mm_Brain.nii \
    --workers 4 \
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
