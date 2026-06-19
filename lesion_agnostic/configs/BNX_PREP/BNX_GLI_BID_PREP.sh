#!/bin/bash
#SBATCH --job-name=bnxt_prep_gli_bid
#SBATCH --output=/home/kozdemir/logs/banext_prep_gli_bid_%j.out
#SBATCH --error=/home/kozdemir/logs/banext_prep_gli_bid_%j.err

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

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
    --input-dir /home/kozdemir/brainid_gligan \
    --output-dir /home/kozdemir/brainid_gli_banextprepped \
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
