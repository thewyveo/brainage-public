#!/bin/bash
#SBATCH --job-name=brats_n4_rigid
#SBATCH --output=/home/kozdemir/logs/brats_n4_rigid_%j.out
#SBATCH --error=/home/kozdemir/logs/brats_n4_rigid_%j.err

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -eo pipefail
set -x

mkdir -p /home/kozdemir/logs

cd /home/kozdemir/brainage/lesion_agnostic

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source /home/kozdemir/venvs/brainage/bin/activate

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8

python /home/kozdemir/brainage/lesion_agnostic/exp_0/processing/preprocessing/overall/cm_prep.py \
  --brats-img-dir /home/kozdemir/BraTS_T1 \
  --brats-seg-dir /home/kozdemir/BraTS_seg \
  --out-img-dir /home/kozdemir/BraTS_T1_n4rigid \
  --out-seg-dir /home/kozdemir/BraTS_seg_rigid \
  --mni /home/kozdemir/brainage/lesion_agnostic/data/MNI152_T1_1mm_Brain.nii \
  --image-filter t1n \
  --seg-filter seg \
  --workers 8
