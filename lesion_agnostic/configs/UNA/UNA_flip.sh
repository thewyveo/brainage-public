#!/bin/bash
#SBATCH --job-name=una_flipreg
#SBATCH --output=/home/kozdemir/logs/una_flipreg_%j.out
#SBATCH --error=/home/kozdemir/logs/una_flipreg_%j.err

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1

#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euxo pipefail

mkdir -p /home/kozdemir/logs

cd /home/kozdemir

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source /home/kozdemir/venvs/brainage/bin/activate

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8

python /home/kozdemir/brainage/lesion_agnostic/exp_1/inpainting/UNA/flip.py \
  --input-dir /home/kozdemir/GLI10 \
  --output-dir /home/kozdemir/GLI10_flip_UNA
