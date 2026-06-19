#!/bin/bash
#SBATCH --job-name=cm_ixi_rerun
#SBATCH --output=logs/cm_ixi_rerun_%j.out
#SBATCH --error=logs/cm_ixi_rerun_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1

set -e
set -x

cd /home/kozdemir/

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/venvs/brainage/bin/activate

mkdir -p logs

python3 carvemix_final.py \
  --healthy-dir "/home/kozdemir/IXI_n4_rigid" \
  --library-dir "/home/kozdemir/CM_lib/CM_lib" \
  --gligan-pairings-csv "/home/kozdemir/successful_ixi_brats_pairings.csv" \
  --output-dir "/home/kozdemir/data/outputs/IXI_CM_rerun" \
  --target-total 550 \
  --extra-random-if-under 50 \
  --seed 42
