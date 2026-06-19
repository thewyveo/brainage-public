#!/bin/bash
#SBATCH --job-name=filter_cm
#SBATCH --output=filter_cm_%j.out
#SBATCH --error=filter_cm_%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

set -e
set -x

cd /home/kozdemir/brainage/lesion_agnostic

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

python3 /home/kozdemir/brainage/lesion_agnostic/filtercm.py \
  --gligan-dir /home/kozdemir/gligan_n4_rigid \
  --carvemix-dir /home/kozdemir/CM_output/synthetic \
  --metadata-dir /home/kozdemir/CM_output/metadata \
  --out-dir /home/kozdemir/CM_filtered_against_GliGAN \
  --mode copy
