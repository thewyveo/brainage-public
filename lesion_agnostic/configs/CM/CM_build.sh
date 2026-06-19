#!/bin/bash -l
#SBATCH --job-name=build_tumor_lib
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/kozdemir/logs/build_tumor_lib_%j.out
#SBATCH --error=/home/kozdemir/logs/build_tumor_lib_%j.err

set -exo pipefail

mkdir -p /home/kozdemir/logs

cd /home/kozdemir/brainage

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source /home/kozdemir/venvs/usb/bin/activate

export PYTHONPATH=/home/kozdemir/brainage

python /home/kozdemir/brainage/lesion_agnostic/exp_0/synth_lesion_generator/CarveMix/build.py \
  --brats-img-dir /home/kozdemir/BraTS_T1_n4_rigid \
  --brats-seg-dir /home/kozdemir/BraTS_seg_n4_rigid \
  --output-dir /home/kozdemir/CM_lib \
  --mask-type whole \
  --image-filter rigid \
  --seg-filter rigid

echo "DONE"
date
