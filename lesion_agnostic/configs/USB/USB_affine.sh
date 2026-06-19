#!/bin/bash
#SBATCH --job-name=usb_pair_affines
#SBATCH --output=logs/usb_pair_affines_%j.out
#SBATCH --error=logs/usb_pair_affines_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

set -e
set -x

cd /home/kozdemir/

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source ~/venvs/brainage/bin/activate

mkdir -p logs

python3 usb_affine_pair.py \
  --paired-healthy-dir /home/kozdemir/new_usb_inputs_paired/healthy \
  --old-affine-dir /home/kozdemir/USB_IXI_input/IXI_T1_affine \
  --new-affine-dir /home/kozdemir/new_usb_inputs_paired/affines \
  --mode copy
