#!/bin/bash
#SBATCH --job-name=binary_masks
#SBATCH --output=/home/kozdemir/logs/binary_masks_%j.out
#SBATCH --error=/home/kozdemir/logs/binary_masks_%j.err

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euxo pipefail

mkdir -p /home/kozdemir/logs

cd /home/kozdemir/brainage/lesion_agnostic

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source /home/kozdemir/venvs/brainage/bin/activate

python exp_1/inpainting/USB_og/USB/assets/binary_mask.py \
    --input_dir /home/kozdemir/IXI_USB_masks/ \
    --output_dir /home/kozdemir/IXI_USB_out_binary
