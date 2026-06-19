#!/bin/bash
#SBATCH --job-name=mni_map
#SBATCH --output=/home/kozdemir/logs/mni_map_%j.out
#SBATCH --error=/home/kozdemir/logs/mni_map_%j.err

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
set -x

mkdir -p /home/kozdemir/logs

cd /home/kozdemir/brainage/lesion_agnostic

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

source /home/kozdemir/venvs/brainage/bin/activate

export OMP_NUM_THREADS=4
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

python exp_1/inpainting/USB_og/USB/scripts/mni_mapping.py \
    --input_path /home/kozdemir/USB_USB_input/T1 \
    --label_path /home/kozdemir/USB_USB_input/USB_IXI_synthseg \
    --new_affine_path /home/kozdemir/USB_USB_affine \
    --workers 8
