#!/bin/bash -l
#SBATCH --job-name=usb_h2p
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/kozdemir/logs/usb_h2p_%j.out
#SBATCH --error=/home/kozdemir/logs/usb_h2p_%j.err

set -exo pipefail

mkdir -p /home/kozdemir/logs

# -----------------------------
# GO TO USB REPO
# -----------------------------
cd /home/kozdemir/brainage/lesion_agnostic/exp_1/inpainting/USB_og/USB

# -----------------------------
# MODULES
# -----------------------------
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# -----------------------------
# ACTIVATE VENV
# -----------------------------
source /home/kozdemir/venvs/usb/bin/activate

# -----------------------------
# PYTHONPATH
# -----------------------------
export PYTHONPATH=.

# -----------------------------
# DEBUG INFO
# -----------------------------
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
which python
python --version
nvidia-smi

# -----------------------------
# RUN USB H2P
# -----------------------------
python scripts/test.py \
    --mode h2p_edit \
    --config_path cfgs/trainer/test/test.yaml

echo "DONE"
date
