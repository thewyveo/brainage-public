#!/bin/bash -l
#SBATCH --job-name=usb_dataset
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/kozdemir/logs/usb_dataset_%j.out
#SBATCH --error=/home/kozdemir/logs/usb_dataset_%j.err

set -exo pipefail

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
# CREATE VENV IF MISSING
# -----------------------------
if [ ! -d "/home/kozdemir/venvs/usb" ]; then
    mkdir -p /home/kozdemir/venvs
    python -m venv /home/kozdemir/venvs/usb
fi

# -----------------------------
# ACTIVATE VENV
# -----------------------------
source /home/kozdemir/venvs/usb/bin/activate

# -----------------------------
# INSTALL REQUIREMENTS IF NEEDED
# -----------------------------
pip install --upgrade pip

if [ ! -f "/home/kozdemir/venvs/usb/.requirements_installed" ]; then
    pip install -r requirements.txt
    touch /home/kozdemir/venvs/usb/.requirements_installed
fi

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
# RUN DATASET CREATION
# -----------------------------
python scripts/demo_create_dataset.py \
  --data_config_path cfgs/dataset/test/create_test.yaml \
  --save_path /home/kozdemir/USB_USB_PREP

echo "DONE"
date

