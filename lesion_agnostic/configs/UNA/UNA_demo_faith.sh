#!/bin/bash
#SBATCH --job-name=una_official
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/kozdemir/logs/una_official_%j.out
#SBATCH --error=/home/kozdemir/logs/una_official_%j.err

set -euxo pipefail

mkdir -p /home/kozdemir/logs

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

UNA_DIR=/home/kozdemir/brainage/lesion_agnostic/exp_1/inpainting/UNA
VENV=/home/kozdemir/venvs/una

cd "$UNA_DIR"

if [ ! -d "$VENV" ]; then
    python -m venv "$VENV"
fi

source "$VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [ ! -f "$VENV/.una_requirements_installed" ]; then
    pip install -r requirements.txt
    touch "$VENV/.una_requirements_installed"
fi

export PYTHONPATH="$UNA_DIR"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8

echo "=== DATASET CHECK ==="

ls /home/kozdemir/una_gli10/test.txt
ls /home/kozdemir/una_gli10/T1 | head
ls /home/kozdemir/una_gli10/synth_flip2orig | head

echo "=== CUDA CHECK ==="

python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY

echo "=== STARTING UNA TEST ==="

python scripts/test.py
