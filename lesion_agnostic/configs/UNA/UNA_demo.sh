#!/bin/bash
#SBATCH --job-name=una_gli10
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/kozdemir/logs/una_gli10_%j.out
#SBATCH --error=/home/kozdemir/logs/una_gli10_%j.err

set -euxo pipefail

mkdir -p /home/kozdemir/logs

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

UNA_DIR=/home/kozdemir/brainage/lesion_agnostic/exp_1/inpainting/UNA
VENV=/home/kozdemir/venvs/una
INPUT_ROOT=/home/kozdemir/UNA_GLI_DEMO10_INPUT
OUTPUT_ROOT=/home/kozdemir/UNA_GLI_DEMO10_RESULTS
WEIGHTS=$UNA_DIR/assets/una.pth
SCRIPT=$UNA_DIR/infer.py

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

[ -d "$INPUT_ROOT" ] || { echo "MISSING INPUT_ROOT: $INPUT_ROOT"; exit 1; }
[ -f "$WEIGHTS" ] || { echo "MISSING WEIGHTS: $WEIGHTS"; exit 1; }
[ -f "$SCRIPT" ] || { echo "MISSING SCRIPT: $SCRIPT"; exit 1; }

python "$SCRIPT" \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --ckp-path "$WEIGHTS" \
  --model-cfg test.yaml \
  --gen-cfg test.yaml \
  --limit 10
