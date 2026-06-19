#!/bin/bash -l
#SBATCH --job-name=carvemix_gen
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/kozdemir/carvemix_gen_%j.out
#SBATCH --error=/home/kozdemir/carvemix_gen_%j.err

set -exo pipefail

# -----------------------------
# GO TO PROJECT ROOT
# -----------------------------
cd /home/kozdemir/brainage

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
export PYTHONPATH=/home/kozdemir/brainage

# -----------------------------
# DEBUG INFO
# -----------------------------
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
which python
python --version
nvidia-smi || true
date

# -----------------------------
# RUN CARVEMIX GENERATION
# Change these paths before running if needed.
# -----------------------------
python /home/kozdemir/brainage/lesion_agnostic/exp_0/synth_lesion_generator/CarveMix/carvemix_heteregeneous.py \
  --healthy-dir /home/kozdemir/IXI_n4_rigid \
  --library-dir /home/kozdemir/CM_lib \
  --output-dir /home/kozdemir/CM_output \
  --name-filter rigid \
  --seed 42 \
  --max-placement-tries 100 \
  --min-inside-ratio 0.98

echo "DONE"
date
