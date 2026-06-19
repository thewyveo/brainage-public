#!/bin/bash
#SBATCH --job-name=cm_gligan_pairs
#SBATCH --output=cm_gligan_pairs_%j.out
#SBATCH --error=cm_gligan_pairs_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1

set -e
set -x

cd /home/kozdemir/brainage/lesion_agnostic

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

VENV_DIR=/home/kozdemir/venvs/brainage

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install numpy scipy nibabel

python -c "import numpy, scipy, nibabel; print('ENV OK')"

python /home/kozdemir/brainage/lesion_agnostic/exp_0/synth_lesion_generator/CarveMix/carvemix_heterogeneous_gliganmatched.py \
  --pairs-csv /home/kozdemir/CM_filtered_against_GliGAN/gligan_pairs.csv \
  --healthy-dir /home/kozdemir/IXI_n4_rigid \
  --library-dir /home/kozdemir/CM_lib \
  --output-dir /home/kozdemir/CM_GliGAN_matched \
  --seed 42 \
  --max-placement-tries 300 \
  --min-inside-ratio 0.98 \
  --skip-existing
