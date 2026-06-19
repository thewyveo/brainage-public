#!/bin/bash
#SBATCH --job-name=synthsr
#SBATCH --output=/home/kozdemir/logs/synthsr_%j.out
#SBATCH --error=/home/kozdemir/logs/synthsr_%j.err
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00

set -x

mkdir -p /home/kozdemir/logs

echo "START"
hostname
date

module purge

export FREESURFER_HOME=/home/kozdemir/freesurfer
export SUBJECTS_DIR=$FREESURFER_HOME/subjects
export FUNCTIONALS_DIR=$FREESURFER_HOME/sessions

echo "Sourcing FreeSurfer..."
set +u
source $FREESURFER_HOME/SetUpFreeSurfer.sh
set -u

echo "Checking command..."
which mri_synthsr || true
ls -l $FREESURFER_HOME/bin/mri_synthsr || true

echo "Running SynthSR..."
mri_synthsr \
  --i /home/kozdemir/GLI10 \
  --o /home/kozdemir/GLI_SR_DEMO10_RERUN

echo "DONE"
date
