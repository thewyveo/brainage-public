#!/bin/bash
#SBATCH --job-name=joos_gli_usb_prep
#SBATCH --output=/home/kozdemir/logs/joos_gli_usb_prep_%j.out
#SBATCH --error=/home/kozdemir/logs/joos_gli_usb_prep_%j.err

#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euxo pipefail

mkdir -p /home/kozdemir/logs

cd /home/kozdemir/brainage/lesion_agnostic

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0

# FSL env
source /home/kozdemir/miniforge3/etc/profile.d/conda.sh
conda activate fsl

export FSLDIR="/home/kozdemir/miniforge3/envs/fsl"
export FSLOUTPUTTYPE=NIFTI_GZ
export PATH="$FSLDIR/bin:$PATH"

which fslreorient2std
which robustfov

# Brainage env
source /home/kozdemir/venvs/brainage/bin/activate

# Re-add FSL after venv activation
export FSLDIR="/home/kozdemir/miniforge3/envs/fsl"
export FSLOUTPUTTYPE=NIFTI_GZ
export PATH="$FSLDIR/bin:$PATH"

which python
which fslreorient2std
which robustfov

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8

/home/kozdemir/venvs/brainage/bin/python \
/home/kozdemir/brainage/lesion_agnostic/exp_0/processing/preprocessing/overall/preprocess.py \
  --profile joos \
  --input-dir /home/kozdemir/GLI_USB/y_h/mris \
  --output-dir /home/kozdemir/JOOS_GLI_USB_PREP \
  --mni /home/kozdemir/brainage/lesion_agnostic/data/MNI152_T1_1mm_Brain.nii \
  --workers 8 \
  --disable-synthstrip \
  --registration-backend ants_affine
