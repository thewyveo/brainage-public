#!/bin/bash -l
#SBATCH --job-name=usb_synthseg
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/kozdemir/logs/usb_synthseg_%j.out
#SBATCH --error=/home/kozdemir/logs/usb_synthseg_%j.err

set -xo pipefail

REPO="/home/kozdemir/brainage/lesion_agnostic/exp_1/inpainting/USB_og/USB/assets"

module purge
module load 2023

export FREESURFER_HOME="/home/kozdemir/freesurfer"
export SUBJECTS_DIR="$FREESURFER_HOME/subjects"
export FUNCTIONALS_DIR="$FREESURFER_HOME/sessions"
mkdir -p "$SUBJECTS_DIR" "$FUNCTIONALS_DIR"

mkdir -p /home/kozdemir/logs

set +e
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
FS_SETUP_STATUS=$?
set -e

if [ "$FS_SETUP_STATUS" -ne 0 ]; then
  echo "WARNING: FreeSurfer setup returned $FS_SETUP_STATUS, continuing anyway."
fi

echo "HOST=$(hostname)"
echo "REPO=$REPO"
echo "FREESURFER_HOME=$FREESURFER_HOME"
echo "SUBJECTS_DIR=$SUBJECTS_DIR"
date

which mri_synthseg
mri_synthseg --help | head -20

cd "$REPO"
mkdir -p usb_synthseg_labels

mri_synthseg \
  --i "/home/kozdemir/USB_IXI_rerun/y_p/mris" \
  --o "/home/kozdemir/USB_IXI_synthseg" \
  --robust

find "$REPO/usb_synthseg_labels" -type f -ls

echo "DONE"
date
