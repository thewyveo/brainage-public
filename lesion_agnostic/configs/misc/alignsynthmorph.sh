#!/bin/bash
#SBATCH --job-name=synthmorph_all
#SBATCH --output=/home/kozdemir/logs/synthmorph_all_%j.out
#SBATCH --error=/home/kozdemir/logs/synthmorph_all_%j.err
#SBATCH --partition=gpu_mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -x

REF="/home/kozdemir/fig1/healthy.nii.gz"
IN_DIR="/home/kozdemir/fig1"
OUT_DIR="/home/kozdemir/makefig1_synthmorph"
MODE="affine"

mkdir -p /home/kozdemir/logs
mkdir -p "$OUT_DIR"

export FS_FREESURFERENV_NO_OUTPUT=1
export FREESURFER_HOME=$HOME/freesurfer
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"

echo "Reference: $REF"
echo "Input dir: $IN_DIR"
echo "Output dir: $OUT_DIR"
echo "Mode: $MODE"

for MOV in "$IN_DIR"/*.nii.gz "$IN_DIR"/*.nii; do
    [ -e "$MOV" ] || continue

    BASE=$(basename "$MOV")
    BASE="${BASE%.nii.gz}"
    BASE="${BASE%.nii}"

    OUT="$OUT_DIR/${BASE}_aligned.nii.gz"

    if [ "$MOV" = "$REF" ]; then
        echo "[REF COPY] $MOV"
        cp "$MOV" "$OUT"
        continue
    fi

    echo "=========================================="
    echo "[ALIGN] $MOV"
    echo "[OUT]   $OUT"
    echo "=========================================="

    mri_synthmorph \
      -m "$MODE" \
      -o "$OUT" \
      "$MOV" \
      "$REF"

done

echo "DONE"
ls -lh "$OUT_DIR"
