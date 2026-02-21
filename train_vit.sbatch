#!/bin/bash
#SBATCH --job-name=mito_convnext
#SBATCH --output=logs/mito_convnext_%j.out
#SBATCH --error=logs/mito_convnext_%j.err
#SBATCH --partition=train
#SBATCH --nodelist=lyn-gpu-[03-05]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00

set -euo pipefail

cd /home/pwojcik/PIPNet
mkdir -p logs

PY=/home/pwojcik/.conda/envs/dinov2/bin/python

echo "Node: $(hostname)"
nvidia-smi || true

echo "PY path is: $PY"
ls -l "$PY"
"$PY" -V
"$PY" -c "import sys; print('sys.executable:', sys.executable)"

"$PY" main.py \
  --dataset grayscale_mito \
  --epochs 16 \
  --epochs_pretrain 8 \
  --batch_size 16 \
  --freeze_epochs 10 \
  --log_dir ./runs/paper_runs/mito_convnext_tiny_2img_hf03_cn_5 \
  --image_size 448 \
  --num_features 256 \
  --image_size_ds 224 \
  --net convnext_tiny_13 \
  --gpu_ids 0