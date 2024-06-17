#!/bin/bash
#SBATCH --job-name=vit
#SBATCH --time=72:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
module load cuda12.1/toolkit/12.1
module load cuDNN/cuda12.1/9.1.0.70
source /var/scratch/mdr317/miniconda3/bin/activate
conda activate
cd /var/scratch/mdr317/thesis

echo "ViT experiment 10 epochs some original hyperparameters, but with LR schedule, initial lr 0.0003"
python /var/scratch/mdr317/thesis/MainVIT.py
