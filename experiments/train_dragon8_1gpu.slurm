#!/bin/bash

#SBATCH --job-name=ppg_ft
#SBATCH --nodelist=whitedragon8
#SBATCH --partition=gpu_4090
#SBATCH --nodes 1 #MUST MATCH num_nodes IN TRAINER. DEFAULT IS 1
#SBATCH --ntasks-per-node=1 # MUST MATCH devices IN TRAINER
#SBATCH --gres=gpu:1
#SBATCH --output=%j_train_ppg.out
#SBATCH --error=%j_train_ppg.err
#SBATCH --time=10-00:00:00
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=potassium@sjtu.edu.cn

# use wandb in slurm: https://github.com/wandb/wandb/issues/4034
# pytorch lightning with DDP: https://github.com/Lightning-AI/pytorch-lightning/issues/14781

export SWANLAB_API_KEY=NvH3hL4b9GE5U9gom9itl
srun python train_pockets.py ++experiment.identifier=pep_finetune ++experiment.finetune=True ++experiment.finetune_start="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-new/2026-02-04_22-11_full_edge_bf16_preln_peptide/epoch\=509-step\=642600.ckpt"