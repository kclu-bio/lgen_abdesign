#!/bin/bash

#SBATCH --job-name=ppg
#SBATCH --nodelist=whitedragon1
#SBATCH --partition=gpu_3090
#SBATCH --nodes 1 #MUST MATCH num_nodes IN TRAINER. DEFAULT IS 1
#SBATCH --ntasks-per-node=4 # MUST MATCH devices IN TRAINER
#SBATCH --gres=gpu:4
#SBATCH --output=%j_train_ppg.out
#SBATCH --error=%j_train_ppg.err
#SBATCH --time=10-00:00:00
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=potassium@sjtu.edu.cn

# use wandb in slurm: https://github.com/wandb/wandb/issues/4034
# pytorch lightning with DDP: https://github.com/Lightning-AI/pytorch-lightning/issues/14781

export SWANLAB_API_KEY=NvH3hL4b9GE5U9gom9itl
srun python train_pockets.py ++experiment.identifier=aux_loss0.25_from11090141 ++experiment.num_devices=4

