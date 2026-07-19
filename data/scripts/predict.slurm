#!/bin/bash

#SBATCH --job-name=protenix
#SBATCH --nodelist=whitedragon8
#SBATCH --partition=gpu_4090
#SBATCH --nodes 1 #MUST MATCH num_nodes IN TRAINER. DEFAULT IS 1
#SBATCH --ntasks-per-node=1 # MUST MATCH devices IN TRAINER
#SBATCH --gres=gpu:1
#SBATCH --output=%j_ptx.out
#SBATCH --error=%j_ptx.err
#SBATCH --time=10-00:00:00
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=potassium@sjtu.edu.cn

export LAYERNORM_TYPE=torch
protenix pred -i /home/kechen/peppocketgen/dataset/sabdab/Ab-sugar/ab_sugar.json -o /home/kechen/peppocketgen/dataset/sabdab/Ab-sugar/protenix -s 101 -n protenix_base_20250630_v1.0.0
protenix pred -i /home/kechen/peppocketgen/dataset/sabdab/Ab-hapten/ab_hapten.json -o /home/kechen/peppocketgen/dataset/sabdab/Ab-hapten/protenix -s 101 -n protenix_base_20250630_v1.0.0
#protenix pred -i /home/kechen/peppocketgen/dataset/sabdab/Ab-pep/ab_pep.json -o /home/kechen/peppocketgen/dataset/sabdab/Ab-pep/protenix -s 101 -n protenix_base_20250630_v1.0.0

