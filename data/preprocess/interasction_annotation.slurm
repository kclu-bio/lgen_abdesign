#!/bin/bash

#SBATCH --job-name=ppg
#SBATCH --nodelist=whitedragon6
#SBATCH --partition=cpu
#SBATCH --nodes 1 
#SBATCH --ntasks-per-node=8
#SBATCH --output=%j_int.out
#SBATCH --error=%j_int.err
#SBATCH --time=10-00:00:00
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=potassium@sjtu.edu.cn

python interaction_annotation.py --raw_dir /home/kechen/peppocketgen/dataset/propedia/merged_complex --processed_dir /home/kechen/peppocketgen/dataset/processed_training/merged_propedia_5.5 --save_dir /home/kechen/peppocketgen/dataset/processed_training/merged_propedia_5.5/inter_charge --n_jobs 8 --chunksize 20 --maxtasksperchild 20 --task_timeout 300 --mode peptide