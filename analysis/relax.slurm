#!/bin/bash

#SBATCH --job-name=relax
#SBATCH --nodelist=whitedragon2
#SBATCH --partition=gpu_3090
#SBATCH --nodes 1 #MUST MATCH num_nodes IN TRAINER. DEFAULT IS 1
#SBATCH --ntasks-per-node=1 # MUST MATCH devices IN TRAINER
#SBATCH --gres=gpu:1
#SBATCH --output=%j_relax.out
#SBATCH --error=%j_relax.err
#SBATCH --time=10-00:00:00
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=potassium@sjtu.edu.cn

# python copy_ligand_back.py --model diffab --mode h3
# python relax_pipeline.py --mode pep_baseline --path /home/kechen/antibody_design/diffab/results/peptide_standard/codesign_single --save_path /home/kechen/antibody_design/testset/0217/diffab_pep_h3_dg.pkl --subdir_name H_CDR3
# python relax_pipeline.py --mode ligand_diffab_h3 --save_path /home/kechen/antibody_design/testset/0217/diffab_lig_h3_vina.pkl --subdir_name H_CDR3_with_ligand
PATH1="/home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-24_10-08_pep_finetune_continue/epoch=1391-step=174000/pocket/run_2026-03-25_13-44-24"
PATH2="/home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-22_00-27_ligand_finetune/epoch=1403-step=175500/pocket/run_2026-03-25_13-44-24"
PATH3="/home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-24_09-57_ligand_addpro_finetune/epoch=764-step=463590/pocket/run_2026-03-26_15-21-09"

PATH_CDR="/home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-22_00-27_ligand_finetune/epoch=1403-step=175500/CDR_H3/run_2026-04-03_12-25-26"
nocipa_PATH="/home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-25_15-29_ligand_finetune_nocipa/epoch=1385-step=500346/pocket/run_2026-04-07_13-25-42"

python relax_pipeline.py --mode ligand_ppg --path $nocipa_PATH
python relax_pipeline.py --mode pep_ppg --path $nocipa_PATH







