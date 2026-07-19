#!/bin/bash

#SBATCH --job-name=ligandmpnn
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

# sugar CDRH3

python /home/kechen/protein_design/LigandMPNN/run.py \
            --checkpoint_ligand_mpnn "/home/kechen/protein_design/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt" \
            --checkpoint_path_sc "/home/kechen/protein_design/LigandMPNN/model_params/ligandmpnn_sc_v_32_002_16.pt" \
            --model_type "ligand_mpnn" \
            --seed 111 \
            --pdb_path_multi "/home/kechen/peppocketgen/dataset/ligandmpnn/sugar_redesigned_pdbs.json" \
            --out_folder "../dataset/ligandmpnn/CDRH3" \
            --ligand_mpnn_use_side_chain_context 1 \
            --redesigned_residues_multi "/home/kechen/peppocketgen/dataset/ligandmpnn/sugar_redesigned_residues_H3.json" \
            --batch_size 8 \
            --pack_side_chains 1 \
            --number_of_packs_per_design 1 \
            --pack_with_ligand_context 1

# sugar all CDRS
python /home/kechen/protein_design/LigandMPNN/run.py \
            --checkpoint_ligand_mpnn "/home/kechen/protein_design/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt" \
            --checkpoint_path_sc "/home/kechen/protein_design/LigandMPNN/model_params/ligandmpnn_sc_v_32_002_16.pt" \
            --model_type "ligand_mpnn" \
            --seed 111 \
            --pdb_path_multi "/home/kechen/peppocketgen/dataset/ligandmpnn/sugar_redesigned_pdbs.json" \
            --out_folder "../dataset/ligandmpnn/allCDRs" \
            --ligand_mpnn_use_side_chain_context 1 \
            --redesigned_residues_multi "/home/kechen/peppocketgen/dataset/ligandmpnn/sugar_redesigned_residues_all_cdr.json" \
            --batch_size 8 \
            --pack_side_chains 1 \
            --number_of_packs_per_design 1 \
            --pack_with_ligand_context 1

# hapten CDRH3
python /home/kechen/protein_design/LigandMPNN/run.py \
            --checkpoint_ligand_mpnn "/home/kechen/protein_design/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt" \
            --checkpoint_path_sc "/home/kechen/protein_design/LigandMPNN/model_params/ligandmpnn_sc_v_32_002_16.pt" \
            --model_type "ligand_mpnn" \
            --seed 111 \
            --pdb_path_multi "/home/kechen/peppocketgen/dataset/ligandmpnn/hapten_redesigned_pdbs.json" \
            --out_folder "../dataset/ligandmpnn/CDRH3" \
            --ligand_mpnn_use_side_chain_context 1 \
            --redesigned_residues_multi "/home/kechen/peppocketgen/dataset/ligandmpnn/hapten_redesigned_residues_H3.json" \
            --batch_size 8 \
            --pack_side_chains 1 \
            --number_of_packs_per_design 1 \
            --pack_with_ligand_context 1

# hapten all CDRs
python /home/kechen/protein_design/LigandMPNN/run.py \
            --checkpoint_ligand_mpnn "/home/kechen/protein_design/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt" \
            --checkpoint_path_sc "/home/kechen/protein_design/LigandMPNN/model_params/ligandmpnn_sc_v_32_002_16.pt" \
            --model_type "ligand_mpnn" \
            --seed 111 \
            --pdb_path_multi "/home/kechen/peppocketgen/dataset/ligandmpnn/hapten_redesigned_pdbs.json" \
            --out_folder "../dataset/ligandmpnn/allCDRs" \
            --ligand_mpnn_use_side_chain_context 1 \
            --redesigned_residues_multi "/home/kechen/peppocketgen/dataset/ligandmpnn/hapten_redesigned_residues_all_cdr.json" \
            --batch_size 8 \
            --pack_side_chains 1 \
            --number_of_packs_per_design 1 \
            --pack_with_ligand_context 1
