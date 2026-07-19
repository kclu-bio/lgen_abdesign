python /home/kechen/protein_design/LigandMPNN/run.py \
            --checkpoint_ligand_mpnn "/home/kechen/protein_design/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt" \
            --checkpoint_path_sc "/home/kechen/protein_design/LigandMPNN/model_params/ligandmpnn_sc_v_32_002_16.pt" \
            --model_type "ligand_mpnn" \
            --seed 111 \
            --pdb_path "/home/kechen/peppocketgen/dataset/sabdab/Ab-sugar/split_imgt_clean_fixed_fv/6dwa_X_B_A.pdb" \
            --out_folder "./ligandmpnn_test" \
            --ligand_mpnn_use_side_chain_context 1 \
            #--redesigned_residues "B105 B106 B107 B108 B109 B114 B115 B116 B117" \
            --batch_size 8 \
            --pack_side_chains 1 \
            --number_of_packs_per_design 1 \
            --pack_with_ligand_context 1
