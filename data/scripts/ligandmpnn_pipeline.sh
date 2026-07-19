# step1: generate ligandmpnn design

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
            --pdb_path_multi "/home/kechen/peppocketgen/dataset/CDRH3/sugar_redesigned_pdbs.json" \
            --out_folder "../dataset/ligandmpnn/CDRH3" \
            --ligand_mpnn_use_side_chain_context 1 \
            --redesigned_residues_multi "/home/kechen/peppocketgen/dataset/CDRH3/sugar_redesigned_residues_all_cdr.json" \
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
            --pdb_path_multi "/home/kechen/peppocketgen/dataset/CDRH3/hapten_redesigned_pdbs.json" \
            --out_folder "../dataset/ligandmpnn/CDRH3" \
            --ligand_mpnn_use_side_chain_context 1 \
            --redesigned_residues_multi "/home/kechen/peppocketgen/dataset/CDRH3/hapten_redesigned_residues_H3.json" \
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
            --pdb_path_multi "/home/kechen/peppocketgen/dataset/CDRH3/hapten_redesigned_pdbs.json" \
            --out_folder "../dataset/ligandmpnn/CDRH3" \
            --ligand_mpnn_use_side_chain_context 1 \
            --redesigned_residues_multi "/home/kechen/peppocketgen/dataset/CDRH3/hapten_redesigned_residues_all_cdr.json" \
            --batch_size 8 \
            --pack_side_chains 1 \
            --number_of_packs_per_design 1 \
            --pack_with_ligand_context 1

# step2: predict original structure to get msa
# (1) generate json
export LAYERNORM_TYPE=torch
nohup bash /home/kechen/antibody_design/testset/0217/structure_prediction/gen_json.sh -i /home/kechen/peppocketgen/dataset/sabdab/Ab-hapten/split_imgt_clean_fixed_fv -o /home/kechen/peppocketgen/dataset/sabdab/Ab-hapten/json > gen_json_hapten.log 2>&1 &
nohup bash /home/kechen/antibody_design/testset/0217/structure_prediction/gen_json.sh -i /home/kechen/peppocketgen/dataset/sabdab/Ab-sugar/split_imgt_clean_fixed_fv -o /home/kechen/peppocketgen/dataset/sabdab/Ab-sugar/json > gen_json_sugar.log 2>&1 &
nohup bash /home/kechen/antibody_design/testset/0217/structure_prediction/gen_json.sh -i /home/kechen/peppocketgen/dataset/sabdab/Ab-pep/split_imgt_clean_fixed_fv -o /home/kechen/peppocketgen/dataset/sabdab/Ab-pep/json > gen_json_pep.log 2>&1 &

# (2) merge json
python merge_json.py -i /home/kechen/peppocketgen/dataset/sabdab/Ab-sugar/json -o /home/kechen/peppocketgen/dataset/sabdab/Ab-sugar/ab_sugar.json -m merge
python merge_json.py -i /home/kechen/peppocketgen/dataset/sabdab/Ab-pep/json -o /home/kechen/peppocketgen/dataset/sabdab/Ab-pep/ab_pep.json -m merge
python merge_json.py -i /home/kechen/peppocketgen/dataset/sabdab/Ab-hapten/json -o /home/kechen/peppocketgen/dataset/sabdab/Ab-hapten/ab_hapten.json -m merge

# (3) predict structure
protenix pred -i /home/kechen/peppocketgen/dataset/sabdab/Ab-sugar/ab_sugar.json -o /home/kechen/peppocketgen/dataset/sabdab/Ab-sugar/protenix -s 101 -n protenix_base_20250630_v1.0.0
protenix pred -i /home/kechen/peppocketgen/dataset/sabdab/Ab-hapten/ab_hapten.json -o /home/kechen/peppocketgen/dataset/sabdab/Ab-hapten/protenix -s 101 -n protenix_base_20250630_v1.0.0
CUDA_VISIBLE_DEVICES=2 nohup protenix pred -i /home/kechen/peppocketgen/dataset/sabdab/Ab-pep/ab_pep.json -o /home/kechen/peppocketgen/dataset/sabdab/Ab-pep/protenix -s 101 -n protenix_base_20250630_v1.0.0 > protenix_pep.log 2>&1 &

