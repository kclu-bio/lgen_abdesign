#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

conda activate ppg
# (0) Get resolution data ../../dataset/resolution
python << 'EOF'
import os
import pickle
from tqdm import tqdm
resolution_dir = "../../dataset/resolution"
resolution_list = []
for txt in os.listdir(resolution_dir):
    txt_file = os.path.join(resolution_dir, txt)
    with open(txt_file, 'r', encoding='utf-8') as file:
        content = file.read()
        list = content.split(",")
        resolution_list.extend(list)

with open('../../dataset/pdb_res_lt3.pkl', 'wb') as f: 
    pickle.dump(resolution_list, f)
EOF

# (1)  get propedia
wget -P ../../dataset/pdb/propedia https://bioinfo.dcc.ufmg.br/propedia2/public/download/complex2_3.zip
unzip ../../dataset/pdb/propedia/complex2_3.zip -d ../../dataset/pdb/propedia/
python ../preprocess/merge_propedia.py --propedia_dir ../../dataset/pdb/propedia/complex --output_dir ../../dataset/pdb/propedia/merged_complex
# filter resolution
python << 'EOF'
import os
import pickle
base_dir_propedia = "../../dataset/pdb/propedia/merged_complex"
removed = []
resolution_list = pickle.load(open('../../dataset/pdb_res_lt3.pkl', 'rb'))
for propedia in tqdm(os.listdir(base_dir_propedia)):
    propedia_path = os.path.join(base_dir_propedia, propedia)
    propedia_basename = propedia.split("_")[0]
    if propedia_basename.upper() not in resolution_list:
        os.remove(propedia_path)
EOF

# (2) get and process Q-biolip
mkdir -p ../../dataset/pdb/qbiolip/protein_ligand
wget -P ../../dataset/pdb/qbiolip/protein_ligand https://yanglab.qd.sdu.edu.cn/Q-BioLiP/DATA/application/PL/nonredund_rec.tar.gz
wget -P ../../dataset/pdb/qbiolip/protein_ligand https://yanglab.qd.sdu.edu.cn/Q-BioLiP/DATA/application/PL/nonredund_rec_lig.tar.gz
wget -P ../../dataset/pdb/qbiolip/protein_ligand https://yanglab.qd.sdu.edu.cn/Q-BioLiP/DATA/Browse/Q-BioLiP_relevant.csv
tar -xzvf ../../dataset/pdb/qbiolip/protein_ligand/nonredund_rec.tar.gz -C ../../dataset/pdb/qbiolip/protein_ligand/
tar -xzvf ../../dataset/pdb/qbiolip/protein_ligand/nonredund_rec_lig.tar.gz -C ../../dataset/pdb/qbiolip/protein_ligand/
    # filter and merge file
python ../preprocess/parse_qbiolip.py --lig_dir ../../dataset/pdb/qbiolip/protein_ligand/nonredund_lig  --rec_dir ../../dataset/pdb/qbiolip/protein_ligand/nonredund_rec --output_dir ../../dataset/pdb/complex/small_molecule
# (3) Gnerate training metadata
# ligand
python ../pocket_parsing.py -i ../../dataset/pdb/complex/small_molecule -o ../../dataset/processed_training/protein_small_molecule --pocket_size 5.5 --mode new --peptide_chain A
# propedia_complex
python ../pocket_parsing.py -i ../../dataset/pdb/propedia/merged_complex -o ../../dataset/processed_training/merged_propedia --pocket_size 5.5 --mode new

# (4) Merge metadata and generate fasta file
python3 << 'EOF'
import pandas as pd
import ast
import os
from tqdm import tqdm


metadata1 = pd.read_csv("../../dataset/processed_training/merged_propedia_5.5/metadata.csv")
metadata1["type"] = "merged_propedia_5.5"

metadata2 = pd.read_csv("../../dataset/processed_training/protein_small_molecule_5.5/metadata.csv")
metadata2["type"] = "protein_small_molecule_5.5"

merged_metadata = pd.concat([metadata1, metadata2])

output_path = "../../dataset/processed_training/training_metadata.csv"
merged_metadata.to_csv(output_path, index=False)
print(f"Merged metadata saved to: {output_path}")

fasta_lines = []

for idx, row in tqdm(merged_metadata.iterrows()):
    item_name = row['item_name']

    try:
        seq_dict = ast.literal_eval(row['seq']) if isinstance(row['seq'], str) else row['seq']
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Unable to parse seq field for item_name={item_name}: {e}")
        continue
    
    if not isinstance(seq_dict, dict):
        print(f"Warning: seq field for item_name={item_name} is not a dictionary: {type(seq_dict)}")
        continue
    
    for chain_id, sequence in seq_dict.items():
        if not sequence or pd.isna(sequence):
            continue
            
        header = f">{item_name}_{chain_id}"
        
        fasta_lines.append(header)
        
        for i in range(0, len(sequence), 80):
            fasta_lines.append(sequence[i:i+80])

# Save FASTA file
fasta_path = "../../dataset/processed_training/training_sequences.fasta"
with open(fasta_path, 'w') as f:
    f.write('\n'.join(fasta_lines))

print(f"FASTA file saved to: {fasta_path}")
print(f"Total sequences generated: {len(fasta_lines)//2}")

# Statistics
print("\nStatistics:")
print(f"Total metadata records: {len(merged_metadata)}")
print(f"Total chains: {sum(len(ast.literal_eval(row['seq'])) if isinstance(row['seq'], str) else len(row['seq']) 
                   for _, row in merged_metadata.iterrows() 
                   if pd.notna(row['seq']))}")
EOF

# (5) generate propedia peptide fasta file
python ../preprocess/propedia_pep2fasta.py
conda deactivate
conda activate bioinfo
bash ../../dataset/mmseqs.sh -i 0.5 -c 0.8 -f "../../dataset/propedia_pep_sequences.fasta"