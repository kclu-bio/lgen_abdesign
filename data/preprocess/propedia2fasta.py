import pandas as pd
import ast
import os
from tqdm import tqdm



merged_metadata = pd.read_csv("/home/kechen/peppocketgen/dataset/processed_training/merged_propedia_5.5/metadata.csv")


# Generate a FASTA file.
fasta_lines = []

for idx, row in tqdm(merged_metadata.iterrows()):
    item_name = row['item_name']

    try:
        seq_dict = ast.literal_eval(row['seq']) if isinstance(row['seq'], str) else row['seq']
    except (ValueError, SyntaxError) as e:
        print(f"Warning: unable to parse the seq field for item_name={item_name}: {e}")
        continue
    
    if not isinstance(seq_dict, dict):
        print(f"Warning: the seq field for item_name={item_name} is not a dictionary: {type(seq_dict)}")
        continue
    
    for chain_id, sequence in seq_dict.items():
        if not sequence or pd.isna(sequence):
            continue
            
        header = f">{item_name}_{chain_id}"
        
        # Add the sequence with at most 80 characters per line.
        fasta_lines.append(header)
        
        # Split the sequence into 80-character lines.
        for i in range(0, len(sequence), 80):
            fasta_lines.append(sequence[i:i+80])

# Save the FASTA file.
fasta_path = "/home/kechen/peppocketgen/dataset/processed_training/propedia_sequences.fasta"
with open(fasta_path, 'w') as f:
    f.write('\n'.join(fasta_lines))

print(f"FASTA file saved to: {fasta_path}")
print(f"Generated {len(fasta_lines)//2} sequence records")

# Summary statistics.
print("\nSummary statistics:")
print(f"Total metadata records: {len(merged_metadata)}")
total_chains = sum(
    len(ast.literal_eval(row['seq'])) if isinstance(row['seq'], str) else len(row['seq'])
    for _, row in merged_metadata.iterrows()
    if pd.notna(row['seq'])
)
print(f"Total chains: {total_chains}")