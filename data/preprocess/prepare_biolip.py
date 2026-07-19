"""prepare_biolip.py

Parse Q-BioLiP metadata, filter complexes by resolution and stoichiometry,
clean binding site text, and merge ligand and receptor PDB files into an
output directory using utilities from `data.utils`.

Usage: python prepare_biolip.py -i <metadata.csv> --lig_dir <ligand_dir> \
    --rec_dir <receptor_dir> -o <out_dir>
"""

import os
import re
from tqdm import tqdm
import pandas as pd
import data.utils as du
import argparse
from datetime import datetime

def tprint(*args, **kwargs):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]", *args, **kwargs)

parser = argparse.ArgumentParser(
    description='Complex processing script.')

parser.add_argument(
    '-i', '--metadata',
    help='Path to Q-BioLiP metadata CSV file.',
    default="/home/kechen/peppocketgen/dataset/pdb/qbiolip/Q-BioLiP_relevant.csv")

parser.add_argument(
    '--lig_dir',
    help='Directory containing ligand PDB files.',
    default="/home/kechen/peppocketgen/dataset/pdb/qbiolip/protein_ligand/nonredund_lig")

parser.add_argument(
    '--rec_dir',
    help='Directory containing receptor PDB files.',
    default="/home/kechen/peppocketgen/dataset/pdb/qbiolip/protein_ligand/nonredund_rec")

parser.add_argument(
    '-o', '--out_dir',
    help='Directory to save processed complex files.',
    default="/home/kechen/peppocketgen/dataset/pdb/complex/small_molecule")


def get_mer_count(mer_string):
    text = mer_string.lower()
    if "monomer" in text:
        return 1
    
    match = re.search(r'(\d+)-mer', text)
    if match:
        return int(match.group(1))
    return 1


def clean_binding_site(text):
    if not isinstance(text, str):
        return text
    
    # Pattern logic:
    # 1. [A-Za-z0-9]+ matches one or more letters or digits, such as B1, ChainA, or 6LU7.
    # 2. : matches the colon.
    # 3. \s* matches optional whitespace after the colon.
    # 4. re.sub replaces the matched prefix with an empty string.
    cleaned = re.sub(r'[A-Za-z0-9]+:\s*', '', text)
    
    # Finally, strip surrounding whitespace and collapse repeated internal spaces.
    return ' '.join(cleaned.split())


if __name__ == "__main__":
    args = parser.parse_args()
    lig_base_path = args.lig_dir
    rec_base_path = args.rec_dir
    save_dir = args.out_dir
    metadata = pd.read_csv(args.metadata)
    metadata['Resolution (Å)'] = pd.to_numeric(metadata['Resolution (Å)'], errors='coerce')

    keywords = []
    if "molecule" in save_dir:
        keywords = ["DNA", "RNA", "III", "_MG_", "_MN_", "_ZN_"]

    skip = 0
    for rec_pdb in tqdm(os.listdir(rec_base_path), desc = "processing pdbs"):
        rec_pdb_path = os.path.join(rec_base_path, rec_pdb)
        asseembly_id = os.path.splitext(os.path.basename(rec_pdb_path))[0] #1a0h_1
        # step1: filter by pdb level data
        sub_df = metadata[metadata["Assembly ID"]==asseembly_id].copy()
        if len(sub_df)<1:
            skip+=1
            continue
        
        # filter1: resolution > 3.0
        if sub_df["Resolution (Å)"].iloc[0] > 3.0:
            skip+=1
            continue
        
        # filter 2
        # length of subdf means the number of ligand-receptor pairs in the complex
        # we exclude ligand numbers that are more than the stoichiometry of the complex, as they may indicate multiple binding sites
        stoichiometry_type = sub_df["Stoichiometry"].iloc[0]
        chain_num = get_mer_count(stoichiometry_type)
        if len(sub_df)> chain_num:
            skip+=1
            continue
        
        # filter3: clean binding site text and drop duplicates
        # same binding site may be represented by different ligand names, we only keep one of them to avoid redundant merging
        sub_df["binding_residue"] = sub_df["Binding Site"].apply(clean_binding_site)
        sub_df = sub_df.drop_duplicates(subset='binding_residue', keep='first')
        if "Homo" in stoichiometry_type:
            sub_df = sub_df.drop_duplicates(subset='Ligand ID', keep='first')
        
        for i,row in sub_df.iterrows():
            # merging chains
            ligand_name = row["Ligand Detail"]
            if "kmer" in ligand_name:
                ligand_name = ligand_name.replace("_kmer_1_", "_kmer_")
            if any(sub in ligand_name for sub in keywords):
                continue
            ligand_pdb_path = os.path.join(lig_base_path, f"{ligand_name}.pdb")
            try:
                du.merge_PDB(ligand_pdb_path, rec_pdb_path, save_dir)
            except Exception as e:
                tprint(e)
                continue
        
    tprint(f"all tasks:{len(os.listdir(rec_base_path)[:30])}, skip:{skip}")