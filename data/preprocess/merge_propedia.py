# merge_propedia.py
# Purpose: Merge Propedia peptide–protein PDB files into composite complex PDBs grouped by peptide chain.
#
# Summary:
# - Groups input PDB files by PDB ID and peptide chain using filename pattern: xxxx_C_*.pdb.
# - For each group, uses the first PDB as the base structure and attempts to add non-duplicate chains
#   from other files only when the chain (1) contacts the peptide and (2) does not clash with existing chains.
# - Saves merged complexes as {pdbid}_{peptide}_{chain1_chain2...}.pdb.
# - Processes groups in parallel using ProcessPoolExecutor to speed up large datasets.
#
# Notes:
# - Uses Biopython for PDB parsing/writing and the repository helper data.utils.check_atom_clash_contact
#   to determine contacts and clashes. Thresholds are configurable via function args.
# - Intended for preprocessing Propedia dataset complexes before downstream modeling or analysis.

import re
from collections import defaultdict
from tqdm import tqdm
import argparse
import os
from Bio.PDB import PDBParser, PDBIO
import shutil
from datetime import datetime
from data.utils import check_atom_clash_contact
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

def tprint(*args, **kwargs):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]", *args, **kwargs)


parser = argparse.ArgumentParser(
    description='Complex processing script.')

parser.add_argument(
    '-i', '--propedia_dir',
    help='Path to Propedia complex file.',
    default="/home/kechen/peppocketgen/dataset/propedia/complex")

parser.add_argument(
    '-o', '--output_dir',
    help='Directory to save merged complex files.',)


def merge_PDB(file_list:list, save_dir:str, clash_distance=1.0, contact_distance = 5.5):
    # Merge multiple PDBs for the same (pdb_id, peptide_chain) into one composite PDB.
    # Uses the first file as the base and appends non-duplicate chains that
    # (a) contact the peptide and (b) do not clash with existing chains.
    if not file_list:
        print("Error: No pdb files")
        return
    
    peptide_chain = os.path.basename(file_list[0]).split("_")[1]
    pdb_id = os.path.basename(file_list[0]).split("_")[0]

    parser = PDBParser()
    struct1 = parser.get_structure("struct1", file_list[0])

    existing_chain_ids = {chain.id for chain in struct1.get_chains()}

    for pdb_file in file_list[1:]:
        struct2 = parser.get_structure("struct1", pdb_file)
        for chain in struct2.get_chains():
            new_chain_id = chain.id
            if new_chain_id in existing_chain_ids:
                continue
            
            # Selection 1: skip if new chain does not contact the peptide.
            # (check_atom_clash_contact returns (has_clash, has_contact)).
            _, has_contact = check_atom_clash_contact(struct1[0][peptide_chain], chain, clash_distance, contact_distance)
            if not has_contact:
                continue
            # Selection 2: ensure the new chain does not sterically clash with any existing chain.
            for existing_chain_id in existing_chain_ids:
                has_clash, _ = check_atom_clash_contact(struct1[0][existing_chain_id], chain, clash_distance, contact_distance)
                if has_clash:
                    break 

            # If no clash found, append the new chain to the base model.
            if not has_clash:
                new_chain = chain.copy()
                new_chain.id = new_chain_id
                struct1[0].add(new_chain)  # Add the new chain to the model.
                existing_chain_ids.add(new_chain_id)
            else:
                print(f"skip {chain.id} for has clash or no contact with ligand")
                continue

    io = PDBIO()
    io.set_structure(struct1)

    existing_chain_ids = list(existing_chain_ids)
    existing_chain_ids.remove(peptide_chain)
    existing_chain_ids = sorted(existing_chain_ids)
    chain_string = "_".join(existing_chain_ids)
    path = os.path.join(save_dir, f"{pdb_id}_{peptide_chain}_{chain_string}.pdb")
    #print(f"file will save to {path}")
    io.save(path)

def process_one_group(group_pdb, output_dir):
    # Worker wrapper for parallel execution:
    # - Copies single-file groups; calls merge_PDB for multi-file groups.
    # - Returns a short status string for logging.
    try:
        if len(group_pdb) == 1:
            shutil.copy2(group_pdb[0], output_dir)
            return f"Moved: {os.path.basename(group_pdb[0])}"
        else:
            merge_PDB(group_pdb, save_dir=output_dir, clash_distance=1.0)
            return f"Merged: {os.path.basename(group_pdb[0])} group"
    except Exception as e:
        return f"Error processing {group_pdb}: {e}"

if __name__ == "__main__":
    # Main: parse args, group input files by (pdb_id, peptide_chain),
    # and process groups in parallel with progress reporting.
    args = parser.parse_args()
    propedia_dir = args.propedia_dir
    output_dir = args.output_dir

    pattern = r'^([a-zA-Z0-9]{4})_([a-zA-Z])_.*\.pdb$'
    file_groups = defaultdict(list)

    tprint("begin grouping complex files")
    all_files = [f for f in os.listdir(propedia_dir) if f.endswith('.pdb')]

    #example: 1a1a_C_A.pdb
    for filename in tqdm(all_files):
        match = re.match(pattern, filename)
        if match:
            pdb_id = match.group(1).upper() # 1A1A
            chain_letter = match.group(2).upper() #C peptide chain number
            
            # Use (pdb_id, chain_letter) as the key.
            key = (pdb_id, chain_letter)
            file_groups[key].append(os.path.join(propedia_dir, filename))
        
    grouped_files =  list(file_groups.values())
    grouped_files.sort(key=lambda x: (x[0][:4].upper(), x[0][5].upper() if len(x[0]) > 5 else ''))

    tprint(f"begin merging  {len(grouped_files)} groups")
    max_workers = os.cpu_count() 
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        worker_func = partial(process_one_group, output_dir=output_dir)
        futures = [executor.submit(worker_func, group) for group in grouped_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if "Error" in result: print(result)

    tprint("All tasks completed.")