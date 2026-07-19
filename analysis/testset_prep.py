from pandas import pd
from Bio.PDB import PDBParser, PDBIO
import shutil
import os
from tqdm import tqdm
import argparse
import analysis.utils as au

def get_non_standard_residues(pdb_file, chain_id):
    """
    Check for nonstandard residues in a specified chain.
    
    Args:
    pdb_file (str): Path to the PDB file.
    chain_id (str): Chain ID, for example 'A'.
    
    Returns:
    list: Deduplicated names of all nonstandard residues.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    non_standard_residues = []
    
    # Inspect the first model, as PDB files typically contain one model.
    model = structure[0]
    
    if chain_id not in model:
        print(f"Warning: chain {chain_id} is not present in {pdb_file}.")
        return []

    chain = model[chain_id]
    
    for residue in chain:
        # get_id() returns (hetfield, resseq, icode).
        res_id = residue.get_id()
        hetfield = res_id[0]
        
        # Keep residues whose hetfield starts with 'H_' and exclude water ('W').
        if hetfield.startswith('H_'):
            res_name = residue.get_resname().strip()
            non_standard_residues.append(res_name)
            
    # Return unique names; return non_standard_residues directly to preserve every occurrence.
    return list(set(non_standard_residues))

def reorder_filename(pdb_file, new_dir):
    basename = os.path.splitext(os.path.basename(pdb_file))[0]
    pdb_id = basename.split("_")[0]
    agchain = basename.split("_")[1]
    hchain = basename.split("_")[2]
    lchain = basename.split("_")[3]
    new_filename = f"{pdb_id}_{hchain}_{lchain}_{agchain}.pdb"
    shutil.copy(pdb_file, os.path.join(new_dir, new_filename))


def delete_chain(pdb_path, chain_to_remove, output_pdb):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, au.ChainSelect(chain_to_remove, mode = "receptor"))
    
    print(f"Removed chain {chain_to_remove} from {pdb_path}; saved result to {output_pdb}")

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Prepare test set for peppocketgen")
    args.add_argument("--test_metadata", type=str, default="/home/kechen/peppocketgen/dataset/processed_finetune/test_0217.csv", help="Path to the test metadata CSV file")
    args.add_argument("--testset_id", type=str, default="0217", help="ID of the test set")
    args = args.parse_args()
    test_metadata = pd.read_csv(args.test_metadata)
    tqdm.pandas()
    test_metadata["non_standard_residues"] = test_metadata.progress_apply(lambda x: get_non_standard_residues(x["raw_path"], x["item_name"].split("_")[1]), axis=1)
    test_metadata.to_csv(f"/home/kechen/peppocketgen/dataset/processed_finetune/test_data_with_non_standard_residues_{args.testset_id}.csv", index=False)
    hapten_dir = f"/home/kechen/antibody_design/testset/{args.testset_id}/hapten"
    sugar_dir = f"/home/kechen/antibody_design/testset/{args.testset_id}/sugar"
    peptide_standard_dir = f"/home/kechen/antibody_design/testset/{args.testset_id}/peptide_standard"
    peptide_non_standard_dir = f"/home/kechen/antibody_design/testset/{args.testset_id}/peptide_non_standard"
    for dir in [hapten_dir, sugar_dir, peptide_standard_dir, peptide_non_standard_dir]:
        os.makedirs(dir, exist_ok=True)
    for row in tqdm(test_metadata.itertuples()):
        raw_path = row.raw_path
        item_name = row.item_name
        agtype = row.agtype
        non_standard_residues = row.non_standard_residues
        print(raw_path)
        if agtype == "hapten":
            dest_dir = hapten_dir
        elif agtype == "sugar":
            dest_dir = sugar_dir
        elif agtype == "peptide":
            if len(non_standard_residues)>2:
                dest_dir = peptide_non_standard_dir
            else:
                dest_dir = peptide_standard_dir
        else:
            print(f"Unknown antigen type {agtype}; skipping {item_name}")
            continue
        dest_path = os.path.join(dest_dir, os.path.basename(raw_path))
        shutil.copy(raw_path, dest_path)

    # Turn name to Abx style
    for dir in [peptide_standard_dir, peptide_non_standard_dir]:
        output_dir = os.path.join(dir, "abx")
        os.makedirs(output_dir, exist_ok=True)
        for pdb_file in tqdm(os.listdir(dir)):
            if pdb_file.endswith(".pdb"):
                reorder_filename(os.path.join(dir, pdb_file), output_dir)

    # delete antigen chain
    for dir in [hapten_dir, sugar_dir]:
        os.makedirs(os.path.join(dir, "non_ligand"), exist_ok=True)
    for pdb_file in tqdm(os.listdir(dir)):
        if pdb_file.endswith(".pdb") and "cothia" not in pdb_file:
            input_pdb = os.path.join(dir, pdb_file)
            output_pdb = os.path.join(dir, "non_ligand", pdb_file)
            chains_to_remove = os.path.basename(pdb_file).split("_")[1]
            delete_chain(input_pdb, chains_to_remove, output_pdb)