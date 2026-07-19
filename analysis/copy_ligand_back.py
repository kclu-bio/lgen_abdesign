import argparse
import os
from Bio import PDB
from Bio.PDB import PDBParser, PDBIO
from tqdm import tqdm
import re

# python copy_ligand_back.py --model diffab --mode h3
def remove_trailing_number(text):
    pattern = r'_\d+$'
    return re.sub(pattern, '', text)

def find_gt_pdb(generated_file, gt_paths:list):
    basename = os.path.basename(generated_file).split("_gen_len")[0]
    basename = remove_trailing_number(basename)
    for gt_path in gt_paths:
        for gt_file in os.listdir(gt_path):
            if gt_file.startswith(basename):
                antigen_type = gt_path.split("/")[-1]
                return os.path.join(gt_path, gt_file), antigen_type
    print(f"Warning: No matching GT PDB found for {generated_file}")
    return None

def add_chain_to_pdb(chain_obj, target_pdb_path, output_pdb_path=None):
    """
    Add a Biopython Chain object to an existing PDB file.
    
    Args:
        chain_obj: Bio.PDB.Chain.Chain object.
        target_pdb_path: Path to the target PDB file.
        output_pdb_path: Output path. Overwrite the input when None.
    """
    # 1. Parse the target PDB file.
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('target_struct', target_pdb_path)
    
    # 2. Get the first model; PDB files typically contain only model 0.
    # A model can be created manually if the target file is empty.
    if len(structure) == 0:
        model = PDB.Model.Model(0)
        structure.add(model)
    else:
        model = structure[0]

    # 3. Check for chain-ID conflicts.
    # If the ID exists, assign another available ID such as B or C.
    existing_ids = [c.id for c in model.get_chains()]
    if chain_obj.id in existing_ids:
       raise ValueError(f"Chain ID '{chain_obj.id}' already exists in the target PDB. Please change the chain ID of the input chain.")

    # 4. Add the chain to the model.
    # chain_obj must have no parent; use copy() when it comes from another structure.
    new_chain = chain_obj.copy() 
    model.add(new_chain)

    # 5. Save the result.
    io = PDB.PDBIO()
    io.set_structure(structure)
    
    save_path = output_pdb_path if output_pdb_path else target_pdb_path
    io.save(save_path)
    #print(f"Successfully wrote chain {new_chain.id} to {save_path}")

def copy_back(gen_base_dir, gt_path, sub_dir, sample_num, mode):
    parser = PDBParser(QUIET=True)
    if mode == "cdrs":
        sub_name = ["codesign_multicdrs", "MultipleCDRs"]
    elif mode == "h3":
        sub_name = ["codesign_single", "H_CDR3"]
    else:
        raise ValueError("Invalid mode. Choose 'cdrs' or 'h3'.")
    for sub in sub_dir:
        type_dir = os.path.join(gen_base_dir, sub, sub_name[0])
        for gen_dir in tqdm(os.listdir(type_dir), desc="processing_files"):
            item_name = gen_dir.split(".pdb")[0]
            full_gen_dir = os.path.join(type_dir, gen_dir)
            gt_file_with_ligand_full, ag_type =  find_gt_pdb(item_name, gt_path)
            #print(gt_file_with_ligand_full)
            ligand_chain = os.path.basename(gt_file_with_ligand_full).split("_")[1]
            ligand_object = parser.get_structure("ligand", gt_file_with_ligand_full)[0][ligand_chain]
            os.makedirs(os.path.join(full_gen_dir, f"{sub_name[1]}_with_ligand"), exist_ok=True)
            for i in range(sample_num):
                gen_file_full = os.path.join(full_gen_dir, sub_name[1], f"000{i}.pdb")  # DiffAb sample.
                if os.path.exists(gen_file_full):
                    output_pdb_path = gen_file_full.replace(sub_name[1], f"{sub_name[1]}_with_ligand")
                    add_chain_to_pdb(ligand_object, gen_file_full, output_pdb_path)
                else:
                    print(f"Generated file {gen_file_full} does not exist. Skipping.")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, choices=["diffab", "abeg"], required=True, help="Model type to process")
    args.add_argument("--mode", type=str, choices=["cdrs", "h3"], required=True, help="Whether to process CDRs or H3")
    args.add_argument("--sample_num", type=int, default=8, help="Number of samples to process")
    args = args.parse_args()

    if args.model == "diffab":
        gen_base_dir = "/home/kechen/antibody_design/diffab/results"
        gt_path = ["/home/kechen/antibody_design/testset/0217/hapten",
                "/home/kechen/antibody_design/testset/0217/sugar",
                "/home/kechen/antibody_design/testset/0217/peptide_standard", 
                "/home/kechen/antibody_design/testset/0217/peptide_non_standard"]
        sub_dir = ["hapten_non_ligand", "sugar_non_ligand"]
    elif args.model == "abeg":
        gen_base_dir = "/home/kechen/antibody_design/AbEgDiffuser/results"
        gt_path = ["/home/kechen/antibody_design/testset/0217/hapten",
                "/home/kechen/antibody_design/testset/0217/sugar",
                "/home/kechen/antibody_design/testset/0217/peptide_standard", 
                "/home/kechen/antibody_design/testset/0217/peptide_non_standard", ]
        sub_dir = ["hapten_non_ligand", "sugar_non_ligand"]
    
    copy_back(gen_base_dir, gt_path, sub_dir, args.sample_num, args.mode)