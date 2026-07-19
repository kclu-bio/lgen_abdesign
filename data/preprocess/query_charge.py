import pandas as pd
import argparse
from interaction_annotation import get_chain_pdb_string, process_mol_string, add_charge_feature
from Bio.PDB import PDBParser
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pickle

elements_dict = {
    "H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "NE": 10,
    "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15, "S": 16, "CL": 17, "AR": 18, "K": 19, "CA": 20,
    "SC": 21, "TI": 22, "V": 23, "CR": 24, "MN": 25, "FE": 26, "CO": 27, "NI": 28, "CU": 29, "ZN": 30,
    "GA": 31, "GE": 32, "AS": 33, "SE": 34, "BR": 35, "KR": 36, "RB": 37, "SR": 38, "Y": 39, "ZR": 40,
    "NB": 41, "MO": 42, "TC": 43, "RU": 44, "RH": 45, "PD": 46, "AG": 47, "CD": 48, "IN": 49, "SN": 50,
    "SB": 51, "TE": 52, "I": 53, "XE": 54, "CS": 55, "BA": 56, "LA": 57, "CE": 58, "PR": 59, "ND": 60,
    "PM": 61, "SM": 62, "EU": 63, "GD": 64, "TB": 65, "DY": 66, "HO": 67, "ER": 68, "TM": 69, "YB": 70,
    "LU": 71, "HF": 72, "TA": 73, "W": 74, "RE": 75, "OS": 76, "IR": 77, "PT": 78, "AU": 79, "HG": 80,
    "TL": 81, "PB": 82, "BI": 83, "PO": 84, "AT": 85, "RN": 86, "FR": 87, "RA": 88, "AC": 89, "TH": 90,
    "PA": 91, "U": 92, "NP": 93, "PU": 94, "AM": 95, "CM": 96, "BK": 97, "CF": 98, "ES": 99, "FM": 100,
    "MD": 101, "NO": 102, "LR": 103, "RF": 104, "DB": 105, "SG": 106, "BH": 107, "HS": 108, "MT": 109,
    "DS": 110, "RG": 111, "CN": 112, "NH": 113, "FL": 114, "MC": 115, "LV": 116, "TS": 117, "OG": 118, "OXT":8
}

def element2num(element_list):
    num_list = [elements_dict.get(element, 1) for element in element_list]
    return num_list

def read_mol2_file(filename):

    """
    Read a MOL2 file and extract the second column (atom type) and last column (charge).
    
    Args:
        filename: MOL2 filename.
        
    Returns:
        atom_types: List of atom types.
        charges: List of charge values.
    """
    atom_types = []
    charges = []
    
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # Locate the start of the ATOM section.
        in_atom_section = False
        
        for line in lines:
            line = line.strip()
            
            # Check whether the ATOM section has started.
            if line.startswith('@<TRIPOS>ATOM'):
                in_atom_section = True
                continue
            # Check whether the ATOM section has ended.
            elif line.startswith('@<TRIPOS>'):
                if in_atom_section:
                    in_atom_section = False
                continue
            
            # Parse atom records within the ATOM section.
            if in_atom_section and line:
                # Split the line on whitespace.
                parts = line.split()
                
                if len(parts) >= 9:  # Ensure enough columns are present.
                    # The second column is the atom type (index 1).
                    atom_type = parts[1]
                    
                    # The last column is the charge (index -1).
                    try:
                        charge = float(parts[-1])
                    except ValueError:
                        charge = 0.0
                    
                    atom_types.append(atom_type)
                    charges.append(charge)
    first_H = atom_types.index('H') if "H" in atom_types else len(atom_types)
    return atom_types[:first_H], charges[:first_H]

def charge_cal_pipeline(raw_path, row, data_dict):
    pdb_parser = PDBParser(QUIET=True)
    raw_path = f"{raw_path}/{row['item_name']}.pdb"
    pdb = pdb_parser.get_structure("complex", raw_path)
    ligand_chain = pdb[0]["A"]
    ligand_string = get_chain_pdb_string(ligand_chain)
    mol = process_mol_string(ligand_string)
    try:
        charge_list = add_charge_feature(mol, data_dict["ligand"]["ligand_pos"])
        charge_array = np.array(charge_list)
        if np.mean(np.abs(charge_array)) > 1e-05:
            row["has_charge"] = True
            row["error_reason"] = ""

        else:
            row["has_charge"] = False
            row["error_reason"] = "zero charge"

    except:
        charge_list = []
        row["has_charge"] = False
        row["error_reason"] = "exception in charge calculation"
    
    return charge_list, row

def process_row(row_dict, processed_path, raw_path, CCD_mol2_path):
    row = pd.Series(row_dict)
    

    processed_path = f"{processed_path}/{row['item_name']}.pkl"
    data_dict = pd.read_pickle(processed_path)
    CCD_id = row["item_name"].split("_")[-2]

    CCD_path = os.path.join(CCD_mol2_path, f"{CCD_id}.mol2")
    if os.path.exists(CCD_path):
        atom_types, charge_list = read_mol2_file(CCD_path)
        atom_num = element2num(atom_types)
        if atom_num == data_dict["ligand"]["element"].tolist():
            row["has_charge"] = True
            row["error_reason"] = ""
        else:
            #row["error_reason"] = "atom order does not match"
            #row["has_charge"] = False
            #charge_list = []
            charge_list, row = charge_cal_pipeline(raw_path, row, data_dict)

    else:
        charge_list, row = charge_cal_pipeline(raw_path, row, data_dict)

    return charge_list, row.to_dict()

# small molecule:
#  nohup python query_charge.py --raw_dir /home/kechen/peppocketgen/dataset/pdb/complex/protein_small_molecule --processed_dir /home/kechen/peppocketgen/dataset/processed_training/protein_small_molecule_5.5 --save_dir /home/kechen/peppocketgen/dataset/processed_training/protein_small_molecule_5.5/charge > charge_smml.log 2>&1 &
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--raw_dir', type=str, required=False, help='Directory containing raw PDB files.')
    args.add_argument('--processed_dir', type=str, required=False, help='Directory containing processed PKL files.')
    args.add_argument('--save_dir', type=str, required=False, help='Directory to save updated metadata CSV, interaction and charge_list file.')
    args.add_argument('--CCD_mol2_path', type=str, required=False, help='Directory containing CCD mol2 files.', default = '/home/kechen/peppocketgen/dataset/pdb/CCD')
    
    args = args.parse_args()
    metadata = pd.read_csv(os.path.join(args.processed_dir, "metadata.csv"))

    updated_rows = []
    result = defaultdict(dict)
    for idx in tqdm(range(len(metadata))):
        row_dict = metadata.iloc[idx].to_dict()
        charge_list, row_dict = process_row(row_dict, args.processed_dir, args.raw_dir, args.CCD_mol2_path)
        updated_rows.append(row_dict)
        result[row_dict["item_name"]]["charge"] = charge_list
    
    updated_df = pd.DataFrame(updated_rows)
    os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.save_dir, "inter_charge.pkl"), "wb") as f:
        pickle.dump(result, f)

    updated_df.to_csv(os.path.join(args.save_dir, "processed_metadata.csv"), index=False)
    print(f"Processing completed and results saved to {args.save_dir}.")
