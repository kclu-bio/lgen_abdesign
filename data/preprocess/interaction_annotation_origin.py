# https://prolif.readthedocs.io/en/stable/notebooks/pdb.html#pdb

from pdb2pqr import run_pdb2pqr
import os
from pdb2pqr import io
from rdkit import Chem
from openbabel import pybel
import prolif as plf
import pandas as pd
from collections import defaultdict
import re
import numpy as np
from Bio.PDB import PDBIO, Structure, Model, PDBParser
import argparse
import pickle
from multiprocessing import Pool, cpu_count  # Process pool.
from tqdm import tqdm 
import gc
import multiprocessing

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def prot_addHs(input_pdb, output_dir):
    args = [
        "--ff=AMBER", 
        "--ffout=AMBER",
        "--keep-chain",
        "--with-ph=7.4",
        "--quiet",
        f"{input_pdb}",       # Input PDB file.
        f"{output_dir}/{os.path.basename(input_pdb).replace('.pdb', '.pqr')}"       # Output PQR file.
    ]

    missed_residues, pka_df, biomolecule = run_pdb2pqr(args)
    pdb_lines = io.print_biomolecule_atoms(biomolecule.atoms, chainflag=True, pdbfile=True)
    pdb_string = ''.join(pdb_lines)
    return pdb_string

def process_mol_string(string_data):
    mol = pybel.readstring("pdb", string_data)
    mol.OBMol.AddHydrogens(False, True, 7.4)  # False: nonpolar hydrogens; True: correct for pH 7.4.
    charge_model = pybel.ob.OBChargeModel.FindType("gasteiger")
    charge_model.ComputeCharges(mol.OBMol)
    return mol

def add_charge_feature(mol, ligand_coord:np.array):
    charge_list = [atom.partialcharge for atom in mol.atoms if atom.atomicnum > 1]
    coord_array = np.array([atom.coords for atom in mol.atoms if atom.atomicnum > 1])
    assert len(charge_list) == ligand_coord.shape[0]
    bias = ligand_coord[0] - coord_array[0]
    ligand_translated = ligand_coord - bias
    if np.sum(~np.isclose(coord_array, ligand_translated, rtol=1e-01)) < 0.05*len(charge_list):
        return charge_list
    else:
        raise ValueError("Coordinates do not match closely enough.")

def extract_interaction(ligand_sdf_string, protein_string):
    supplier = Chem.SDMolSupplier()
    supplier.SetData(ligand_sdf_string)
    rdkit_mol = supplier[0]
    ligand_mol = plf.Molecule.from_rdkit(rdkit_mol)

    rdkit_prot = Chem.MolFromPDBBlock(protein_string, sanitize=False)
    rdkit_protein_mol = plf.Molecule(rdkit_prot)
    # use default interactions
    fp = plf.Fingerprint()
    # run on your poses
    fp.run_from_iterable([ligand_mol], rdkit_protein_mol)
    df = fp.to_dataframe()
    return df.T

def get_result_from_df(df):
    result_dict = defaultdict(list)

    for _, row in df.iterrows():
        _, res_info, interaction = row.name  # Unpack the tuple.
        
        # Validate the residue format, for example ARG158.A.
        if "." in res_info:
            # 1. Split residue and chain ID: ARG158.A -> ['ARG158', 'A'].
            res_name_num, chain = res_info.split('.')
            # 2. Extract the residue number: 'ARG158' -> 158.
            res_id_match = re.search(r'\d+', res_name_num)
            if res_id_match:
                res_id = int(res_id_match.group())
                # 3. Build the target format (chain ID, number).
                key = (chain, res_id)
                # 4. Add it to the list when the value is True.
                if row[0] == True:
                    result_dict[key].append(interaction)

    final_dict = dict(result_dict)
    return final_dict

def get_chain_pdb_string(chain):
    import io
    # Turn chain to a temporary structure
    temp_structure = Structure.Structure("temp")
    temp_model = Model.Model(0)
    temp_model.add(chain)
    temp_structure.add(temp_model)

    io_buffer = io.StringIO()
    pdb_io = PDBIO()
    pdb_io.set_structure(temp_structure)
    pdb_io.save(io_buffer)

    return io_buffer.getvalue()


def process_row(row_dict, processed_path, raw_path):

    # Step 1: read processed_path and raw_path.
    row = pd.Series(row_dict)

    pqr_dir = f"{raw_path}/pqr_files"
    os.makedirs(pqr_dir, exist_ok=True)
    processed_path = f"{processed_path}/{row['item_name']}.pkl"
    raw_path = f"{raw_path}/{row['item_name']}.pdb"

    data_dict = pd.read_pickle(processed_path)
    pdb_parser = PDBParser(QUIET=True)
    pdb = pdb_parser.get_structure("complex", raw_path)
    ligand_chain_id = row["item_name"].split("_")[1]
    ligand_chain = pdb[0][ligand_chain_id]
    ligand_string = get_chain_pdb_string(ligand_chain)

    mol = process_mol_string(ligand_string)
    try:
        charge_list = add_charge_feature(mol, data_dict["ligand"]["ligand_pos"])
        charge_array = np.array(charge_list)
        if np.mean(np.abs(charge_array)) > 1e-05:
            row["has_charge"] = True
        else:
            row["has_charge"] = False
    except:
        charge_list = []
        row["has_charge"] = False
    
    sdf_string = mol.write("sdf")
    try:
        reduced_pdb_string = prot_addHs(raw_path, pqr_dir)
        interaction_data = extract_interaction(sdf_string, reduced_pdb_string)
        interaction_dict = get_result_from_df(interaction_data)
        row["has_interaction"] = True
    except:
        interaction_dict = {}
        row["has_interaction"] = False

    chains = data_dict["receptor"]["chain_index"]
    ids = data_dict["receptor"]["residue_index"]
    inter_list = [
        [inter for inter in interaction_dict.get((c, i), []) if inter != 'VdWContact']
        for c, i in zip(chains, ids)
    ]

    result = {"interaction": inter_list, "charge": charge_list}

    del mol, charge_array, interaction_data
    gc.collect() 
    return result, row.to_dict()

def _worker(args):
    row_dict, processed_dir, raw_dir = args
    return process_row(row_dict, processed_dir, raw_dir)

if __name__ == "__main__":
    # nohup python interaction_annotation.py --raw_dir /home/kechen/peppocketgen/dataset/propedia/merged_complex --processed_dir /home/kechen/peppocketgen/dataset/processed_training/merged_propedia_5.5 --save_dir /home/kechen/peppocketgen/dataset/processed_training/merged_propedia_5.5/inter_charge --n_jobs 16 > interaction_annotation.log 2>&1 &
    args = argparse.ArgumentParser()
    args.add_argument('--raw_dir', type=str, required=False, help='Directory containing raw PDB files.')
    args.add_argument('--processed_dir', type=str, required=False, help='Directory containing processed PKL files.')
    args.add_argument('--save_dir', type=str, required=False, help='Directory to save updated metadata CSV, interaction and charge_list file.')
    args.add_argument('--n_jobs', type=int, default=cpu_count(), help='Number of processes for parallelization.')
    args = args.parse_args()

    metadata = pd.read_csv(os.path.join(args.processed_dir, "metadata.csv"))
    tasks = [
        (metadata.iloc[idx].to_dict(), args.processed_dir, args.raw_dir)
        for idx in range(len(metadata))
    ]

    updated_rows = []
    os.makedirs(args.save_dir, exist_ok=True)

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    with Pool(processes=args.n_jobs, maxtasksperchild=50) as pool:
        for res, row_dict in tqdm(pool.imap_unordered(_worker, tasks), total=len(tasks)):
            item_name = row_dict["item_name"]
            file_path = os.path.join(args.save_dir, f"{item_name}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(res, f)
            updated_rows.append(row_dict)

    # Update metadata with parallel results while preserving order via item_name merge.
    updated_df = pd.DataFrame(updated_rows)
    # Update only matching columns to preserve the original metadata column order.
    for col in metadata.columns:
        if col in updated_df.columns:
            metadata[col] = metadata["item_name"].map(
                updated_df.set_index("item_name")[col]
            )

    metadata.to_csv(os.path.join(args.save_dir, "processed_metadata.csv"), index=False)
