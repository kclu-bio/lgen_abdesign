# https://prolif.readthedocs.io/en/stable/notebooks/pdb.html#pdb

from pdb2pqr import run_pdb2pqr
import os
from pdb2pqr import io
from rdkit import Chem
from openbabel import pybel
import prolif as plf
import pandas as pd
from collections import defaultdict, deque  # deque supports timeout-aware task scheduling.
import re
import numpy as np
from Bio.PDB import PDBIO, Structure, Model, PDBParser
import argparse
import pickle
from multiprocessing import Pool, cpu_count  # Process pool.
from tqdm import tqdm 
import gc
import multiprocessing
from openbabel import openbabel as ob
import signal

# Avoid the gsd exit issue.
signal.signal(signal.SIGTERM, signal.SIG_DFL)

ob.obErrorLog.SetOutputLevel(ob.obError) 
tqdm.monitor_interval = 0

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def prot_addHs(input_pdb, output_dir):
    argument = [
        "--ff=AMBER", 
        "--ffout=AMBER",
        "--keep-chain",
        "--with-ph=7.4",
        "--quiet",
        f"{input_pdb}",       # Input PDB file.
        f"{output_dir}/{os.path.basename(input_pdb).replace('.pdb', '.pqr')}"       # Output PQR file.
    ]

    missed_residues, pka_df, biomolecule = run_pdb2pqr(argument)
    pdb_lines = io.print_biomolecule_atoms(biomolecule.atoms, chainflag=True, pdbfile=True)
    pdb_string = ''.join(pdb_lines)
    return pdb_string

def process_mol_string(string_data):
    mol = pybel.readstring("pdb", string_data)
    mol.OBMol.AddHydrogens(False, True, 7.4)  # False: nonpolar hydrogens; True: correct for pH 7.4.
    charge_model = pybel.ob.OBChargeModel.FindType("eem")
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


def process_row(row_dict, processed_path, raw_path, charge_only=False, mode = "peptide"):

    # Step 1: read processed_path and raw_path.
    row = pd.Series(row_dict)

    pqr_dir = f"{raw_path}/pqr_files"
    processed_path = f"{processed_path}/{row['item_name']}.pkl"
    raw_path = f"{raw_path}/{row['item_name']}.pdb"

    data_dict = pd.read_pickle(processed_path)
    pdb_parser = PDBParser(QUIET=True)
    pdb = pdb_parser.get_structure("complex", raw_path)
    if mode == "peptide":
        ligand_chain_id = row["item_name"].split("_")[1]
    elif mode == "small_molecule":
        ligand_chain_id = "A"
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    ligand_chain = pdb[0][ligand_chain_id]
    ligand_string = get_chain_pdb_string(ligand_chain)

    mol = process_mol_string(ligand_string)
    charge_array = None  # Initialize before exception-prone paths.
    try:
        charge_list = add_charge_feature(mol, data_dict["ligand"]["ligand_pos"])
        charge_array = np.array(charge_list)
        if np.mean(np.abs(charge_array)) > 1e-04:
            row["has_charge"] = True
        else:
            row["has_charge"] = False
    except Exception as e:
        charge_list = []
        row["has_charge"] = False
        row["fail_reason"] = str(e)
    
    if charge_only:
        result = {"charge": charge_list}
        return result, row.to_dict()
    
    sdf_string = mol.write("sdf")
    interaction_data = None  # Initialize before exception-prone paths.
    
    try:
        reduced_pdb_string = prot_addHs(raw_path, pqr_dir)
        interaction_data = extract_interaction(sdf_string, reduced_pdb_string)
        interaction_dict = get_result_from_df(interaction_data)
        row["has_interaction"] = True
    except Exception as e:
        interaction_dict = {}
        row["fail_reason"] = str(e)
        row["has_interaction"] = False

    chains = data_dict["receptor"]["chain_index"]
    ids = data_dict["receptor"]["residue_index"]
    
    try:
        inter_list = [
            [inter for inter in interaction_dict.get((c, i), []) if inter != 'VdWContact']
            for c, i in zip(chains, ids)
        ]
    except Exception as e:
        inter_list = [[] for _ in range(len(ids))]
        row["fail_reason"] = str(e)
        row["has_interaction"] = False

    result = {"interaction": inter_list, "charge": charge_list}

    if mol is not None:
        del mol
    if charge_array is not None:  # Clean up only after successful generation.
        del charge_array
    if interaction_data is not None:  # Clean up only after successful generation.
        del interaction_data
    gc.collect() 
    return result, row.to_dict()

def _worker(args):  # Wrap worker exceptions and return a consistent structure.
    row_dict, processed_dir, raw_dir, charge_only, mode = args
    try:
        result, updated_row = process_row(row_dict, processed_dir, raw_dir, charge_only, mode)
        return True, result, updated_row, None  # ok, result, row, error
    except Exception as exc:
        failed_row = dict(row_dict)
        failed_row["has_charge"] = False
        failed_row["has_interaction"] = False
        return False, {"interaction": [], "charge": []}, failed_row, repr(exc)  # Return the same structure on failure.


def _iter_results_with_timeout(pool, tasks, timeout, max_pending, pool_factory):  # Timeout-aware iterator.
    task_queue = deque(tasks)
    pending = deque()

    def submit_more():
        while task_queue and len(pending) < max_pending:
            task = task_queue.popleft()
            pending.append((task, pool.apply_async(_worker, (task,))))

    submit_more()
    while pending or task_queue:
        if not pending:
            submit_more()
            if not pending:
                break

        task, async_res = pending.popleft()
        try:
            if timeout and timeout > 0:
                yield async_res.get(timeout=timeout), None
            else:
                yield async_res.get(), None
        except multiprocessing.TimeoutError:
            failed_row = dict(task[0])
            failed_row["has_charge"] = False
            failed_row["has_interaction"] = False
            yield (False, {"interaction": [], "charge": []}, failed_row, "timeout"), None
            submit_more()
            continue
        except Exception as exc:
            yield None, repr(exc)

        submit_more()

if __name__ == "__main__":
    # peptide charge_only
    # nohup python interaction_annotation.py --raw_dir /home/kechen/peppocketgen/dataset/propedia/merged_complex --processed_dir /home/kechen/peppocketgen/dataset/processed_training/merged_propedia_5.5 --save_dir /home/kechen/peppocketgen/dataset/processed_training/merged_propedia_5.5/inter_charge --n_jobs 16 --chunksize 20 --maxtasksperchild 20 --task_timeout 300 --charge_only --mode peptide > charge_only.log 2>&1 &
    
    # peptide interaction + charge
    # nohup python interaction_annotation.py --raw_dir /home/kechen/peppocketgen/dataset/propedia/merged_complex --processed_dir /home/kechen/peppocketgen/dataset/processed_training/merged_propedia_5.5 --save_dir /home/kechen/peppocketgen/dataset/processed_training/merged_propedia_5.5/inter_charge_test --n_jobs 2 --chunksize 20 --maxtasksperchild 20 --task_timeout 300 --mode peptide > peptide_inter.log 2>&1 &

    # small_molecule
    # nohup python interaction_annotation.py --raw_dir  /home/kechen/peppocketgen/dataset/pdb/complex/protein_small_molecule --processed_dir /home/kechen/peppocketgen/dataset/processed_training/protein_small_molecule_5.5 --save_dir /home/kechen/peppocketgen/dataset/processed_training/protein_small_molecule_5.5/inter_charge --n_jobs 16 --chunksize 20 --maxtasksperchild 20 --task_timeout 300 --mode small_molecule > molecule_inter.log 2>&1 &

    args = argparse.ArgumentParser()
    args.add_argument('--raw_dir', type=str, required=False, help='Directory containing raw PDB files.')
    args.add_argument('--processed_dir', type=str, required=False, help='Directory containing processed PKL files.')
    args.add_argument('--save_dir', type=str, required=False, help='Directory to save updated metadata CSV, interaction and charge_list file.')
    args.add_argument('--n_jobs', type=int, default=cpu_count(), help='Number of processes for parallelization.')
    args.add_argument('--chunksize', type=int, default=20, help='Chunksize for pool.imap_unordered.')  # Task chunk size.
    args.add_argument('--maxtasksperchild', type=int, default=20, help='Recycle worker after N tasks.')  # Worker recycle threshold.
    args.add_argument('--task_timeout', type=int, default=300, help='Per-task timeout in seconds; 0 means no timeout.')  # Per-task timeout.
    args.add_argument('--charge_only', action='store_true', help='Only compute charge, skip interaction.')
    args.add_argument('--mode', type=str, required=False, help='Mode of operation.')

    args = args.parse_args()

    metadata = pd.read_csv(os.path.join(args.processed_dir, "metadata.csv"))
    tasks = [
        (metadata.iloc[idx].to_dict(), args.processed_dir, args.raw_dir, args.charge_only, args.mode)
        for idx in range(len(metadata))
    ]

    updated_rows = []
    pqr_dir = f"{args.raw_dir}/pqr_files"
    os.makedirs(pqr_dir, exist_ok=True)

    os.makedirs(args.save_dir, exist_ok=True)
    results = defaultdict(dict)

    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    if args.task_timeout and args.task_timeout > 0:  # Timeout mode.
        ctx = multiprocessing.get_context("spawn")
        def _make_pool():
            return ctx.Pool(processes=args.n_jobs, maxtasksperchild=args.maxtasksperchild)

        with _make_pool() as pool:
            max_pending = max(args.n_jobs * 4, 8)
            iterator = _iter_results_with_timeout(
                pool, tasks, args.task_timeout, max_pending, _make_pool
            )
            for result, err in tqdm(iterator, total=len(tasks)):
                if result is None:
                    print("no results")
                    continue
                ok, res, row_dict, worker_err = result
                item_name = row_dict["item_name"]
                results[item_name] = res
                #file_path = os.path.join(args.save_dir, f"{item_name}.pkl")
                #with open(file_path, "wb") as f:
                #    pickle.dump(res, f)
                if worker_err:  # Record worker exception details.
                    row_dict["error"] = worker_err
                updated_rows.append(row_dict)
    else:
        with Pool(processes=args.n_jobs, maxtasksperchild=args.maxtasksperchild) as pool:
            for result in tqdm(
                pool.imap_unordered(_worker, tasks, chunksize=args.chunksize),
                total=len(tasks),
            ):
                ok, res, row_dict, worker_err = result
                item_name = row_dict["item_name"]
                results[item_name] = res
                
                if worker_err:  # Record worker exception details.
                    row_dict["error"] = worker_err
                updated_rows.append(row_dict)

    # Update metadata with parallel results while preserving order via item_name merge.

    with open(os.path.join(args.save_dir, "inter_charge.pkl"), "wb") as f:
        pickle.dump(results, f)

    updated_df = pd.DataFrame(updated_rows)
    # Update only matching columns to preserve the original metadata column order.
    '''
    for col in metadata.columns:
        if col in updated_df.columns:
            metadata[col] = metadata["item_name"].map(
                updated_df.set_index("item_name")[col]
            )
    '''
    updated_df.to_csv(os.path.join(args.save_dir, "processed_metadata.csv"), index=False)
    print(f"Processing completed and metadata saved to {os.path.join(args.save_dir, 'processed_metadata.csv')}.")