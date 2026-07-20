import argparse
import dataclasses
import functools as fn
import pandas as pd
import os
import multiprocessing as mp
import time
from Bio import PDB
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from data import utils as du
from data import parsers
from data import errors
from data import residue_constants as rc

from scipy.spatial import cKDTree
from Bio.PDB.Chain import Chain
from Bio.PDB import PDBIO, Structure, Model
import io

# example:
# python pocket_parsing.py -i ../dataset/PPDBench/complexes113 -o ../dataset/processed_PPDBench -m ppdbench
# python pocket_parsing.py -i ../../pepflow/dataset/propedia/complex -o ../dataset/processed_propedia -s 5.5
# python pocket_parsing.py -r "H.5-7,10_L.59-61,28,90" -i ../dataset/example -o ../dataset/example -p A --mode new

# python pocket_parsing.py -r "A.175-186,204-206,243-250,287-294,312-319,358-373" -i ../dataset/muc -o ../dataset/muc --mode new
# python pocket_parsing.py -r "A.27-38,56-58,95-102,164-171,189-196,235-250" -i ../dataset/design_new/muc/scfv -o ../dataset/muc/scfv --mode new

# for antibody
# python pocket_parsing.py -i ../dataset/sabdab/Ab-pep/split_imgt_clean_fixed_fv -o ../dataset/processed_finetune/Ab_pep_fv --mode imgt_filename 
# python pocket_parsing.py -i ../dataset/sabdab/Ab-sugar/split_imgt_clean_fixed_fv -o ../dataset/processed_finetune/Ab_sugar_fv --mode imgt_filename 
# python pocket_parsing.py -i ../dataset/sabdab/Ab-hapten/split_imgt_clean_fixed_fv -o ../dataset/processed_finetune/Ab_hapten_fv --mode imgt_filename 
# python pocket_parsing.py -i ../dataset/sabdab/Ab-protein/split_imgt_clean_fixed_fv -o ../dataset/processed_finetune/Ab_protein_fv --mode imgt_filename --ligand_crop_threshold 8.0

# Abeta
# python pocket_parsing.py -i ../dataset/design_new/abeta/docked_complex -o ../dataset/design_new/abeta/processed_docked_complex --mode imgt

# tau
# python pocket_parsing.py -i ../dataset/design_new/tau -o ../dataset/design_new/tau/processed --mode imgt

#competition
# python pocket_parsing.py -i ../dataset/design_new/competition/compelx -o ../dataset/design_new/competition/complex_processed --mode imgt

# monomer
# python pocket_parsing.py -i ../dataset/pdb/filtered_monomer -o ../dataset/processed_training/monomer --mode none
# multimer
# python pocket_parsing.py -i ../dataset/pdb/complex/protein_protein -o ../dataset/processed_training/hetero_multimer --mode none
# ligand
# python pocket_parsing.py -i ../dataset/pdb/complex/protein_small_molecule -o ../dataset/processed_training/protein_small_molecule --pocket_size 5.5 --mode new --peptide_chain A
# propedia_complex
# python pocket_parsing.py -i ../dataset/propedia/merged_complex -o ../dataset/processed_training/merged_propedia --pocket_size 5.5 --mode new

# testset H3
# python pocket_parsing.py -i /home/kechen/antibody_design/testset/0217/full -o /home/kechen/antibody_design/testset/0217/processed_full --mode imgt_H3

def parse_range(s:str) -> dict:
    """
    Input Example: "H.5-7,10_L.59-61,28,90", 
    Return Dict: {"H":[5,6,7,10], "L": [59,60,61,28,90]}
    """
    result = {}
    
    for group in s.split('_'):
        group = group.strip()
        if not group:
            continue
            
        if '.' not in group:
            raise argparse.ArgumentTypeError(f"Invalid format, missing dot in: {group}")
        
        key_values = group.split('.')
        if len(key_values) != 2:
            raise argparse.ArgumentTypeError(f"Invalid key-value format: {group}")
            
        key, value_str = key_values[0].strip(), key_values[1].strip()
        
        values = []
        for part in value_str.split(','):
            part = part.strip()
            if not part:
                continue
                
            if '-' in part:
                start_end = part.split('-')
                if len(start_end) != 2:
                    raise argparse.ArgumentTypeError(f"Invalid range: {part}")
                try:
                    start, end = map(int, start_end)
                    values.extend(range(start, end + 1))
                except ValueError:
                    raise argparse.ArgumentTypeError(f"Non-integer in range: {part}")
            else:
                try:
                    values.append(int(part))
                except ValueError:
                    raise argparse.ArgumentTypeError(f"Invalid value: {part}")
        
        result[key] = values
    return result

# Define the parser
parser = argparse.ArgumentParser(
    description='Complex processing script.')
group = parser.add_mutually_exclusive_group()

group.add_argument(
    '-s','--pocket_size',
    help='Path to write results to.',
    type =float)
group.add_argument(
    '-r','--pocket_idx',
    help='Residue indices defining the pocket, e.g., "H:5-7,10;L:59-61,28,90".',
    type = str)
parser.add_argument(
    '-i', '--complex_dir',
    help='Path to directory with PDB files.',
    type=str)
parser.add_argument(
    '-n','--num_processes',
    help='Number of processes.',
    type=int,
    default=80)
parser.add_argument(
    '-l','--ligand_type',
    help='Type of ligand. Choices: none, small_molecule, peptide',
    type=str)
parser.add_argument(
    '-o', '--write_dir',
    help='Path to write results to.',
    type=str)
parser.add_argument(
    '-p', '--peptide_chain',
    help='The chain id of peptide chain.',
    type=str)
parser.add_argument(
    '-m', '--mode',
    help='Data mode, either propedia or ppdbench.',
    type=str,
    required=True)
parser.add_argument(
    '--debug',
    help='Turn on for debugging.',
    action='store_true')
parser.add_argument(
    '--keep_chains',
    help='Turn on for not removing chains.',
    action='store_true')
parser.add_argument(
    '--ligand_crop_threshold',
    help='Distance threshold for cropping ligand atoms.',
    type=float,
    default=0.0
)
parser.add_argument(
    '--verbose',
    help='Whether to log everything.',
    action='store_true')

# for protein antigen
def crop_ligand(ligand_dict, receptor_pos, threshold):
    ligand_pos = ligand_dict["ligand_pos"]
    # Build a cKDTree from the receptor coordinates
    tree = cKDTree(receptor_pos)
    # For each ligand atom, find its nearest distance to the receptor; keep atoms within threshold
    ligand_mask = tree.query(ligand_pos, k=1)[0] <= threshold
    cropped_ligand_dict = {}
    for key, value in ligand_dict.items():
        cropped_ligand_dict[key] = value[ligand_mask]
    return cropped_ligand_dict

def get_chain_pdb_string(ligand: dict[Chain]):
    # Turn chain to a temporary structure
    temp_structure = Structure.Structure("temp")
    temp_model = Model.Model(0)
    for chain_id, chain in ligand.items():
        temp_model.add(chain)
    temp_structure.add(temp_model)

    io_buffer = io.StringIO()
    pdb_io = PDBIO()
    pdb_io.set_structure(temp_structure)
    pdb_io.save(io_buffer)

    return io_buffer.getvalue()

def feats2seq(feats: dict) -> dict:
    # design for single chain
    chain_aatype = feats["aatype"]
    chain_index = feats["chain_index"]
    unique_chains = np.unique(chain_index)
    restypes = rc.restypes
    restypes.append("X") #avoid non-standard amino acids
    seq_dict = {}
    for chain_id in unique_chains:
        mask = (chain_index == chain_id)
        chain_residues = chain_aatype[mask]
        chain_seq = "".join([restypes[res] for res in chain_residues])
        seq_dict[str(chain_id)] = chain_seq
    return seq_dict

def center_ligand_object(ligand: dict[Chain], center:np.array) -> dict[Chain]:
    assert center.shape == (3,), f"Expected shape (3,), but got {center.shape}"
    for chain_id, chain in ligand.items():
        for atom in chain.get_atoms():
            atom.coord -= center
    return ligand

def process_ligand(ligand: dict[Chain]) -> dict:
    
    # TREAT LIGAND AS GROUP OF HEAVY ATOMS
    ligand_feats = []
    for chain_id, chain in ligand.items():
        ligand_feat = {}
        ligand_feat["ligand_pos"] = np.array([atom.coord for atom in chain.get_atoms()])
        ligand_feat["element"] = np.array([rc.elements_dict.get(atom.element, 0) for atom in chain.get_atoms()])
        atom_residue = [rc.restype_3to1.get(atom.parent.resname, "X") for atom in chain.get_atoms()]
        atom_residue = np.array([rc.restype_order.get(
                res_shortname, rc.restype_num) for res_shortname in atom_residue])
        # which residue each atom belongs to
        ligand_feat["atom_residue"] = atom_residue
        #Remove hydrogens
        mask = ligand_feat["element"] != 1 
        for key in ligand_feat:
            ligand_feat[key] = ligand_feat[key][mask]

        ligand_feats.append(ligand_feat)
    
    if len(ligand_feats) > 1:
        concated_ligand_feat = du.concat_np_features(ligand_feats, False)
    else:
        concated_ligand_feat = ligand_feats[0]

    # ligand pos, element, atom_residue, nonhetatm_mask
    return concated_ligand_feat

def detect_pocket(
    receptor_coord: np.ndarray, ligand_coord: np.ndarray, 
    cutoff: float = 3.5) -> np.ndarray:
    """
    Detect pocket by distance to ligand.
    receptor_coord: (N, 37, 3) array of receptor atom coordinates.
    ligand_coord: (M, 3) array of ligand atom coordinates.
    """
    N = receptor_coord.shape[0]

    points = receptor_coord.reshape(-1, 3) #(N*37, 3)
    valid_mask = np.any(points != 0, axis=1)
    tree = cKDTree(ligand_coord)
    valid_points = points[valid_mask]
    if valid_points.size == 0:
        return np.zeros(N, dtype=int)

    # Query nearest distances from valid receptor points to the ligand
    valid_dists, _ = tree.query(valid_points, k=1) #(N*37,)
    full_dists = np.full(N*37, np.inf) #(N1*37,)
    full_dists[valid_mask] = valid_dists

    group_dists = full_dists.reshape(receptor_coord.shape[0], 37) #(N1, 37)
    mask = (group_dists <= cutoff).any(axis=1).astype(int)

    return mask

def filter_chain(features):
    # key : 'atom_positions', 'aatype', 'atom_mask', 'residue_index', 'chain_index', 'b_factors', 'bb_positions', "insertion_code"
    # Remove chains that do not contain any pocket residues
    pocket_mask = features["pocket_mask"]
    chain_index = features["chain_index"]
    
    keep_indices = []
    removed_chains = []
    
    unique_chains = np.unique(chain_index)
    for chain in unique_chains:
        chain_mask = (chain_index == chain)
        chain_pocket_mask = pocket_mask[chain_mask]
        
        if np.any(chain_pocket_mask == 1):
            chain_indices = np.where(chain_mask)[0]
            keep_indices.extend(chain_indices)
        else:
            removed_chains.append(str(chain))
    
    result_features = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray) and len(value) == len(pocket_mask):
            result_features[key] = value[keep_indices]
        else:
            raise ValueError
    
    return result_features, removed_chains

def process_file(file_path: str, 
                 write_dir: str, 
                 mode:str, 
                 pocket_size:int = 3.5, 
                 pocket_idx:dict = None,
                 peptide_chain:str = None,
                 keep_chains:bool = False,
                 ligand_crop_threshold:float = 0.0) -> dict:
    """Processes protein file into usable, smaller pickles.

    Args:
        file_path: Path to file to read.
        write_dir: Directory to write pickles to.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    metadata = {}
    item_name = os.path.basename(file_path).replace('.pdb', '') #1a1a_C_A
    metadata['item_name'] = item_name

    processed_path = os.path.join(write_dir, f'{item_name}.pkl')
    metadata['processed_path'] = processed_path
    metadata['raw_path'] = file_path
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(item_name, file_path)

    # Extract All Chains
    # Note: list(structure.get_chains()) is equivalent to list(structure[0])
    all_chains = {
        chain.id: chain
        for chain in structure.get_chains()}
    metadata['num_chains'] = len(all_chains)

    # propedia-specific
    if mode == "propedia":
        assert len(all_chains) == 2, \
            f'PPDBench file {file_path} should have exactly 2 chains, found {len(all_chains)}.'
        ligand_chain_id = item_name.split('_')[1] #C
        receptor_chain_id = item_name.split('_')[2] #A
        ligand = {ligand_chain_id: all_chains[ligand_chain_id]} # Chain Object
        receptor_dict = {receptor_chain_id: all_chains[receptor_chain_id]} #Dictionary of Chain Object

    # Condition: only one ligand chain, and the ligand chain is the shortest chain
    elif mode == "ppdbench" or (mode == "new" and peptide_chain is None) or mode == "imgt":
        ## Accept Multichain
        chains = list(all_chains.values())
        sorted_chains = sorted(chains, key=lambda chain: len(chain), reverse=True) # longer chains first
        ligand = {sorted_chains[-1].id: sorted_chains[-1]}
        ligand_chain_id = list(ligand.keys())[0]
        all_chains.pop(ligand_chain_id)
        receptor_dict = all_chains

    # Condition: filename contains ligand chain info, supports multiple ligand chains
    elif mode == "imgt_filename" or mode == "imgt_H3":
        ligand_chain_ids = item_name.split('_')[1].split('|') 
        try:
            ligand = {chain_id: all_chains.pop(chain_id) for chain_id in ligand_chain_ids}
        except:
            raise errors.DataError(f'File {file_path} specified ligand chains {ligand_chain_ids} not found in PDB chains {list(all_chains.keys())}')
        receptor_dict = all_chains

    # Condition: user specified peptide chain, supports multiple ligand chains
    elif mode == "new" and peptide_chain is not None:
        ## Accept Ligand multichain like "A|B"
        ligand_chain_ids = peptide_chain.split('|')
        ligand = {chain_id: all_chains.pop(chain_id) for chain_id in ligand_chain_ids}
        receptor_dict = all_chains
    
    # for non-ligand system
    elif mode == "none":
        receptor_dict = all_chains
        ligand = None
        pocket_idx = None

    else:
        raise errors.DataError(f'Unknown mode {mode} for file {file_path}')

    # Extract protein features
    all_receptor_feats = []
    for receptor_chain_id, receptor in receptor_dict.items():
        chain_prot = parsers.process_chain(receptor, receptor_chain_id)
        chain_dict = dataclasses.asdict(chain_prot) 
        all_receptor_feats.append(chain_dict)
    # key : 'atom_positions', 'aatype', 'atom_mask', 'residue_index', 'chain_index', 'b_factors', 'bb_positions', "insertion_code"
    complex_chain_dict = du.concat_np_features(all_receptor_feats, False)
    complex_chain_dict = du.parse_chain_feats(complex_chain_dict)    # Center chain atom coordinates using CA coordinates from all chains.
    # Extract ligand atom features

    gen_idx = None
    if ligand is not None:
        ligand_dict = process_ligand(ligand)
    # Center ligand atom coordinates the same way as the receptor
        ligand_dict["ligand_pos"] -= complex_chain_dict["bb_center"][None, :] #represented as all-atom molecule
        ligand = center_ligand_object(ligand, complex_chain_dict["bb_center"]) # centering the coordinate in the Ligand chain object

        metadata['num_ligand_atoms'] = len(ligand_dict['ligand_pos'])
        metadata["peptide_len"] = sum(len(chain) for chain in ligand.values())
        if ligand_crop_threshold > 0.0:
            ligand_dict = crop_ligand(ligand_dict, complex_chain_dict["atom_positions"].reshape(-1, 3), ligand_crop_threshold)
            metadata['num_ligand_atoms'] = len(ligand_dict['ligand_pos'])
    else:
        ligand_dict =  {
            'ligand_pos': np.zeros((0, 3)),
            'element': np.zeros((0, )),
            'atom_residue': np.zeros((0,)),
            'nonhetatm_mask': np.zeros((0,)),
        }
        metadata['num_ligand_atoms'] = 0
        metadata["peptide_len"] = 0

    # Remove the bb_center key to avoid errors later when slicing by min_idx/max_idx
    complex_chain_dict.pop("bb_center", None) 
    
    # Process geometry features
    if mode == "imgt" or mode == "imgt_filename":
        imgt_range = list(range(27, 39))+list(range(56, 65)) + list(range(105, 118))
        pocket_idx = {k:imgt_range for k in receptor_dict}

    elif mode == "imgt_H3":
        heavy_chain_id = item_name.split('_')[2]
        gen_range = list(range(105, 118))
        imgt_range = list(range(27, 39))+list(range(56, 65)) + list(range(105, 118))
        pocket_idx = {k:imgt_range for k in receptor_dict}
        if heavy_chain_id:
            gen_idx = {heavy_chain_id: gen_range}
        else:
            return None

    if pocket_idx is not None and len(pocket_idx) > 0:
        # Mode 1: specify binding region according to given pocket_idx
        for key in pocket_idx:
            if key not in complex_chain_dict["chain_index"]:
                raise errors.DataError(
                    f'Pocket idx chain {key} not found in complex {file_path} chains {complex_chain_dict["chain_index"]}')
        
        # Not every chain necessarily has regions to generate
        complex_chain_dict["pocket_mask"] = np.array([complex_chain_dict["residue_index"][i] in pocket_idx.get(key, []) 
                    for i, key in enumerate(complex_chain_dict["chain_index"])]).astype(int)
        if gen_idx is not None:
            complex_chain_dict["diffuse_mask"] = np.array([complex_chain_dict["residue_index"][i] in gen_idx.get(key, [])
                        for i, key in enumerate(complex_chain_dict["chain_index"])]).astype(int)
        # In antibody PDB files, residues like 100A and 100B may both be indexed as 100 in residue_index; if 100 is in pocket_idx both positions will be marked 1
    else:
        # Mode 2: specify binding region by distance
        if ligand is not None:
            complex_chain_dict["pocket_mask"] = detect_pocket(
                complex_chain_dict["atom_positions"], ligand_dict["ligand_pos"], 
                cutoff=pocket_size
            ) # Pocket residues are 1; all others are 0.
        # pocket residues marked as 1, others 0
        else:
            # Mode 3: no ligand or no specified hotspots, all zeros
            complex_chain_dict["pocket_mask"] = np.zeros(len(complex_chain_dict["residue_index"]), dtype=int)
    
    removed_chains = []
    if len(np.unique(complex_chain_dict["chain_index"])) > 1 and (ligand is not None) and not keep_chains:
        complex_chain_dict, removed_chains = filter_chain(complex_chain_dict)
        metadata["num_chains"] -=len(removed_chains)

    metadata["removed_chains"] = removed_chains
    receptor_aatype = complex_chain_dict['aatype']
    metadata['seq_len'] = len(receptor_aatype)
    metadata['pocket_len'] = np.sum(complex_chain_dict["pocket_mask"])
    nonhetatm_idx = np.where(receptor_aatype != 20)[0]
    nonhetatm_mask = (receptor_aatype != 20) # Non-standard amino acids are 0, others are 1
    if np.sum(nonhetatm_mask) == 0:
        raise errors.LengthError(f'{file_path}: No modeled residues')
    min_modeled_idx = np.min(nonhetatm_idx)
    max_modeled_idx = np.max(nonhetatm_idx)
    metadata['modeled_seq_len'] = max_modeled_idx - min_modeled_idx + 1
    complex_chain_dict['nonhetatm_idx'] = nonhetatm_idx
    complex_chain_dict['nonhetatm_mask'] = nonhetatm_mask
    
    metadata["seq"] = feats2seq(complex_chain_dict)
    # Check if pocket has non-standard residues.
    assert len(complex_chain_dict["pocket_mask"]) == len(complex_chain_dict["nonhetatm_mask"]), \
        "Pocket mask and non-hetatm mask must be the same length."
    non_standard_in_pocket = np.any(complex_chain_dict["pocket_mask"] & ~complex_chain_dict["nonhetatm_mask"])
    metadata["non_standard_in_pocket"] = non_standard_in_pocket
    # Write features to pickles.
    
    # Turn Ligand Obejet to PDB string to reduce size
    ligand = get_chain_pdb_string(ligand) if ligand is not None else ""
    data_dict = {
        "receptor": complex_chain_dict,
        "ligand": ligand_dict,
        "ligand_object": ligand
    }
    du.write_pkl(processed_path, data_dict)

    # Return metadata
    return metadata


def process_serially(all_paths: list[str], 
                     write_dir:str, 
                     mode:str,
                     pocket_size:int,
                     pocket_idx:dict,
                     peptide_chain:str,
                     keep_chains:bool = False,
                     ligand_crop_threshold:float = 0.0):
    all_metadata = []
    for i, file_path in tqdm(enumerate(all_paths)):
        try:
            start_time = time.time()
            metadata = process_file(
                file_path,
                write_dir,
                mode,
                pocket_size,
                pocket_idx,
                peptide_chain,
                keep_chains= keep_chains,
                ligand_crop_threshold=ligand_crop_threshold)
            if metadata is None:
                continue
            elapsed_time = time.time() - start_time
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
            all_metadata.append(metadata)
        except (errors.DataError, errors.LengthError, Exception) as e:
            print(f'Failed {file_path}: {e}')
            return None
    return all_metadata


def process_fn(
        file_path: str,
        pocket_size: int,
        mode:str,
        pocket_idx:dict,
        peptide_chain:str,
        verbose: bool = None,
        write_dir: str = None,
        keep_chains:bool = False,
        ligand_crop_threshold:float = 0.0
    ) -> dict:
    try:
        start_time = time.time()
        metadata = process_file(
            file_path,
            write_dir,
            mode,
            pocket_size,
            pocket_idx,
            peptide_chain,
            keep_chains= keep_chains,
            ligand_crop_threshold=ligand_crop_threshold)
            
        elapsed_time = time.time() - start_time
        if verbose:
            print(f'Finished {file_path} in {elapsed_time:2.2f}s')
        return metadata
    except (errors.DataError, errors.LengthError, Exception, ValueError) as e:
        print(f'Failed {file_path}: {e}')
        return None


def main(args):
    pdb_dir = args.complex_dir
    all_file_paths = [
        os.path.join(pdb_dir, x)
        for x in os.listdir(args.complex_dir) if '.pdb' in x]
    total_num_paths = len(all_file_paths)
    print(f'Begin processing {total_num_paths} files')
    pocket_size = args.pocket_size
    pocket_idx = args.pocket_idx
    if pocket_size is not None:
        write_dir = f"{args.write_dir}_{args.pocket_size}"
    else:
        write_dir = f"{args.write_dir}_{pocket_idx_string}"
    
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    if args.debug:
        metadata_file_name = 'metadata_debug.csv'
    else:
        metadata_file_name = 'metadata.csv'
    metadata_path = os.path.join(write_dir, metadata_file_name)
    print(f'Files will be written to {write_dir}')

    # Process each complex file
    if args.num_processes == 1 or args.debug:
        all_metadata = process_serially(
            all_file_paths,
            write_dir,
            args.mode,
            pocket_size,
            pocket_idx,
            args.peptide_chain,
            args.keep_chains,
            args.ligand_crop_threshold
        )
    else:
        _process_fn = fn.partial(
            process_fn,
            verbose=args.verbose,
            write_dir=write_dir,
            pocket_size=pocket_size,
            mode = args.mode,
            peptide_chain = args.peptide_chain,
            pocket_idx = pocket_idx,
            keep_chains = args.keep_chains,
            ligand_crop_threshold = args.ligand_crop_threshold
            )

        with mp.Pool(processes=args.num_processes) as pool:
            all_metadata = list(tqdm(
                pool.imap(_process_fn, all_file_paths),
                total=len(all_file_paths),
                desc="Processing Files",
                dynamic_ncols=True  
            ))
        all_metadata = [x for x in all_metadata if x is not None]
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)
    print(
        f'Finished processing {succeeded}/{total_num_paths} files')

if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = parser.parse_args()
    if args.peptide_chain is not None:
        assert args.mode == "new", "Peptide chain argument only works for new mode."
    if args.pocket_idx is not None:
        pocket_idx_string = args.pocket_idx
        args.pocket_idx = parse_range(args.pocket_idx)
        assert args.mode == "new", "Pocket idx argument only works for new mode."
    if args.mode == "imgt" or args.mode == "imgt_filename":
        pocket_idx_string = "imgt"
    if args.mode == "imgt_H3":
        pocket_idx_string = "imgt_H3"
    if args.mode == "none":
        pocket_idx_string = "no_ligand"
    main(args)