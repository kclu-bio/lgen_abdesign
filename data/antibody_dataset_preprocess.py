import argparse
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence
import shutil
from Bio import PDB
from Bio.PDB import PDBIO, PDBParser
from tqdm import tqdm
import logging
import pandas as pd
logger = logging.getLogger(__name__)

CDR1 = (27, 38)
CDR2 = (56, 65)
CDR3 = (105, 117)
FR4_EDGE = 129
IMGT_INDEX = list(range(27, 39)) + list(range(56, 65)) + list(range(105, 118))

EXCLUSION = ["HOH", "WAT", "CA", "NA", "MG", "CL", "SO4", "PO4", "K", "ZN", "FE", "MN", "CU", "NI", "IN", "CO", "CD", "GD3", "YT3", "IOD", "OH",
             "PEG", "P6G", "1PE", "PE3", "PG4" , "PC", 
             "GOL", "EDO", "CAC", "PMS", "IMD", "CIT", "ACT", "TRS", "NH2", "DMS"]

sabdab_metadata = pd.read_csv("/home/kechen/peppocketgen/dataset/sabdab/sabdab_summary_all.tsv", sep="\t")

class ResidueSelect:
    def __init__(self, retain_range: Dict[str, Sequence[int]]):
        self.retain_range = retain_range

    def accept_model(self, model):
        return True

    def accept_chain(self, chain):
        return True

    def accept_residue(self, residue):
        chain_id = residue.get_parent().id
        res_id = residue.id[1]
        if chain_id in self.retain_range:
            start, end = self.retain_range[chain_id]
            return start <= res_id <= end
        return True

    def accept_atom(self, atom):
        return True


class ChainSelect:
    """Biopython Select helper for chain filtering."""

    def __init__(self, chain_ids: Iterable[str]):
        self.chain_ids = set(chain_ids)

    def accept_model(self, model):
        return True

    def accept_chain(self, chain):
        return chain.get_id() in self.chain_ids

    def accept_residue(self, residue):
        return True

    def accept_atom(self, atom):
        return True

def parse_info_by_tsv(pdb_file):
    paired_info = []
    
    with open(pdb_file, 'r') as f:
        pdb_id = os.path.basename(pdb_file).split(".pdb")[0]
        sub_df = sabdab_metadata[sabdab_metadata["pdb"] == pdb_id]
        if sub_df.empty:
            print(f"No matching TSV metadata found for {pdb_file}")
            return []
        for _, row in sub_df.iterrows():
            chain_info = {}
            chain_info['type'] = 'paired'  # assume all are PAIRED_HL type
            if pd.notna(row['Hchain']):
                chain_info['hchain'] = row['Hchain'] 
            if pd.notna(row['Lchain']):
                chain_info['lchain'] = row['Lchain']
            if pd.notna(row['antigen_chain']):
                chain_info['agchain'] = row['antigen_chain'].replace(" ", "")
            else:
                logger.warning(f"No antigen chain info for {pdb_file} in TSV; skipping this entry")
                continue  
            if pd.notna(row['antigen_type']):
                chain_info['agtype'] = row['antigen_type'].replace(" ", "").lower()
            else:
                logger.warning(f"No antigen type info for {pdb_file} in TSV; skipping this entry")
                continue
            paired_info.append(chain_info)
    return paired_info

def parse_info_by_file(pdb_file):
    """
    Parse REMARK 5 lines from a PDB file, supporting PAIRED_HL and SINGLE formats.
    Improvement: supports multiple values separated by semicolons.
    """
    paired_info = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            chain_info = {}
            
            # Check whether this is a supported REMARK format.
            if line.startswith('REMARK   5 PAIRED_HL'):
                chain_info['type'] = 'paired'
            elif line.startswith('REMARK   5 SINGLE'):
                chain_info['type'] = 'single'
            else:
                continue
            
            # Extract all possible fields consistently.
            h_match = re.search(r'HCHAIN=([\w;]+)', line)
            l_match = re.search(r'LCHAIN=([\w;]+)', line)
            ag_match = re.search(r'AGCHAIN=([\w;]+)', line)
            agtype_match = re.search(r'AGTYPE=([\w;]+)', line)
            
            def process_field(value):
                """Process a field value, converting semicolon-separated items to pipe-separated."""
                if ';' in value:
                    # split and strip possible spaces
                    parts = [part.strip() for part in value.split(';')]
                    # filter out empty strings
                    parts = [part for part in parts if part]
                    # join with pipe
                    return '|'.join(parts)
                return value
            
            if h_match:
                chain_info['hchain'] = process_field(h_match.group(1))
            if l_match:
                chain_info['lchain'] = process_field(l_match.group(1))
            if ag_match:
                chain_info['agchain'] = process_field(ag_match.group(1))
            if agtype_match:
                chain_info['agtype'] = process_field(agtype_match.group(1)).lower()
            
            paired_info.append(chain_info)
    
    return paired_info

def extract_chains_from_pdb(structure, chain_ids: Sequence[str], output_file: str) -> None:
    for id in chain_ids:
        if id not in structure[0]:
            logger.error(f"Chain ID {id} not found in {structure.id}. Available chains: {[c.id for c in structure[0]]}")
            
    io = PDBIO()
    io.set_structure(structure)
   
    io.save(output_file, ChainSelect(chain_ids), preserve_atom_numbering=True)

def split_pdb_file(pdb_file: str, output_dir: str, agtype: str) -> None:
    """Split a PDB into per-REMARK chain groups."""
    parser = PDBParser(QUIET=True)
    pdb_id = os.path.basename(pdb_file).replace(".pdb", "")
    structure = parser.get_structure(pdb_id, pdb_file)
    all_chains = [chain.id for chain in structure[0]]
    if not any(isinstance(item, str) and item.islower() for item in all_chains):
        # No lowercase chain IDs present — parse REMARK lines inside the file
        # Numeric chain IDs also count as absence of lowercase chains
        paired_info = parse_info_by_file(pdb_file)
    else:
        # Lowercase chain IDs exist; use TSV metadata parsing instead of REMARK parsing
        logger.warning(f"Chain IDs in {pdb_file}  contain lowercase letters; using TSV metadata parsing instead of REMARK parsing")
        paired_info = parse_info_by_tsv(pdb_file)
    if not paired_info:
        logger.error(f"No PAIRED_HL remarks found in {pdb_file}")
        raise ValueError(f"No PAIRED_HL remarks found in {pdb_file}")

    for info in paired_info:
        # Antigen chain types may include multiple values separated by '|'; all must match
        info_agtype = set(info.get("agtype").split("|"))
        if (len(info_agtype) > 1) or agtype not in info_agtype:
            logger.warning(f"Skipping {pdb_file} due to agtype mismatch: {info_agtype} vs {agtype}")
            continue
        chain_ids: List[str] = []
        if "agchain" in info:
            chain_ids.extend(info["agchain"].split("|"))
        if "hchain" in info:
            chain_ids.append(info["hchain"])
        if "lchain" in info:
            chain_ids.append(info["lchain"])
        if not chain_ids:
            logger.error(f"No chain ids found for {pdb_id} in {pdb_file}")
            continue

        ag_chain = info.get("agchain", "")
        h_chain = info.get("hchain", "")
        l_chain = info.get("lchain", "")
        output_filename = f"{pdb_id}_{ag_chain}_{h_chain}_{l_chain}.pdb"
        output_path = os.path.join(output_dir, output_filename)
        logger.info(f"Writing split PDB: {output_path}")
        extract_chains_from_pdb(structure, chain_ids, output_path)

def split_pdb_in_dir(base_path: str, output_dir: str, agtype: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Splitting PDBs in {base_path} -> {output_dir} (agtype={agtype})")
    for imgt_pdb in tqdm(os.listdir(base_path), desc="splitting pdbs"):
        pdb_file = os.path.join(base_path, imgt_pdb)
        if pdb_file.endswith(".pdb"):
            try:
                split_pdb_file(pdb_file, output_dir, agtype)
            except Exception as e:
                logger.error(f"Error processing {pdb_file}: {e}. Skipping this file.")
                shutil.copy(pdb_file, os.path.join(output_dir, "ERR")) 
                continue

def remove_water_and_ion(
    pdb_file: str,
    output_dir: Optional[str] = None,
    exclusion: Optional[Sequence[str]] = None,
) -> None:
    """Remove water/ion/solvent residues from a PDB."""
    exclusion = exclusion or EXCLUSION
    output_file = pdb_file if output_dir is None else os.path.join(output_dir, os.path.basename(pdb_file))
    logger.debug(f"Removing waters/ions from {pdb_file} -> {output_file}")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("original", pdb_file)
    for model in structure:
        for chain in list(model):
            for residue in list(chain):
                if residue.get_resname() in exclusion:
                    chain.detach_child(residue.id)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)


def remove_waters_in_dir(base_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Removing waters/ions in {base_path} -> {output_dir}")
    for imgt_pdb in tqdm(os.listdir(base_path), desc="removing waters/ions"):
        if "NONE" in imgt_pdb:
            continue
        pdb_file = os.path.join(base_path, imgt_pdb)
        try:
            remove_water_and_ion(pdb_file, output_dir)
        except Exception as e:
            logger.error(f"Error processing {pdb_file}: {e}. Skipping this file.")
            shutil.copy(pdb_file, os.path.join(output_dir, "ERR"))  # on error, copy original file to output directory
            continue


def crop_pdb(pdb_file: str, output_dir: str, retain_range: Dict[str, Sequence[int]]) -> str:
    """Crop PDB to retain residues within the specified range."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("original", pdb_file)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(pdb_file).replace(".pdb", "")
    output_file = os.path.join(output_dir, f"{base_name}.pdb")
    logger.debug(f"Cropping {pdb_file} -> {output_file} with ranges {retain_range}")
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file, ResidueSelect(retain_range))
    return output_file

def crop_fv_in_dir(base_path: str, output_dir: str, fv_start: int = 1, fv_end: int = 129) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Cropping FV in {base_path} -> {output_dir} (range={fv_start}-{fv_end})")
    for imgt_pdb in tqdm(os.listdir(base_path), desc="cropping fv"):
        retain_range: Dict[str, Sequence[int]] = {}
        pdb_file = os.path.join(base_path, imgt_pdb)
        imgt_pdb = imgt_pdb.replace(".pdb", "")
        parts = imgt_pdb.split("_")
        if len(parts) < 4:
            continue
        h_chain = parts[2]
        l_chain = parts[3]
        if h_chain:
            retain_range[h_chain] = [fv_start, fv_end]
        if l_chain:
            retain_range[l_chain] = [fv_start, fv_end]
        try:
            crop_pdb(pdb_file, output_dir, retain_range)
        except Exception as e:
            logger.error(f"Error processing {pdb_file}: {e}. Skipping this file.")
            shutil.copy(pdb_file, os.path.join(output_dir, "ERR"))  
            continue

def split_antigen_hetatm(pdb_file, output_dir):
    # 1. Parse filename
    basename = os.path.basename(pdb_file)
    name_parts = basename.replace(".pdb", "").split("_")

    if len(name_parts) < 4:
        logger.warning(f"Filename format invalid: {basename}")
        return

    prefix = name_parts[0]
    antigen_id = name_parts[1]  # "A|B"
    heavy_id = name_parts[2]
    light_id = name_parts[3]

    def pick_chain_id(model, used_ids):
        candidates = ["A", "X", "Y", "Z", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        for candidate in candidates:
            if candidate in used_ids:
                continue
            if candidate.upper() in [heavy_id.upper(), light_id.upper()]:
                continue
            if model.has_id(candidate):
                continue
            return candidate
        raise ValueError(f"No available chain id for {basename}")

    # 2. Load structure
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(prefix, pdb_file)
    model = structure[0]

    antigen_ids = [antigen_id]
    if "|" in antigen_id:
        antigen_ids = [s for s in antigen_id.split("|") if s]

    updated_antigen_ids = []
    for antigen_chain_id in antigen_ids:
        # Get the original antigen chain object
        try:
            target_chain = model[antigen_chain_id]
        except KeyError:
            logger.warning(f"Antigen chain {antigen_chain_id} not found in file: {basename}")
            continue

        # 3. Extract HETATM records.
        # 3. Extract HETATM
        # res.id[0] non-space indicates HETATM or water (W)
        het_residues = [res for res in target_chain if res.id[0].strip() != ""]
        if not het_residues:
            logger.warning(f"No HETATM found in chain {antigen_chain_id}; skipping")
            continue

        # 4. Apply the core processing logic.
        # 4. Core logic
        is_conflict = (
            antigen_chain_id.upper() == heavy_id.upper()
            or antigen_chain_id.upper() == light_id.upper()
        )

        if is_conflict:
            # Case A: conflict. Create a new chain ID and move HETATM
            used_ids = {c.id for c in model}
            new_chain_id = pick_chain_id(model, used_ids)
            new_chain = PDB.Chain.Chain(new_chain_id)

            for res in het_residues:
                target_chain.detach_child(res.id)  # remove from original chain
                new_chain.add(res)

            model.add(new_chain)
            logger.info(
                f"Conflict: moved HETATM from {antigen_chain_id} to new chain {new_chain_id} and removed from original"
            )
            updated_antigen_ids.append(new_chain_id)
        else:
            # Case B: no conflict. Clean the original chain, keeping only HETATM
            # Find all standard amino acid residues (res.id[0] == ' ')
            protein_residues = [res for res in target_chain if res.id[0].strip() == ""]
            for res in protein_residues:
                target_chain.detach_child(res.id)  # remove standard amino acids

            logger.info(
                f"Independent chain: removed standard residues from chain {antigen_chain_id}, kept HETATM"
            )
            updated_antigen_ids.append(antigen_chain_id)
    # 5. Save the file.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    updated_antigen_id = "|".join(updated_antigen_ids) if updated_antigen_ids else antigen_id
    updated_basename = f"{prefix}_{updated_antigen_id}_{heavy_id}_{light_id}.pdb"
    output_path = os.path.join(output_dir, updated_basename)
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_path)

    logger.info(f"Saved: {output_path}")

def split_antigen_hetatms_in_dir(base_path: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Fixing antigen HETATM in {base_path} -> {output_dir}")
    for imgt_pdb in tqdm(os.listdir(base_path), desc="split_antigen_hetatms"):
        if "NONE" in imgt_pdb:
            continue
        pdb_file = os.path.join(base_path, imgt_pdb)
        if pdb_file.endswith(".pdb"):
            try:
                split_antigen_hetatm(pdb_file, output_dir)
            except Exception as e:
                logger.error(f"Error processing {pdb_file}: {e}. Skipping this file.")
                shutil.copy(pdb_file, os.path.join(output_dir, "ERR"))  # on error, copy original file to output directory
                continue

def main(base_path, ag_type) -> None:
    base_dir = os.path.dirname(base_path)
    logger.info(f"Starting preprocessing: base_path={base_path} ag_type={ag_type}")
    os.makedirs((os.path.join(base_dir, "ERR")), exist_ok=True)
    # 1. Split PDBs by REMARK chain groups
    split_pdb_in_dir(base_path, output_dir=os.path.join(base_dir, "split_imgt"), agtype=ag_type)
    # 2. Remove water/ions from split PDBs
    remove_waters_in_dir(base_path = os.path.join(base_dir, "split_imgt"), 
                        output_dir=os.path.join(base_dir, "split_imgt_clean"))
    # 2.5 Split antigen HETATM into separate chain if needed
    if ag_type in ["hapten", "carbohydrate"]:
        split_antigen_hetatms_in_dir(os.path.join(base_dir, "split_imgt_clean"), 
                                    output_dir=os.path.join(base_dir, "split_imgt_clean_fixed"))
        crop_fv_in_dir(base_path=os.path.join(base_dir, "split_imgt_clean_fixed"), 
                    output_dir=os.path.join(base_dir, "split_imgt_clean_fixed_fv"))
    # 3. Crop to retain only FV region
    else:
        crop_fv_in_dir(base_path=os.path.join(base_dir, "split_imgt_clean"), 
                    output_dir=os.path.join(base_dir, "split_imgt_clean_fixed_fv"))

        
if __name__ == "__main__":
    # python antibody_dataset_preprocess.py --base-path /home/kechen/peppocketgen/dataset/sabdab/Ab-sugar/imgt --agtype carbohydrate
    # python antibody_dataset_preprocess.py --base-path /home/kechen/peppocketgen/dataset/sabdab/Ab-pep/imgt --agtype peptide
   
    # python antibody_dataset_preprocess.py --base-path /home/kechen/peppocketgen/dataset/sabdab/Ab-protein/imgt --agtype protein
    parser = argparse.ArgumentParser(description="Preprocess antibody PDB files by splitting chains and removing water/ions.")
    parser.add_argument("--base-path", type=str, required=True, help="Directory containing original PDB files.")
    parser.add_argument("--agtype", type=str, required=True, help="Antigen type to filter by (e.g., HAPTEN, PROTEIN).")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    assert args.agtype in ["hapten", "carbohydrate", "peptide", "protein"], "agtype must be either 'hapten', 'carbohydrate', 'peptide', or 'protein'"
    main(args.base_path, args.agtype)
