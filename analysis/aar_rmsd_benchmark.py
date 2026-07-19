from Bio.SeqUtils import seq1
import models.utils as mu
from data import residue_constants
import torch 
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO
import logging
import os
from tqdm import tqdm
import argparse
import analysis.utils as au
from Bio import PDB

# python aar_rmsd_benchmark.py --model ppg --input_dir /home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-18_14-01_addpro_ablation/epoch=833-step=505404/all_CDR/run_2026-03-24_11-06-11
# python aar_rmsd_benchmark.py --model diffab --mode h3 --input_dir /home/kechen/antibody_design/diffab/results
# python aar_rmsd_benchmark.py --model diffab --mode cdrs --input_dir /home/kechen/antibody_design/diffab/results
# python aar_rmsd_benchmark.py --model abx --input_dir /home/kechen/antibody_design/AbX/output
# python aar_rmsd_benchmark.py --model abx --input_dir /home/kechen/antibody_design/AbX/output_H3
# python aar_rmsd_benchmark.py --model abeg --input_dir /home/kechen/antibody_design/AbEgDiffuser/results
logger = logging.getLogger('AAR_RMSD_Benchmark')
logger.setLevel(logging.INFO)
# imgt
CDR1 = list(range(27, 39+1))
CDR2 = list(range(56, 65+1))
CDR3 = list(range(105, 117+1))

# cothia 
CDRH1 = list(range(26, 32+1))
CDRH2 = list(range(52, 56+1))
CDRH3 = list(range(95, 102+1))

CDRL1 = list(range(24, 34+1))
CDRL2 = list(range(50, 56+1))
CDRL3 = list(range(89, 97+1))

gt_path = [
    "/home/kechen/antibody_design/testset/0217/peptide_standard",
    "/home/kechen/antibody_design/testset/0217/peptide_non_standard",
    "/home/kechen/antibody_design/testset/0217/hapten",
    "/home/kechen/antibody_design/testset/0217/sugar",
]


def get_sequential_cdr_ranges(pdb_file, h_chain_id, l_chain_id):
    """
    Locate original numbering boundaries in the physical sequence.
    Insertion codes are included, while numbering gaps automatically shorten the range.
    """
    # Original CDR numbering definitions (closed intervals)
    cdr_defs = {
        'H': {'CDR1': (27, 37), 'CDR2': (56, 64), 'CDR3': (105, 116)},
        'L': {'CDR1': (27, 37), 'CDR2': (56, 64), 'CDR3': (105, 116)}
    }

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('struct', pdb_file)
    model = structure[0]

    final_results = {}

    for chain_label in ['H', 'L']:
        chain_id = h_chain_id if chain_label == 'H' else l_chain_id
        
        if chain_id not in model:
            continue
            
        # Extract residues in physical order (1-based indices)
        res_list = [res for res in model[chain_id] if PDB.is_aa(res)]
        
        for cdr_name, (orig_start, orig_end) in cdr_defs[chain_label].items():
            # Find residues whose original numbers fall within [orig_start, orig_end].
            # res.get_id()[1] uses only the numeric part, so 3A and 3B are both treated as 3.
            matched_indices = [
                i + 1 for i, res in enumerate(res_list) 
                if orig_start <= res.get_id()[1] <= orig_end
            ]
            
            if matched_indices:
                # The sequential range spans the first through last matched residues.
                final_results[f"CDR{chain_label}{cdr_name[-1]}"] = list(range(min(matched_indices), max(matched_indices) + 1))
            else:
                final_results[f"CDR{chain_label}{cdr_name[-1]}"] = (None, None)

    return final_results

def find_gt_pdb_abx(basename, gt_paths:list):
    for gt_path in gt_paths:
        for gt_file in os.listdir(gt_path):
            if gt_file.startswith(basename):
                antigen_type = gt_path.split("/")[-1]
                return os.path.join(gt_path, gt_file)
    return None

def calculate_identity_enumerate(str1, str2):
    if len(str1) != len(str2):
        raise ValueError("The lengths of the two strings must be equal.")
    
    length = len(str1)
    if length == 0:
        return 1
    matches = 0
    for i, char1 in enumerate(str1):
        if char1 == str2[i]:
            matches += 1
    
    return matches / length


_PDB_PARSER = PDBParser(QUIET=True)

def parse_structure(path):
    """Parse PDB once."""
    return _PDB_PARSER.get_structure("protein", path)


def get_chain_from_structure(structure, chain_id):
    """Get first model's chain."""
    for model in structure:
        if chain_id in model:
            return model[chain_id]
        break
    raise KeyError(f"Chain {chain_id} not found in structure.")


def extract_chain_atom_pos_and_residx(structure, chain_id):
    """
    Extract per-residue atom positions for all 37 atom types (or atom_type_num),
    and residue indices, for one chain.
    Returns:
      atom_pos: (L, atom_type_num, 3) float32
      res_idx: (L,) int
    """
    chain = get_chain_from_structure(structure, chain_id)

    atom_positions = []
    residue_index = []

    for res in chain:
        # Keep the same behavior as your original code (no filtering on hetero/insertion)
        # [0, 37 ,3]
        pos = np.zeros((residue_constants.atom_type_num, 3), dtype=np.float32)
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord.astype(np.float32)
        # Preserve insertion codes.
        res_str_id = f"{res.id[1]}{res.id[2]}" 
        residue_index.append(res_str_id)
        atom_positions.append(pos)
        #residue_index.append(res.id[1])

    return np.asarray(atom_positions, dtype=np.float32), np.asarray(residue_index, dtype=object)


def build_mask_from_residx(res_idx, selected_resnums):
    """
    res_idx: (L,) string array, for example ["26 ", "27 ", "100A"]
    selected_resnums: list[int], for example [26, 27, 100]
    """
    # Compare the numeric portion so insertion codes such as 100A remain supported.
    selected_resnums_set = set(selected_resnums)
    
    mask = []
    for r_str in res_idx:
        # r_str has a form such as "100A" or "26 "; extract its numeric portion.
        num_part = int(''.join([c for c in r_str if c.isdigit()]))
        if num_part in selected_resnums_set:
            mask.append(1)
        else:
            mask.append(0)
    if sum(mask) == 0:
        logger.warning(f"No residues matched for selected_resnums {selected_resnums} in res_idx {res_idx}")
        raise ValueError(f"No residues{res_idx} matched for selected_resnums {selected_resnums}.")
    return np.asarray(mask, dtype=np.int32)


def align_by_residx(gt_pos, gt_residx, gen_pos, gen_residx, mode = "default"):
    """
    Align two chains by residue index (res.id[1]) intersection, preserving order.
    This avoids min_length truncation misalignment.
    Returns aligned (gt_pos_a, gen_pos_a, residx_a).
    """
    # Map residx -> position in array (first occurrence kept)
    # If duplicate residx occurs (rare), this keeps first.
    if mode == "abx":
        # Force alignment by truncating both chains in physical sequence order.
        min_length = min(gt_pos.shape[0], gen_pos.shape[0])
        # Generate 1-based sequential IDs in PDB residue-ID form for downstream masking.
        seq_residx = np.asarray([f"{i} " for i in range(1, min_length + 1)], dtype=object)
        return gt_pos[:min_length], gen_pos[:min_length], seq_residx
    
    gt_map = {}
    for i, r in enumerate(gt_residx.tolist()):
        if r not in gt_map:
            gt_map[r] = i

    gen_map = {}
    for i, r in enumerate(gen_residx.tolist()):
        if r not in gen_map:
            gen_map[r] = i

    common = sorted(set(gt_map.keys()) & set(gen_map.keys()))
    if len(common) == 0:
        # Return empty aligned arrays
        return (gt_pos[:0], gen_pos[:0], np.asarray([], dtype=object))

    gt_idx = np.asarray([gt_map[r] for r in common], dtype=np.int64)
    gen_idx = np.asarray([gen_map[r] for r in common], dtype=np.int64)

    return gt_pos[gt_idx], gen_pos[gen_idx], np.asarray(common, dtype=object)


def calc_rmsd_from_aligned(gt_pos_aligned, gen_pos_aligned, residx_aligned, selected_resnums):
    """
    Compute RMSD for selected residues using mu.calc_rmsd exactly as you used it.
    Uses backbone atoms as [:, :3] (N, CA, C) like your original code, but stores full 37 atoms.
    """
    if gt_pos_aligned.shape != gen_pos_aligned.shape:
        raise ValueError(f"Aligned positions shape mismatch: {gt_pos_aligned.shape} vs {gen_pos_aligned.shape}")

    if gt_pos_aligned.shape[0] == 0:
        return np.nan

    mask_res = build_mask_from_residx(residx_aligned, selected_resnums)  # (L,)
    mask_atom = torch.from_numpy(mask_res)[..., None].repeat(1, 3).reshape(-1)  # (L*3,)

    sample_bb_pos = gt_pos_aligned[:, :3].reshape(-1, 3)     # (L*3,3)
    folded_bb_pos = gen_pos_aligned[:, :3].reshape(-1, 3)    # (L*3,3)

    rmsd = mu.calc_rmsd(
        mask=mask_atom,
        sample_bb_pos=sample_bb_pos,
        folded_bb_pos=folded_bb_pos
    )
    return rmsd


def calc_rmsd_multichain_from_cache(chain_cache, residue_index_dict):
    """
    chain_cache[(chain_id)] = {
        'gt_pos': (L,atom_type_num,3), 'gen_pos': same, 'residx': (L,)
    } where arrays already aligned by residx intersection
    residue_index_dict: {chain_id: selected_resnums_list}
    """
    gt_all = []
    gen_all = []
    mask_all = []

    for chain_id, selected_resnums in residue_index_dict.items():
        if chain_id not in chain_cache:
            continue
        gt_pos = chain_cache[chain_id]["gt_pos"]
        gen_pos = chain_cache[chain_id]["gen_pos"]
        residx = chain_cache[chain_id]["residx"]

        if gt_pos.shape[0] == 0:
            continue

        gt_all.append(gt_pos)
        gen_all.append(gen_pos)
        mask_all.append(build_mask_from_residx(residx, selected_resnums))  # (L,)

    if len(gt_all) == 0:
        return np.nan

    gt_all = np.concatenate(gt_all, axis=0)
    gen_all = np.concatenate(gen_all, axis=0)
    mask_all = np.concatenate(mask_all, axis=0)  # (sumL,)

    mask_atom = torch.from_numpy(mask_all)[..., None].repeat(1, 3).reshape(-1)
    sample_bb_pos = gt_all[:, :3].reshape(-1, 3)
    folded_bb_pos = gen_all[:, :3].reshape(-1, 3)

    rmsd = mu.calc_rmsd(
        mask=mask_atom,
        sample_bb_pos=sample_bb_pos,
        folded_bb_pos=folded_bb_pos
    )
    return rmsd


def calc_aar_cached(gt_pdb_path, gen_pdb_path, chain_id, selected_resnums):
    """
    Compute AAR with au.extract_pdb_sequence using the selected residue numbers.
    The caller can extract the union of all CDRs once and split it to avoid three calls.
    This single-range interface is retained as a fallback.
    """
    gt_seq = au.extract_pdb_sequence(gt_pdb_path, chain_id, selected_resnums)
    gen_seq = au.extract_pdb_sequence(gen_pdb_path, chain_id, selected_resnums)
    return calculate_identity_enumerate(gt_seq, gen_seq)


def calc_aar_three_cdrs_onepass(gt_pdb_path, gen_pdb_path, chain_id, cdr1, cdr2, cdr3):
    """
    Extract the union of CDR1/2/3 once, then compute AAR for each segment by position.
    This requires au.extract_pdb_sequence to follow ascending residue_index_list order.
    Build and sort the union explicitly and construct a mask for each CDR.
    """
    all_res = sorted(set(cdr1) | set(cdr2) | set(cdr3))
    gt_seq_all = au.extract_pdb_sequence(gt_pdb_path, chain_id, all_res)
    gen_seq_all = au.extract_pdb_sequence(gen_pdb_path, chain_id, all_res)

    if len(gt_seq_all) != len(all_res) or len(gen_seq_all) != len(all_res):
        # Fall back to separate extraction if missing residues cause a length mismatch.
        return (
            calc_aar_cached(gt_pdb_path, gen_pdb_path, chain_id, cdr1),
            calc_aar_cached(gt_pdb_path, gen_pdb_path, chain_id, cdr2),
            calc_aar_cached(gt_pdb_path, gen_pdb_path, chain_id, cdr3),
        )

    # Build masks over the all_res indices.
    all_res_arr = np.asarray(all_res, dtype=np.int32)
    m1 = np.isin(all_res_arr, np.asarray(cdr1, dtype=np.int32))
    m2 = np.isin(all_res_arr, np.asarray(cdr2, dtype=np.int32))
    m3 = np.isin(all_res_arr, np.asarray(cdr3, dtype=np.int32))

    gt_arr = np.frombuffer(gt_seq_all.encode("ascii"), dtype="S1")
    gen_arr = np.frombuffer(gen_seq_all.encode("ascii"), dtype="S1")

    def _aar(mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return np.nan
        return (gt_arr[idx] == gen_arr[idx]).mean()

    return _aar(m1), _aar(m2), _aar(m3)

def process_ppg(input_dir):
    gt_struct_cache = {}
    data = []

    for generated_file in tqdm(os.listdir(input_dir), desc="processing_files"):
        if not generated_file.endswith(".pdb"):
            continue

        gt_file_full, antigen_type = au.find_gt_pdb(generated_file, gt_path)
        if gt_file_full is None:
            continue

        generated_file_full = os.path.join(input_dir, generated_file)
        base_name = generated_file.split("_gen_len")[0]

        parts = base_name.split("_")
        H_chain = parts[2] if len(parts) > 2 else ""
        L_chain = parts[3] if len(parts) > 3 else ""

        row = {"basename": base_name, "ag_type": antigen_type}

        # Cache parsed ground truth structures; generated structures differ per sample.
        if gt_file_full not in gt_struct_cache:
            gt_struct_cache[gt_file_full] = parse_structure(gt_file_full)
        gt_struct = gt_struct_cache[gt_file_full]
        gen_struct = parse_structure(generated_file_full)

        # Cache aligned coordinates and residue indices for RMSD and multichain reuse.
        chain_cache = {}

        for tag, chain_id in [("H", H_chain), ("L", L_chain)]:
            if not chain_id:
                continue

            # 1) AAR: extract the full CDR union once for GT and GEN, then split it.
            aar1, aar2, aar3 = calc_aar_three_cdrs_onepass(
                gt_file_full, generated_file_full, chain_id, CDR1, CDR2, CDR3
            )
            row[f"{tag}_CDR1_AAR"] = aar1
            row[f"{tag}_CDR2_AAR"] = aar2
            row[f"{tag}_CDR3_AAR"] = aar3

            # 2) Coordinates: extract each chain once and align by residue-index intersection.
            gt_pos, gt_residx = extract_chain_atom_pos_and_residx(gt_struct, chain_id)
            gen_pos, gen_residx = extract_chain_atom_pos_and_residx(gen_struct, chain_id)

            gt_pos_a, gen_pos_a, residx_a = align_by_residx(gt_pos, gt_residx, gen_pos, gen_residx)
            chain_cache[chain_id] = {"gt_pos": gt_pos_a, "gen_pos": gen_pos_a, "residx": residx_a}

            # 3) RMSD: reuse aligned arrays with different masks.
            row[f"{tag}_CDR1_rmsd"] = calc_rmsd_from_aligned(gt_pos_a, gen_pos_a, residx_a, CDR1)
            row[f"{tag}_CDR2_rmsd"] = calc_rmsd_from_aligned(gt_pos_a, gen_pos_a, residx_a, CDR2)
            row[f"{tag}_CDR3_rmsd"] = calc_rmsd_from_aligned(gt_pos_a, gen_pos_a, residx_a, CDR3)

        # 4) Multichain RMSD_CDR: reuse chain_cache without reparsing.
        residue_index_dict = {}
        if H_chain:
            residue_index_dict[H_chain] = CDR1 + CDR2 + CDR3
        if L_chain:
            residue_index_dict[L_chain] = CDR1 + CDR2 + CDR3

        row["RMSD_CDR"] = calc_rmsd_multichain_from_cache(chain_cache, residue_index_dict)

        data.append(row)

    df = pd.DataFrame(data)

    return df

def process_diffab(input_dir, sample_num, mode):
    # gen base dir: "/home/kechen/antibody_design/diffab/results"
    if mode == "h3":
        mode_dir = ["codesign_single", "H_CDR3"]
    elif mode == "cdrs":
        mode_dir = ["codesign_multicdrs", "MultipleCDRs"]
    gt_struct_cache = {}
    sub_dir = ["hapten_non_ligand", "peptide_standard", "sugar_non_ligand"]
    for sub in sub_dir:
        type_dir = os.path.join(input_dir, sub, mode_dir[0])
        for gen_dir in tqdm(os.listdir(type_dir), desc="processing_files"):
            full_gen_dir = os.path.join(type_dir, gen_dir)
            gt_file_full = os.path.join(full_gen_dir, "reference.pdb") # Original structure
            for i in range(sample_num):
                gen_file_full = os.path.join(full_gen_dir, mode_dir[1], f"{str(i).zfill(4)}.pdb")  # DiffAb sample
                if not os.path.exists(gen_file_full) and os.path.exists(gt_file_full):
                    print(f"Generated file {gen_file_full} does not exist, skipping.")
                    continue
                base_name = f"{gen_dir.split('.pdb')[0]}_{i}"
                H_chain = base_name.split("_")[2]
                L_chain = base_name.split("_")[3]
                row = {"basename": base_name,"ag_type": sub.split("_non_ligand")[0]}

                if gt_file_full not in gt_struct_cache:
                    gt_struct_cache[gt_file_full] = parse_structure(gt_file_full)
                gt_struct = gt_struct_cache[gt_file_full]
                gen_struct = parse_structure(gen_file_full)

                chain_cache = {}

                for tag, chain_id in [("H", H_chain), ("L", L_chain)]:
                    if not chain_id:
                        continue
                    if tag == "H":
                        CDR1, CDR2, CDR3 = CDRH1, CDRH2, CDRH3
                    else:
                        CDR1, CDR2, CDR3 = CDRL1, CDRL2, CDRL3
                    # 1) AAR: extract the full CDR union once for GT and GEN, then split it.
                    aar1, aar2, aar3 = calc_aar_three_cdrs_onepass(
                        gt_file_full, gen_file_full, chain_id, CDR1, CDR2, CDR3
                    )
                    row[f"{tag}_CDR1_AAR"] = aar1
                    row[f"{tag}_CDR2_AAR"] = aar2
                    row[f"{tag}_CDR3_AAR"] = aar3

                    # 2) Coordinates: extract each chain once and align by residue-index intersection.
                    gt_pos, gt_residx = extract_chain_atom_pos_and_residx(gt_struct, chain_id)
                    gen_pos, gen_residx = extract_chain_atom_pos_and_residx(gen_struct, chain_id)

                    gt_pos_a, gen_pos_a, residx_a = align_by_residx(gt_pos, gt_residx, gen_pos, gen_residx)
                    chain_cache[chain_id] = {"gt_pos": gt_pos_a, "gen_pos": gen_pos_a, "residx": residx_a}

                    # 3) RMSD: reuse aligned arrays with different masks.
                    row[f"{tag}_CDR1_rmsd"] = calc_rmsd_from_aligned(gt_pos_a, gen_pos_a, residx_a, CDR1)
                    row[f"{tag}_CDR2_rmsd"] = calc_rmsd_from_aligned(gt_pos_a, gen_pos_a, residx_a, CDR2)
                    row[f"{tag}_CDR3_rmsd"] = calc_rmsd_from_aligned(gt_pos_a, gen_pos_a, residx_a, CDR3)
            
                # 4) Multichain RMSD_CDR: reuse chain_cache without reparsing.
                residue_index_dict = {}
                if H_chain:
                    residue_index_dict[H_chain] = CDR1 + CDR2 + CDR3
                if L_chain:
                    residue_index_dict[L_chain] = CDR1 + CDR2 + CDR3

                row["RMSD_CDR"] = calc_rmsd_multichain_from_cache(chain_cache, residue_index_dict)

                data.append(row)

    df = pd.DataFrame(data)

    return df

def process_abx(input_dir, sample_num):
    # input_dir = "/home/kechen/antibody_design/AbX/output"
    # input_dir = "/home/kechen/antibody_design/AbX/output_H3"
    data = []
    gt_struct_cache = {}
    gt_path = ["/home/kechen/antibody_design/testset/0217/peptide", 
            "/home/kechen/antibody_design/testset/0217/hapten",
            "/home/kechen/antibody_design/testset/0217/sugar"]
    for item_name in tqdm(os.listdir(input_dir)):
        # item_name: 1qkz_H_L_P
        full_gen_dir = os.path.join(input_dir, item_name, "design")
        gt_file_full = os.path.join(full_gen_dir, "reference", f"{item_name}.pdb") # Original structure

        H_chain = item_name.split("_")[1]
        L_chain = item_name.split("_")[2]
        ligand_chain = item_name.split("_")[3]
        gt_numbered_file = find_gt_pdb_abx(f"{item_name.split('_')[0]}_{ligand_chain}_{H_chain}_{L_chain}", gt_path) # Numbered original structure
        cdr_dict = get_sequential_cdr_ranges(gt_numbered_file, H_chain, L_chain)
        for i in range(sample_num):
            gen_file_full = os.path.join(full_gen_dir, f"000{i}", f"{item_name}.pdb") 
            base_name = f"{item_name}_{i}"
            # 1qkz_H_L_P_0
            row = {}
            
            H_chain = base_name.split("_")[1]
            L_chain = base_name.split("_")[2]
            row = {"basename": base_name,"ag_type": "peptide_standard"}

            if gt_file_full not in gt_struct_cache:
                gt_struct_cache[gt_file_full] = parse_structure(gt_file_full)
            gt_struct = gt_struct_cache[gt_file_full]
            gen_struct = parse_structure(gen_file_full)

            chain_cache = {}

            for tag, chain_id in [("H", H_chain), ("L", L_chain)]:
                if not chain_id:
                    continue
                
                CDR1,CDR2,CDR3 = cdr_dict[f"CDR{tag}1"], cdr_dict[f"CDR{tag}2"], cdr_dict[f"CDR{tag}3"]
                # 1) AAR: extract the full CDR union once for GT and GEN, then split it.
                aar1, aar2, aar3 = calc_aar_three_cdrs_onepass(
                    gt_file_full, gen_file_full, chain_id, CDR1, CDR2, CDR3
                )
                row[f"{tag}_CDR1_AAR"] = aar1
                row[f"{tag}_CDR2_AAR"] = aar2
                row[f"{tag}_CDR3_AAR"] = aar3

                # 2) Coordinates: extract each chain once and align by residue-index intersection.
                gt_pos, gt_residx = extract_chain_atom_pos_and_residx(gt_struct, chain_id)
                gen_pos, gen_residx = extract_chain_atom_pos_and_residx(gen_struct, chain_id)
                gt_pos_a, gen_pos_a, residx_a = align_by_residx(gt_pos, gt_residx, gen_pos, gen_residx, mode = "abx")
                chain_cache[chain_id] = {"gt_pos": gt_pos_a, "gen_pos": gen_pos_a, "residx": residx_a}

                # 3) RMSD: reuse aligned arrays with different masks.
                row[f"{tag}_CDR1_rmsd"] = calc_rmsd_from_aligned(gt_pos_a, gen_pos_a, residx_a, CDR1)
                row[f"{tag}_CDR2_rmsd"] = calc_rmsd_from_aligned(gt_pos_a, gen_pos_a, residx_a, CDR2)
                row[f"{tag}_CDR3_rmsd"] = calc_rmsd_from_aligned(gt_pos_a, gen_pos_a, residx_a, CDR3)
        
            # 4) Multichain RMSD_CDR: reuse chain_cache without reparsing.
            residue_index_dict = {}
            if H_chain:
                CDR1,CDR2,CDR3 = cdr_dict[f"CDRH1"], cdr_dict[f"CDRH2"], cdr_dict[f"CDRH3"]
                residue_index_dict[H_chain] = CDR1 + CDR2 + CDR3
            if L_chain:
                CDR1,CDR2,CDR3 = cdr_dict[f"CDRL1"], cdr_dict[f"CDRL2"], cdr_dict[f"CDRL3"]
                residue_index_dict[L_chain] = CDR1 + CDR2 + CDR3

            row["RMSD_CDR"] = calc_rmsd_multichain_from_cache(chain_cache, residue_index_dict)
        
            data.append(row)

    df = pd.DataFrame(data)
    return df

def process_abeg(input_dir, sample_num):
    # gen_base_dir = "/home/kechen/antibody_design/AbEgDiffuser/results"
    gt_struct_cache = {}
    sub_dir = ["hapten_non_ligand", "peptide_standard", "sugar_non_ligand"]
    for sub in sub_dir:
        type_dir = os.path.join(input_dir, sub, "codesign_single")
        for gen_dir in tqdm(os.listdir(type_dir), desc="processing_files"):
            full_gen_dir = os.path.join(type_dir, gen_dir)
            gt_file_full = os.path.join(full_gen_dir, "reference.pdb") # Original structure
            for i in range(sample_num):
                gen_file_full = os.path.join(full_gen_dir, "H_CDR3", f"{str(i).zfill(4)}.pdb")  # AbEgDiffuser sample

                base_name = f"{gen_dir.split('.pdb')[0]}_{i}"
                H_chain = base_name.split("_")[2]
                L_chain = base_name.split("_")[3]
                row = {"basename": base_name,"ag_type": sub.split("_non_ligand")[0]}  
                if gt_file_full not in gt_struct_cache:
                    gt_struct_cache[gt_file_full] = parse_structure(gt_file_full)
                gt_struct = gt_struct_cache[gt_file_full]
                gen_struct = parse_structure(gen_file_full)

                chain_cache = {}

                for tag, chain_id in [("H", H_chain), ("L", L_chain)]:
                    if not chain_id:
                        continue
                    if tag == "H":
                        CDR1, CDR2, CDR3 = CDRH1, CDRH2, CDRH3
                    else:
                        CDR1, CDR2, CDR3 = CDRL1, CDRL2, CDRL3
                    # 1) AAR: extract the full CDR union once for GT and GEN, then split it.
                    aar1, aar2, aar3 = calc_aar_three_cdrs_onepass(
                        gt_file_full, gen_file_full, chain_id, CDR1, CDR2, CDR3
                    )
                    row[f"{tag}_CDR1_AAR"] = aar1
                    row[f"{tag}_CDR2_AAR"] = aar2
                    row[f"{tag}_CDR3_AAR"] = aar3

                    # 2) Coordinates: extract each chain once and align by residue-index intersection.
                    gt_pos, gt_residx = extract_chain_atom_pos_and_residx(gt_struct, chain_id)
                    gen_pos, gen_residx = extract_chain_atom_pos_and_residx(gen_struct, chain_id)

                    gt_pos_a, gen_pos_a, residx_a = align_by_residx(gt_pos, gt_residx, gen_pos, gen_residx)
                    chain_cache[chain_id] = {"gt_pos": gt_pos_a, "gen_pos": gen_pos_a, "residx": residx_a}

                    # 3) RMSD: reuse aligned arrays with different masks.
                    row[f"{tag}_CDR1_rmsd"] = calc_rmsd_from_aligned(gt_pos_a, gen_pos_a, residx_a, CDR1)
                    row[f"{tag}_CDR2_rmsd"] = calc_rmsd_from_aligned(gt_pos_a, gen_pos_a, residx_a, CDR2)
                    row[f"{tag}_CDR3_rmsd"] = calc_rmsd_from_aligned(gt_pos_a, gen_pos_a, residx_a, CDR3)
            
                # 4) Multichain RMSD_CDR: reuse chain_cache without reparsing.
                residue_index_dict = {}
                if H_chain:
                    residue_index_dict[H_chain] = CDR1 + CDR2 + CDR3
                if L_chain:
                    residue_index_dict[L_chain] = CDR1 + CDR2 + CDR3

                row["RMSD_CDR"] = calc_rmsd_multichain_from_cache(chain_cache, residue_index_dict)

                data.append(row)

    df = pd.DataFrame(data)
    return df            


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_dir", type=str,required=True, help="Directory containing generated PDB files")
    args.add_argument("--model", choices=["ppg", "diffab", "abx", "abeg"], type=str,required=True, help="Directory containing generated PDB files")
    args.add_argument("--mode", choices=["h3", "cdrs"], required=False, type=str, help="Directory containing generated PDB files")

    args = args.parse_args()
    data = []
    if args.model == "ppg":
        df = process_ppg(args.input_dir)
    elif args.model == "diffab":
        df = process_diffab(args.input_dir, sample_num=8, mode=args.mode)
    elif args.model == "abx":
        df = process_abx(args.input_dir, sample_num=8)
    elif args.model == "abeg":
        df = process_abeg(args.input_dir, sample_num=8)

    output_csv_name = "aar_rmsd_benchmark.csv" if args.model != "diffab" else f"aar_rmsd_benchmark_{args.mode}.csv"
    df.to_csv(os.path.join(args.input_dir, output_csv_name), index=False)
    logger.info(f"Saved results to {os.path.join(args.input_dir, output_csv_name)}")
    columns_to_average = [
        "H_CDR1_AAR", "H_CDR2_AAR", "H_CDR3_AAR",
        "L_CDR1_AAR", "L_CDR2_AAR", "L_CDR3_AAR",
        "H_CDR1_rmsd", "H_CDR2_rmsd", "H_CDR3_rmsd",
        "L_CDR1_rmsd", "L_CDR2_rmsd", "L_CDR3_rmsd",
        "RMSD_CDR",
    ]
    averages = df[columns_to_average].mean(numeric_only=True)
    grouped_mean = df.groupby('ag_type').mean(numeric_only=True)

    grouped_output_path = os.path.join(args.input_dir, f"{output_csv_name.split('.')[0]}_grouped_mean.csv")
    grouped_mean.to_csv(grouped_output_path)
    logger.info(f"Saved grouped mean results to {grouped_output_path}")