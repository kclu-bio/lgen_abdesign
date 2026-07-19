import numpy as np
import os
import re
from io import StringIO
from Bio.SeqUtils import seq1
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB import PDBParser, PDBIO

item_type_mapping = {'6d01_J_G_H': 'peptide_standard',
 '8qy9_A_H_L': 'peptide_standard',
 '7rm0_Q_C_D': 'peptide_standard',
 '8f9s_P_H_L': 'peptide_standard',
 '5kzp_D_G_K': 'peptide_standard',
 '6wfz_C_A_B': 'peptide_standard',
 '4wht_k_K_L': 'peptide_standard',
 '7uym_P_H_L': 'peptide_standard',
 '8b8i_N_F_': 'peptide_standard',
 '5mp3_C_A_B': 'peptide_standard',
 '8fb5_O_A_B': 'peptide_standard',
 '3ifo_Q_A_B': 'peptide_standard',
 '4wht_u_U_V': 'peptide_standard',
 '1qkz_P_H_L': 'peptide_standard',
 '6o29_C_A_B': 'peptide_standard',
 '3mls_P_H_L': 'peptide_standard',
 '1u8i_C_B_A': 'peptide_standard',
 '6b5l_A_H_L': 'peptide_standard',
 '8che_D_A_B': 'peptide_standard',
 '8ek1_P_A_B': 'peptide_standard',
 '7sl5_C_A_B': 'peptide_standard',
 '8ux6_E_B_A': 'peptide_standard',
 '3h0t_C_B_A': 'peptide_standard',
 '4xxd_F_E_D': 'peptide_standard',
 '5ea0_P_H_L': 'peptide_standard',
 '6vi1_Q_J_I': 'peptide_standard',
 '9b0a_C_B_': 'peptide_standard',
 '1xf5_Q_D_C': 'peptide_standard',
 '5mp5_K_A_B': 'peptide_standard',
 '3idn_C_B_A': 'peptide_non_standard',
 '5ocy_C_H_L': 'peptide_non_standard',
 '8s73_N_C_D': 'peptide_non_standard',
 '5n7b_I_H_': 'peptide_non_standard',
 '1mpa_P_H_L': 'peptide_non_standard',
 '6h06_K_A_B': 'peptide_non_standard',
 '5mu0_T_G_H': 'peptide_non_standard',
 '6obd_F_H_L': 'peptide_non_standard',
 '6sf6_D_A_B': 'peptide_non_standard',
 '6xli_E_A_B': 'peptide_non_standard',
 '8us8_R_H_L': 'peptide_non_standard',
 '1yuh_X_B_A': 'hapten',
 '1y18_A_H_L': 'hapten',
 '8y57_X_A_B': 'hapten',
 '7y0g_A_H_G': 'hapten',
 '7qt0_A_E_F': 'hapten',
 '1yee_A_H_L': 'hapten',
 '1nd0_A_D_C': 'hapten',
 '1lo3_A_Y_X': 'hapten',
 '1riu_A_H_L': 'hapten',
 '7lmq_A__B': 'hapten',
 '1a6w_A_H_L': 'hapten',
 '5acm_A__B': 'hapten',
 '4hij_A_D_C': 'sugar',
 '6uuh_A_C_D': 'sugar',
 '4odv_A_H_L': 'sugar',
 '6xuk_A_H_L': 'sugar',
 '4hih_A_D_C': 'sugar',
 '3hnv_A_H_L': 'sugar'}

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
    return None

def renumber_pdb_atoms(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    new_serial = 1
    renumbered_lines = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM", "TER   ")):
            line = f"{line[:6]}{new_serial:>5}{line[11:]}"
            new_serial += 1
        renumbered_lines.append(line)

    with open(output_file, 'w') as f:
        f.writelines(renumbered_lines)


def _strip_pdb_terminators(pdb_text: str) -> str:
    kept_lines = []
    for line in pdb_text.splitlines():
        if line.startswith(("MODEL", "ENDMDL", "END")):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def _ligand_to_pdb_block(ligand) -> str:
    if ligand is None:
        return ""
    if isinstance(ligand, str):
        return _strip_pdb_terminators(ligand)

    from Bio.PDB import Model, Structure

    structure = Structure.Structure('ligand')
    model = Model.Model(0)
    structure.add(model)

    if isinstance(ligand, dict):
        chains = ligand.values()
    else:
        chains = [ligand]

    for chain in chains:
        model.add(chain.copy())

    io = PDBIO()
    io.set_structure(structure)
    buffer = StringIO()
    io.save(buffer)
    return _strip_pdb_terminators(buffer.getvalue())


def _compose_model_pdb(model_pdb: str, ligand_block: str = "") -> str:
    model_lines = model_pdb.rstrip().splitlines()
    if ligand_block:
        model_lines.extend(ligand_block.splitlines())
    model_lines.append('ENDMDL')
    return "\n".join(model_lines) + "\n"
    

def create_full_prot(
        atom37: np.ndarray,
        atom37_mask: np.ndarray,
        aatype=None,
        b_factors=None,
        residue_index=None,
        chain_index=None,
        insertion_code=None
    ):
    from data import protein
    assert atom37.ndim == 3
    assert atom37.shape[-1] == 3
    assert atom37.shape[-2] == 37
    n = atom37.shape[0]
    if residue_index is None:
        residue_index = np.arange(n)
    if chain_index is None:
        chain_index = np.zeros(n)
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors,
        insertion_code=insertion_code)

def write_prot_to_pdb(
        prot_pos: np.ndarray,
        file_path: str,
        aatype: np.ndarray=None,
        overwrite=False,
        no_indexing=False,
        b_factors=None,
        residue_index=None,
        chain_index = None,
        insertion_code = None,
        ligand:str = None
    ):
    from data import protein
    ligand_block = _ligand_to_pdb_block(ligand)
    # chain index should be number
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip('.pdb')
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max([
            int(re.findall(r'_(\d+).pdb', x)[0]) for x in existing_files if re.findall(r'_(\d+).pdb', x)
            if re.findall(r'_(\d+).pdb', x)] + [0])
    if not no_indexing:
        save_path = file_path.replace('.pdb', '') + f'_{max_existing_idx+1}.pdb'
    else:
        save_path = file_path

    if aatype is not None:
        assert aatype.ndim == prot_pos.ndim - 2

    with open(save_path, 'w') as f:
        if prot_pos.ndim == 4:
            for t, pos37 in enumerate(prot_pos):
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
                prot = create_full_prot(
                    pos37, atom37_mask, aatype=aatype[t], b_factors=b_factors, 
                    residue_index = residue_index, chain_index = chain_index, insertion_code = insertion_code)
                pdb_prot = protein.to_pdb(prot, model=t + 1, add_end=False)
                f.write(_compose_model_pdb(pdb_prot, ligand_block))
        elif prot_pos.ndim == 3:
            atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7
            prot = create_full_prot(
                prot_pos, atom37_mask, aatype=aatype, b_factors=b_factors, 
                residue_index = residue_index, chain_index=chain_index, insertion_code = insertion_code)
            pdb_prot = protein.to_pdb(prot, model=1, add_end=False)
            f.write(_compose_model_pdb(pdb_prot, ligand_block))
        else:
            raise ValueError(f'Invalid positions shape {prot_pos.shape}')
        
        f.write('END')
    
    try:
        renumber_pdb_atoms(save_path, save_path)
    except Exception as e:
        print(f"Warning: atom renumbering failed: {e}")

    return save_path

def extract_pdb_sequence(pdb_file:str, chain_id:str, residue_range:list):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    chain = None
    for model in structure:
        if chain_id in model:
            chain = model[chain_id]
            break
    # <Residue GLY het=  resseq=111 icode=A>
    if chain is None:
        raise ValueError(f"Chain {chain_id} does not exist in PDB file")
    residues_in_range = []

    for residue in chain:
        res_number = residue.get_id()[1]
        if res_number in residue_range:
            residues_in_range.append(residue.resname)

    sequence = seq1(''.join(residues_in_range))
    
    return sequence

class ResidueSelect:
    # Retain residues in the specified chain ranges.
    # Delete chains and residues not explicitly included.
    def __init__(self, retain_range, mode = "chain"):
        #  retain_range: {"A":[125,130]}
        #  retain_range: {"PROA":[125,130]}
        self.retain_range = retain_range
        self.retain_mode = mode
    
    def accept_model(self, model):
        return True
    
    def accept_chain(self, chain):
        return True
    
    def accept_residue(self, residue):
        if self.retain_mode == "chain":
            chain_id = residue.get_parent().id
        elif self.retain_mode == "segid":
            chain_id = residue.get_segid()
        else:
            raise ValueError
        res_id = residue.id[1] 
        if chain_id in self.retain_range:
            start, end = self.retain_range[chain_id]
            return start <= res_id <= end
        else:
            return False

    def accept_atom(self, atom):
        return True

def crop_pdb(pdb_file, output_dir, retain_range:dict, mode ="chain", verbose = False):
    """Crop PDB file to retain only residues within the specified range.
    
    Args:
        pdb_file: Input PDB file path
        output_dir: Output directory for cropped PDB
        retain_range: Dictionary with chain IDs as keys and [start, end] lists as values
                     e.g., {"H": [1, 126], "L": [1, 126]}
    """
    from Bio.PDB import PDBParser, PDBIO, Select
    import os
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('original', pdb_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(pdb_file).replace('.pdb', '')
    retain_range_str = "_".join([f"{key}_{value[0]}_{value[1]}" for key, value in retain_range.items()])
    output_name = f"{base_name}_{retain_range_str}"
    output_file = os.path.join(output_dir, f"{base_name}_{retain_range_str}.pdb")
    
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file, ResidueSelect(retain_range, mode))
    
    if verbose:
        print(f"Cropped PDB saved to: {output_file}")
    return output_file

def get_atom37_pos(path):
    from data import residue_constants
    ext = path.split(".")[-1]
    if  ext == "pdb":
        parser = PDBParser(QUIET=True)
    elif ext == "cif":
        parser = MMCIFParser(QUIET=True)
    else:
        raise(ValueError)
    structure = parser.get_structure('protein', path)
    atom_positions = []
    residue_index = []
    chain_id= []
    for chain in structure.get_chains():
        for res in chain:
            pos = np.zeros((residue_constants.atom_type_num, 3))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
            atom_positions.append(pos)
            residue_index.append(res.id[1])
            chain_id.append(chain.id)

    return np.array(atom_positions), np.array(residue_index), np.array(chain_id)

def extract_position(atom37, residue_index, chain_id, positions: dict):
    mask = np.zeros(len(residue_index), dtype=bool)
    for chain, res_indices in positions.items():
        chain_mask = (chain_id == chain)
        res_mask = np.isin(residue_index, res_indices)
        mask |= (chain_mask & res_mask)
        
    return atom37[mask], residue_index[mask], chain_id[mask], mask

class ChainSelect:
    def __init__(self, ligand_chain_id, mode="ligand"):
        """
        :param ligand_chain_id: Ligand chain ID, for example 'B'.
        :param mode: "ligand" retains only this chain; "receptor" excludes it.
        """
        self.ligand_chain_id = ligand_chain_id
        self.mode = mode
    def accept_model(self, model):
        return True
    
    def accept_residue(self, residue):
        return True
    
    def accept_chain(self, chain):
        chain_id = chain.get_id()
        
        if self.mode == "ligand":
            # In ligand mode, accept only the specified chain.
            return 1 if chain_id == self.ligand_chain_id else 0
        else:
            # In receptor mode, accept every chain except the specified chain.
            return 0 if chain_id == self.ligand_chain_id else 1
    def accept_atom(self, atom):
        return True

def split_complex(input_pdb, ligand_chain_id, out_path):
    basename = os.path.splitext(os.path.basename(input_pdb))[0]
    if input_pdb.endswith(".pdb"):
        parser = PDBParser(QUIET=True)
    elif input_pdb.endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError
    structure = parser.get_structure("complex", input_pdb)
    io = PDBIO()
    io.set_structure(structure)

    os.makedirs(out_path, exist_ok=True)
    ligand_out = os.path.join(out_path, f"{basename}_ligand.pdb")
    print(f"Extracting ligand chain: {ligand_chain_id}...")
    io.save(ligand_out, ChainSelect(ligand_chain_id, mode="ligand"))
    
    receptor_out = os.path.join(out_path, f"{basename}_receptor.pdb")
    print(f"Extracting receptor chain (Excluding chain {ligand_chain_id})...")
    io.save(receptor_out, ChainSelect(ligand_chain_id, mode="receptor"))
    return ligand_out, receptor_out

def extract_residues_from_filename(filename , mode):
    """Extract residue ranges for all chain tags in a filename."""
    basename = os.path.basename(filename)
    if mode == "filename":
        # Matches like H.27-38,56-64 or L.27-38,56-57,105-117
        matches = re.findall(r'([A-Za-z])\.([\d,\-]+)', basename)
        if not matches:
            return []
        result = []
        for chain, residues_str in matches:
            for part in residues_str.split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                else:
                    start = end = int(part)
                result.append(((chain, start), (chain, end)))
        return result
    
    elif mode == "imgt":

        h_chain = basename.split("_")[2]
        l_chain = basename.split("_")[3]
        range = []
        if h_chain:
            h_range = [((h_chain, 27), (h_chain, 38)),
                    ((h_chain, 56), (h_chain, 64)),
                    ((h_chain, 105), (h_chain, 117))]
            range.extend(h_range)
        if l_chain:
            l_range = [((l_chain, 27), (l_chain, 38)),
                    ((l_chain, 56), (l_chain, 57)),
                    ((l_chain, 105), (l_chain, 117))]
            range.extend(l_range)
        return range

    elif mode == "cothia":
        h_chain = basename.split("_")[2]
        l_chain = basename.split("_")[3]
        range = []
        if h_chain:
            h_range = [((h_chain, 26), (h_chain, 32)),
                    ((h_chain, 52), (h_chain, 56)),
                    ((h_chain, 95), (h_chain, 102))]
            range.extend(h_range)
        if l_chain:
            l_range = [((l_chain, 24), (l_chain, 34)),
                    ((l_chain, 50), (l_chain, 56)),
                    ((l_chain, 89), (l_chain, 97))]
            range.extend(l_range)
        return range
    
    elif mode == "imgt_cdrh3":
        h_chain = basename.split("_")[2]
        range = [((h_chain, 105), (h_chain, 117))]
        return range
    
    elif mode == "cothia_cdrh3":
        h_chain = basename.split("_")[2]
        range = [((h_chain, 95), (h_chain, 102))]

        return range
    else:
        raise ValueError("Invalid mode. Choose from 'filename', 'imgt', 'cothia', 'imgt_cdrh3', 'cothia_cdrh3'.")

def _is_in_the_range(ch_rs_ic, flexible_range):
    for item in flexible_range:
        flexible_range_first, flexible_range_last = item
        if ch_rs_ic[0] != flexible_range_first[0]:
            continue
        r_first, r_last = flexible_range_first[1], flexible_range_last[1]
        rs_ic = ch_rs_ic[1]
        if r_first <= rs_ic <= r_last:
            return True
    return False