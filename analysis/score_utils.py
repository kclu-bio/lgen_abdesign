from collections import namedtuple

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import *
from rdkit.Chem.ChemicalFeatures import *
from rdkit.Chem.EnumerateStereoisomers import (EnumerateStereoisomers, StereoEnumerationOptions)
from rdkit.Chem.rdChemReactions import *
from rdkit.Chem.rdDepictor import *
from rdkit.Chem.rdDistGeom import *
from rdkit.Chem.rdFingerprintGenerator import *
from rdkit.Chem.rdForceFieldHelpers import *
from rdkit.Chem.rdMolAlign import *
from rdkit.Chem.rdMolDescriptors import *
from rdkit.Chem.rdMolEnumerator import *
from rdkit.Chem.rdMolTransforms import *
from rdkit.Chem.rdPartialCharges import *
from rdkit.Chem.rdqueries import *
from rdkit.Chem.rdReducedGraphs import *
from rdkit.Chem.rdShapeHelpers import *
from rdkit.RDLogger import logger

try:
  from rdkit.Chem.rdSLNParse import *
except ImportError:
  pass

from Bio.Data import IUPACData
from Bio.PDB import PDBParser, PDBIO, Select

logger = logger()

from openmm import unit
from openmm import app

def check_overlapping_pairs(fixer):
    atoms = list(fixer.topology.atoms())
    positions = list(fixer.positions)
    overlap_pairs = []
    for bond in fixer.topology.bonds():
        i = bond[0].index
        j = bond[1].index
        delta = positions[i] - positions[j]
        dist = unit.norm(delta).value_in_unit(unit.nanometer)
        if dist == 0.0:
            overlap_pairs.append((atoms[i], atoms[j]))
    return overlap_pairs

def delete_overlap_atoms(fixer, overlap_pairs):
    atoms_to_delete = set()
    for a1, a2 in overlap_pairs:
        atoms_to_delete.add(a1)
        atoms_to_delete.add(a2)
    modeller = app.Modeller(fixer.topology, fixer.positions)
    modeller.delete(list(atoms_to_delete))
    fixer.topology = modeller.topology
    fixer.positions = modeller.positions
    print(f"Deleted {len(atoms_to_delete)} overlapping atoms.")

def fix_obabel_input_pdb(input_pdb, output_pdb):
  # Prevent atoms such as CE1 and CD1 from being interpreted as Ce and Cd.
  element_symbols = set(IUPACData.atom_weights.keys())

  def normalize_atom_name(atom_name):
    name = atom_name.strip()
    if not name:
      return atom_name

    base = name.rstrip("0123456789")
    if len(base) >= 2:
      first_two = base[:2]
      first_two_capitalized = first_two.capitalize()
      if first_two_capitalized in element_symbols:
        return f" {first_two}".ljust(4)
    return atom_name

  input_path = Path(input_pdb)
  output_path = Path(output_pdb)

  with input_path.open("r") as fin:
    lines = fin.readlines()

  out = []
  for line in lines:
    if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 16:
      atom_field = line[12:16]
      new_atom_field = normalize_atom_name(atom_field)
      if new_atom_field != atom_field:
        line = line[:12] + new_atom_field + line[16:]
    out.append(line)

  with output_path.open("w") as fout:
    fout.writelines(out)

def fix_rdkit_input_pdb(input_path: str | Path, output_path: str | Path) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="ascii", errors="replace") as f:
        lines = f.readlines()

    out = []
    for line in lines:
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            out.append(line)
            continue

        has_nl = line.endswith("\n")
        chars = list(line[:-1] if has_nl else line)
        if chars[12].isalpha() and not chars[75].isalpha():
            new_chars = chars[:12]+[" "]+chars[12:16]+chars[17:]
        elif chars[12].isalpha() and chars[75].isalpha():
            new_chars = chars[:12]+[" "]+chars[12:16]+chars[17:75]+[" "]+chars[75:]
        elif not chars[12].isalpha() and chars[75].isalpha():
            new_chars = chars[:75]+[" "]+chars[75:78]
        else:
            new_chars = chars

        out.append("".join(new_chars) + ("\n" if has_nl else ""))

    with output_path.open("w", encoding="ascii", errors="replace") as f:
        f.writelines(out)


def get_ligand_name(pdb_file, chain_id):
    """
    Get the ligand name from a specified PDB file and chain ID.
    :param pdb_file: Path to the PDB file.
    :param chain_id: Chain ID, for example 'A' or 'B'.
    :return: Ligand name or None.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_file)
        for model in structure:
            if chain_id in model:
                target_chain = model[chain_id]
                for residue in target_chain:
                    # In Biopython, residue.id[0] means:
                    # ' ': standard amino acid.
                    # 'W': water.
                    # 'H_xxx': hetero atom or ligand.
                    
                    if residue.id[0] != ' ':
                        if residue.id[0].startswith('H_'):
                          return residue.resname
                        
    except Exception as e:
        print(f"Error parsing PDB file: {e}")
        return None
    
    return None

def AssignBondOrdersFromTemplateNew(refmol, mol):
  """ assigns bond orders to a molecule based on the
    bond orders in a template molecule

    Arguments
      - refmol: the template molecule
      - mol: the molecule to assign bond orders to

    An example, start by generating a template from a SMILES
    and read in the PDB structure of the molecule

    >>> import os
    >>> from rdkit.Chem import AllChem
    >>> template = AllChem.MolFromSmiles("CN1C(=NC(C1=O)(c2ccccc2)c3ccccc3)N")
    >>> mol = AllChem.MolFromPDBFile(os.path.join(RDConfig.RDCodeDir, 'Chem', 'test_data', '4DJU_lig.pdb'))
    >>> len([1 for b in template.GetBonds() if b.GetBondTypeAsDouble() == 1.0])
    8
    >>> len([1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 1.0])
    22

    Now assign the bond orders based on the template molecule

    >>> newMol = AllChem.AssignBondOrdersFromTemplate(template, mol)
    >>> len([1 for b in newMol.GetBonds() if b.GetBondTypeAsDouble() == 1.0])
    8

    Note that the template molecule should have no explicit hydrogens
    else the algorithm will fail.

    It also works if there are different formal charges (this was github issue 235):

    >>> template=AllChem.MolFromSmiles('CN(C)C(=O)Cc1ccc2c(c1)NC(=O)c3ccc(cc3N2)c4ccc(c(c4)OC)[N+](=O)[O-]')
    >>> mol = AllChem.MolFromMolFile(os.path.join(RDConfig.RDCodeDir, 'Chem', 'test_data', '4FTR_lig.mol'))
    >>> AllChem.MolToSmiles(mol)
    'COC1CC(C2CCC3C(O)NC4CC(CC(O)N(C)C)CCC4NC3C2)CCC1N(O)O'
    >>> newMol = AllChem.AssignBondOrdersFromTemplate(template, mol)
    >>> AllChem.MolToSmiles(newMol)
    'COc1cc(-c2ccc3c(c2)Nc2ccc(CC(=O)N(C)C)cc2NC3=O)ccc1[N+](=O)[O-]'

    """
  refmol2 = rdchem.Mol(refmol)
  mol2 = rdchem.Mol(mol)
  # do the molecules match already?
  matching = mol2.GetSubstructMatch(refmol2)
  if not matching:  # no, they don't match
    # check if bonds of mol are SINGLE
    for b in mol2.GetBonds():
      if b.GetBondType() != BondType.SINGLE:
        b.SetBondType(BondType.SINGLE)
        b.SetIsAromatic(False)
    # set the bonds of mol to SINGLE
    for b in refmol2.GetBonds():
      b.SetBondType(BondType.SINGLE)
      b.SetIsAromatic(False)
    # set atom charges to zero;
    for a in refmol2.GetAtoms():
      a.SetFormalCharge(0)
    for a in mol2.GetAtoms():
      a.SetFormalCharge(0)

    # refmol must have no more atoms or bonds than mol, otherwise matching fails.
    # matching contains atom indices from mol.
    # Element i is the mol atom index matching atom i in refmol.
    matching = mol2.GetSubstructMatches(refmol2, uniquify=False)
    rw_mol2 = Chem.RWMol(mol2)
    bond_indices = []
    for bond in rw_mol2.GetBonds():
        bond_indices.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    for idx1, idx2 in bond_indices:
        rw_mol2.RemoveBond(idx1, idx2)
    # do the molecules match now?
    if matching:
      if len(matching) > 1:
        logger.warning("More than one matching pattern found - picking one")
      matching = matching[0]
      # apply matching: set bond properties
      for b in refmol.GetBonds():
        atom1 = matching[b.GetBeginAtomIdx()]
        atom2 = matching[b.GetEndAtomIdx()]
        rw_mol2.AddBond(atom1, atom2, Chem.rdchem.BondType.UNSPECIFIED)
        b2 = rw_mol2.GetBondBetweenAtoms(atom1, atom2)
        b2.SetBondType(b.GetBondType())
        b2.SetIsAromatic(b.GetIsAromatic())
      # apply matching: set atom properties
      for a in refmol.GetAtoms():
        a2 = rw_mol2.GetAtomWithIdx(matching[a.GetIdx()])
        a2.SetHybridization(a.GetHybridization())
        a2.SetIsAromatic(a.GetIsAromatic())
        a2.SetNumExplicitHs(a.GetNumExplicitHs())
        a2.SetFormalCharge(a.GetFormalCharge())
      mol2 = rw_mol2.GetMol()
      SanitizeMol(mol2)
      if hasattr(mol2, '__sssAtoms'):
        mol2.__sssAtoms = None  # we don't want all bonds highlighted
    else:
      raise ValueError("No matching found")
  return mol2

def AssignBondOrdersFromTemplateInverse(refmol, mol):

  refmol2 = rdchem.Mol(refmol)
  mol2 = rdchem.Mol(mol)
  # do the molecules match already?
  # refmol may have more atoms and bonds than mol.
  matching = refmol2.GetSubstructMatch(mol2)
  if not matching:  # no, they don't match
    # check if bonds of mol are SINGLE
    for b in mol2.GetBonds():
      if b.GetBondType() != BondType.SINGLE:
        b.SetBondType(BondType.SINGLE)
        b.SetIsAromatic(False)
    # set the bonds of mol to SINGLE
    for b in refmol2.GetBonds():
      b.SetBondType(BondType.SINGLE)
      b.SetIsAromatic(False)
    # set atom charges to zero;
    for a in refmol2.GetAtoms():
      a.SetFormalCharge(0)
    for a in mol2.GetAtoms():
      a.SetFormalCharge(0)
    
    # matching contains atom indices from refmol.
    # Element i is the refmol atom index matching atom i in mol.
    matching = refmol2.GetSubstructMatches(mol2, uniquify=False)
    # delete all the bonds in mol2 to avoid false bonding from MolFromPDB
    rw_mol2 = Chem.RWMol(mol2)
    bond_indices = []
    for bond in rw_mol2.GetBonds():
        bond_indices.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    for idx1, idx2 in bond_indices:
        rw_mol2.RemoveBond(idx1, idx2)
    # do the molecules match now?
    if matching:
      if len(matching) > 1:
        logger.warning("More than one matching pattern found - picking one")
      matching = matching[0]
      # apply matching: set bond properties
      for b in refmol.GetBonds():
        if b.GetBeginAtomIdx() in matching and b.GetEndAtomIdx() in matching:
          atom1 = matching.index(b.GetBeginAtomIdx())
          atom2 = matching.index(b.GetEndAtomIdx())
          rw_mol2.AddBond(atom1, atom2, Chem.rdchem.BondType.UNSPECIFIED)
          b2 = rw_mol2.GetBondBetweenAtoms(atom1, atom2)
          b2.SetBondType(b.GetBondType())
          b2.SetIsAromatic(b.GetIsAromatic())
      # apply matching: set atom properties
      for a in refmol.GetAtoms():
        if a.GetIdx() in matching:
          a2 = rw_mol2.GetAtomWithIdx(matching.index(a.GetIdx()))
          a2.SetHybridization(a.GetHybridization())
          a2.SetIsAromatic(a.GetIsAromatic())
          a2.SetNumExplicitHs(a.GetNumExplicitHs())
          a2.SetFormalCharge(a.GetFormalCharge())
      mol2 = rw_mol2.GetMol()
      SanitizeMol(mol2)
      if hasattr(mol2, '__sssAtoms'):
        mol2.__sssAtoms = None  # we don't want all bonds highlighted
    else:
      raise ValueError("No matching found")
  return mol2
