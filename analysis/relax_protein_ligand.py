# Copyright (c) MDLDrugLib. All rights reserved.
import os
import shutil
import typing as t
import argparse
import warnings
import logging
from pathlib import Path
import multiprocessing as mp
import numpy as np
import mdtraj
import openmm as mm
import openmm.app as mm_app
import openmm.unit as mm_unit
from openmm import CustomExternalForce
import pdbfixer
from openmm.app import Modeller
import openff
from openff.toolkit import Molecule
from openff.units import Quantity as openff_Quantity
from openff.units.openmm import from_openmm, to_openmm
from openff.toolkit.utils.exceptions import OpenFFToolkitException
from openmmforcefields.generators import SystemGenerator

from rdkit import Chem
from rdkit.Chem import AllChem
from utils import split_complex, extract_residues_from_filename, _is_in_the_range
from score_utils import fix_obabel_input_pdb, fix_rdkit_input_pdb, AssignBondOrdersFromTemplateNew, AssignBondOrdersFromTemplateInverse
from openbabel import pybel
from openbabel import openbabel as ob
from vina_score import *

from Bio.PDB import (
    PDBParser, MMCIFParser,
    PDBIO, Select, MMCIFIO,
)
logger = logging.getLogger('PLRelaxer')
warnings.filterwarnings("ignore")

ob.obErrorLog.SetOutputLevel(ob.obError) 


def fix_pdb(
        input_pdb_file: str,
        out_fixed_pdb_file: t.Optional[str] = None,
        keepIds: bool = False,
        seed: t.Optional[int] = None,
):
    """Preprocessing of protein."""
    if not Path(input_pdb_file).exists():
        raise FileNotFoundError(input_pdb_file)
    fixer = pdbfixer.PDBFixer(filename=input_pdb_file)
    fixer.removeHeterogens()
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=seed)
    fixer.addMissingHydrogens(7.4)
    if out_fixed_pdb_file is not None:
        Path(out_fixed_pdb_file).parent.mkdir(parents=True, exist_ok=True)
        mm_app.PDBFile.writeFile(fixer.topology, fixer.positions, open(str(out_fixed_pdb_file), 'w'), keepIds=keepIds)
    return fixer.topology, fixer.positions

def load_protein(
        protein_file: str,
):
    if not Path(protein_file).exists():
        raise FileNotFoundError(protein_file)
    protein_file = str(protein_file)
    if protein_file.endswith('.pdb'):
        protein = mm_app.PDBFile(protein_file)
    elif protein_file.endswith('.cif'):
        protein = mm_app.PDBxFile(protein_file)
    else:
        suffix = Path(protein_file).suffix
        raise NotImplementedError(f".pdb or .cif are supported, but {suffix} from {protein_file}")

    return protein

def remove_hydrogen_pdb(pdbFile, toFile):

    parser = MMCIFParser(QUIET=True) if pdbFile[-4:] == ".cif" else PDBParser(QUIET=True)
    s = parser.get_structure("x", pdbFile)
    class NoHydrogen(Select):
        def accept_atom(self, atom):
            if atom.element == 'H' or atom.element == 'D':
                return False
            return True

    io = MMCIFIO() if toFile[-4:] == ".cif" else PDBIO()
    io.set_structure(s)
    io.save(toFile, select=NoHydrogen())

def parse_molfile(
    path: Path,
    sanitize=True,
    removeHs=True,
    strictParsing=True,
    proximityBonding=True,
    cleanupSubstructures=True,
    **kwargs,
) -> Chem.rdchem.Mol:
    """Load one molecule from a file, picking the right RDKit function."""
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".sdf":
        mol = Chem.MolFromMolFile(
            str(path),
            sanitize=sanitize,
            removeHs=removeHs,
            strictParsing=strictParsing,
        )

    elif path.suffix == ".mol2":
        mol = Chem.MolFromMol2File(
            str(path),
            sanitize=False,
            removeHs=removeHs,
            cleanupSubstructures=cleanupSubstructures,
        )

    elif path.suffix == ".pdb":
        mol = Chem.MolFromPDBFile(
            str(path),
            sanitize=sanitize,
            removeHs=removeHs,
            proximityBonding=proximityBonding,
        )
        if mol is None and proximityBonding:
            mol = Chem.MolFromPDBFile(
                str(path),
                sanitize=sanitize,
                removeHs=removeHs,
                proximityBonding=False,
            )

    elif path.suffix == ".mol":
        block = "".join(open(path).readlines()).strip() + "\nM  END"
        mol = Chem.MolFromMolBlock(
            block,
            sanitize=sanitize,
            removeHs=removeHs,
            strictParsing=strictParsing,
        )
    else:
        raise ValueError(f"Unknown file type {path.suffix}")

    if mol is not None:
        mol.SetProp("_Path", str(path))

    return mol

def parse_mol_pybel_to_rdkit(ligand_file: Path, **kwargs):
    if not ligand_file.exists():
        raise FileNotFoundError(ligand_file)

    pybel_mol_h = next(pybel.readfile(str(ligand_file.suffix[1:]), str(ligand_file)))
    pybel_mol_h.OBMol.AddHydrogens(False, True, 7.4)
    if pybel_mol_h is None:
            return None
    sdf_string = pybel_mol_h.write("sdf")
    mol_h = Chem.MolFromMolBlock(
        sdf_string,
        sanitize=kwargs.get('sanitize', True),
        removeHs=False,
        strictParsing=kwargs.get('strictParsing', True),
    )
    if mol_h is not None:
        mol_h.SetProp("_Path", str(ligand_file))
    return mol_h

def parse_mol_rdkit(ligand_file:Path, ref_ligand:Path, **kwargs):
    mol = parse_molfile(
            path = ligand_file,
            sanitize = kwargs.get('sanitize', True),
            removeHs = kwargs.get('removeHs', True),
            strictParsing = kwargs.get('strictParsing', True),
            proximityBonding = kwargs.get('proximityBonding', True),
            cleanupSubstructures = kwargs.get('cleanupSubstructures', True),
        )

    if mol is None:
        return None

    if mol.GetNumConformers() < 0.5:
        raise ValueError(f'mol from {ligand_file} has no conformer.')
    ref_mol = parse_molfile(ref_ligand)
    try:
        new_mol = AssignBondOrdersFromTemplateNew(ref_mol, mol)
    except Exception as e:
        print(f"Error in AssignBondOrdersFromTemplateNew for {ligand_file}: {e}, trying AssignBondOrdersFromTemplateInverse...")
        try:
            new_mol = AssignBondOrdersFromTemplateInverse(ref_mol, mol)
        except Exception as e:
            print(f"Error in AssignBondOrdersFromTemplateInverse for {ligand_file}: {e}, skipping this ligand.")
            return None
    mol_h = Chem.AddHs(new_mol, addCoords=True)
    
    return mol_h

def load_mol(
        ligand_file: str,
        ref_ligand: str,
        **kwargs,
):
    ligand_file = Path(ligand_file)
    parse_with_openbabel = kwargs.get('parse_with_openbabel', False)
    if parse_with_openbabel:
        # Load the molecule with Open Babel and add hydrogens.
        logger.info(f'Parsing ligand {ligand_file} with openbabel')
        mol_h = parse_mol_pybel_to_rdkit(ligand_file,
                                        sanitize=kwargs.get('sanitize', True),
                                        removeHs=kwargs.get('removeHs', True),
                                        strictParsing=kwargs.get('strictParsing', True),
                                        proximityBonding=kwargs.get('proximityBonding', True),
                                        cleanupSubstructures=kwargs.get('cleanupSubstructures', True))
    else:
        # Load the molecule with RDKit.
        assert ref_ligand is not None, "ref_ligand is required when parse ligand by rdkit"
        ref_ligand = Path(ref_ligand)
        logger.info(f'Parsing ligand {ligand_file} with rdkit, and assign bond order from ref ligand {ref_ligand}')
        try:
            mol_h = parse_mol_rdkit(ligand_file, 
                                    ref_ligand,
                                    sanitize=kwargs.get('sanitize', True),
                                    removeHs=kwargs.get('removeHs', True),
                                    strictParsing=kwargs.get('strictParsing', True),
                                    proximityBonding=kwargs.get('proximityBonding', True),
                                    cleanupSubstructures=kwargs.get('cleanupSubstructures', True))
        except Exception as e:
            logger.warning(f'Error {e} in parsing molecule by rdkit, Retring parsing ligand {ligand_file} with openbabel')
            fix_obabel_input_pdb(ligand_file, ligand_file)
            mol_h = parse_mol_pybel_to_rdkit(ligand_file,
                                        sanitize=kwargs.get('sanitize', True),
                                        removeHs=kwargs.get('removeHs', True),
                                        strictParsing=kwargs.get('strictParsing', True),
                                        proximityBonding=kwargs.get('proximityBonding', True),
                                        cleanupSubstructures=kwargs.get('cleanupSubstructures', True))
    
    if mol_h is None:
        return None, ""
    # Reorder atoms using RDKit's _smilesAtomOutputOrder so SMILES atom order matches coordinates.
    smiles = Chem.MolToSmiles(mol_h)
    m_order = list(
        mol_h.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"]
    )
    mol_h = Chem.RenumberAtoms(mol_h, m_order)

    # Build an OpenFF Molecule from SMILES and add the RDKit conformer coordinates in angstroms.
    molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    molecule.add_conformer(mm_unit.Quantity(mol_h.GetConformer().GetPositions(), mm_unit.angstrom))

    try:
        # Compute Gasteiger partial charges; warn and fall back to all-zero charges on failure.
        molecule.assign_partial_charges(partial_charge_method='gasteiger')
    except Exception as e:
        logger.warning(f'partial charge failed for mol from {ligand_file}. Set Zeros.\nERROR: {str(e)}')
        molecule.assign_partial_charges(partial_charge_method='zeros')

    return molecule, smiles

def remove_hydrogen_reorder(mol):
    mol = Chem.RemoveAllHs(mol)
    smiles = Chem.MolToSmiles(mol)
    m_order = list(
        mol.GetPropsAsDict(
            includePrivate=True,
            includeComputed=True,
        )["_smilesAtomOutputOrder"]
    )
    mol = Chem.RenumberAtoms(mol, m_order)
    return mol

def define_restrain(
        atom: mm_app.Atom,
        rst_type: str = "CA",
):
    if rst_type == "non_H":
        return atom.element.name != "hydrogen"
    elif rst_type == "CA":
        return atom.name == "CA"
    # An x marker identifies a ligand atom.
    elif rst_type == 'protein':
        return 'x' not in atom.name
    elif rst_type == "CA+ligand":
        return ('x' in atom.name) or (atom.name == "CA")
    elif rst_type == "ligand":
        return 'x' in atom.name
    else:
        raise NotImplementedError(rst_type)

def set_pr_system(topology):
    """
    Set the system using the topology from the pdb file
        for protein_only relaxation.
    """
    #Put it in a force field to skip adding all particles manually
    forcefield = mm_app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    system = forcefield.createSystem(topology,
                                     removeCMMotion=False,
                                     nonbondedMethod=mm_app.NoCutoff,
                                     rigidWater=True #Use implicit solvent
                                     )
    return system

# https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomExternalForce.html
def add_p_restraints(
        system,
        topology,
        positions,
        n_res: int,
        restraint_type: str,
        flexible_range:list[tuple[tuple[str,int]]] = [],
        stiffness: float = 500.0,
):  
    if flexible_range == []:
        return system
    
    restraint = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
    restraint.addGlobalParameter('k', stiffness * mm_unit.kilojoules_per_mole/mm_unit.nanometer**2)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    for atid, atom in enumerate(topology.atoms()):
        ch_rs_ic = (atom.residue.chain.id, int(atom.residue.id), atom.residue.insertionCode)

        # Restrain non-hydrogen protein atoms outside the flexible range.
        if not _is_in_the_range(ch_rs_ic, flexible_range) and define_restrain(atom, "non_H"):
            # addParticle(particle index, [parameter values]).
            restraint.addParticle(atom.index, positions[atom.index])

    system.addForce(restraint)
    return system

def add_l_restraints(
        system,
        topology,
        positions,
        n_res: int,
        restraint_type: str,
        stiffness: float = 1000.0,
):
    restraint = CustomExternalForce('k_ligand*periodicdistance(x, y, z, x0, y0, z0)^2')
    restraint.addGlobalParameter('k_ligand', stiffness * mm_unit.kilojoules_per_mole/mm_unit.nanometer**2)
    restraint.addPerParticleParameter('x0')
    restraint.addPerParticleParameter('y0')
    restraint.addPerParticleParameter('z0')

    for atid, atom in enumerate(topology.atoms()):
        if atom.residue.index < n_res:
            continue

        if define_restrain(atom, restraint_type):
            restraint.addParticle(atom.index, positions[atom.index])

    system.addForce(restraint)
    return system

def minimize_energy(
        topology,
        system,
        positions,
        reportInterval = 0,
        gpu: bool = True,
        tolerance = 0.01, # as posebusters do 0.01 kj/mol (redock? cross dock with sc packing should be higher)
        maxIterations = 0, # minimization is continued until the results converge without regard
):
    """Function that minimizes energy, given topology, OpenMM system, and positions"""
    # integrator = mm.LangevinIntegrator(0, 0.01, 0.0)
    integrator = mm.LangevinIntegrator(300 * mm_unit.kelvin, 1 / mm_unit.picosecond, 0.002 * mm_unit.picoseconds)
    platform = mm.Platform.getPlatformByName("CUDA" if gpu else "CPU")
    simulation = mm.app.Simulation(topology, system, integrator, platform)

    if reportInterval > 0:
        # Initialize the DCDReporter
        reporter = mdtraj.reporters.DCDReporter('traj.dcd', reportInterval)
        # Add the reporter to the simulation
        simulation.reporters.append(reporter)

    simulation.context.setPositions(positions)

    ENERGY = mm_unit.kilocalories_per_mole
    LENGTH = mm_unit.angstroms
    ret = {}
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["einit"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["posinit"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

    try:
        simulation.minimizeEnergy(
            tolerance * mm_unit.kilojoule_per_mole / mm_unit.nanometer,
            maxIterations)
    except Exception as e:
        logger.info(f'Error When energy minimization: {str(e)}')
        # openmm.OpenMMException: Particle coordinate is nan in EDM-Dock
        ret["efinal"] = state.getPotentialEnergy().value_in_unit(ENERGY)
        ret["pos"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

        if reportInterval > 0:
            reporter.close()
        return ret
    # Save positions
    minstate = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["efinal"] = minstate.getPotentialEnergy().value_in_unit(ENERGY)
    ret["pos"] = minstate.getPositions(asNumpy=True).value_in_unit(LENGTH)

    if reportInterval > 0:
        reporter.close()

    logger.info(f'Energy change: {ret["einit"]} -> {ret["efinal"]}')

    return ret

def export_openff_mol(
        openff_mol,
        to_file,
):
    new_mol = openff_mol.to_rdkit()
    #new_mol = remove_hydrogen_reorder(new_mol)
    Path(to_file).parent.mkdir(parents=True, exist_ok=True)
    w = Chem.SDWriter(str(to_file))
    w.write(new_mol)
    w.close()
    return

def export_obj(
        min_ret,
        modeller,
        protein_topology,
        smiles: t.Optional[str] = None,
        out_relax_pdb_file: t.Optional[str] = None,
        out_relax_lig_file: t.Optional[str] = None,
        out_relax_complex_file: t.Optional[str] = None,
        **kwargs,
):
    if out_relax_pdb_file is not None:
        if out_relax_pdb_file.endswith('.pdb'):
            export_fn = mm_app.PDBFile.writeFile
        elif out_relax_pdb_file[-3:] == 'cif':
            export_fn = mm_app.PDBxFile.writeFile
        else:
            suffix = Path(out_relax_pdb_file).suffix
            raise NotImplementedError(f"{suffix} from {out_relax_pdb_file} not supported (.pdb or .cif)")
        Path(out_relax_pdb_file).parent.mkdir(parents=True, exist_ok=True)
        export_fn(
            protein_topology,
            min_ret["pos"][:protein_topology.getNumAtoms()],
            open(out_relax_pdb_file, 'w'),
            keepIds=kwargs.get('keepIds', True),
        )
        remove_hydrogen_pdb(out_relax_pdb_file, out_relax_pdb_file)
        if out_relax_complex_file is not None:
            Path(out_relax_complex_file).parent.mkdir(parents=True, exist_ok=True)
            export_fn(
                modeller.topology,
                min_ret["pos"],
                open(out_relax_complex_file, 'w'),
                keepIds=kwargs.get('keepIds', True),
            )

    if out_relax_lig_file is not None:
        new_molecule = Molecule.from_smiles(
            smiles, allow_undefined_stereo=True,
        )
        new_molecule.add_conformer(
            mm_unit.Quantity(
                min_ret["pos"][protein_topology.getNumAtoms():],
                mm_unit.angstrom)
        )
        export_openff_mol(new_molecule, out_relax_lig_file)

    return

def relax_pl(
        input_pdb_file: str,
        ligand_file: t.Optional[str] = None,
        ref_ligand_file = None,
        out_fixed_pdb_file: t.Optional[str] = None,
        out_relax_pdb_file: t.Optional[str] = None,
        out_relax_lig_file: t.Optional[str] = None,
        out_relax_complex_file: t.Optional[str] = None,
        ligand_chain_id = None,
        kwargs = {},
):
    ## Skip already done system
    xs = [out_relax_pdb_file, out_relax_lig_file, out_relax_complex_file]
    if all(x is None for x in xs):
        return

    split_input_complex = kwargs.get('split_input_complex', False)
    if split_input_complex and ligand_file is None:
        out_dir = os.path.dirname(out_relax_pdb_file) if out_relax_pdb_file is not None else os.path.dirname(input_pdb_file)
        if ligand_chain_id is None:
            ligand_chain_id = os.path.basename(input_pdb_file).split('_')[1]
        ligand_file, input_pdb_file = split_complex(input_pdb_file, ligand_chain_id, out_dir)
        parse_with_openbabel = kwargs.get('parse_with_openbabel', False)
        if parse_with_openbabel:
            fix_obabel_input_pdb(ligand_file, ligand_file)
        else:
            fix_rdkit_input_pdb(ligand_file, ligand_file)

    ## Read protein PDB and add hydrogens
    logger.info('Preprocessing protein using pdbfixer...')
    protein_topology, protein_positions = fix_pdb(
        input_pdb_file = input_pdb_file,
        out_fixed_pdb_file = out_fixed_pdb_file,
        keepIds = kwargs.get('keepIds', True),
        seed = kwargs.get('seed', None),
    )
    n_res = protein_topology.getNumResidues()

    logger.info('Preparing protein-ligand complex...')
    modeller = Modeller(protein_topology, protein_positions)
    logger.info(f'Protein-Only System has {modeller.topology.getNumAtoms()} atoms of {n_res} residues')

    ligand_mol, smiles = None, None

    if ligand_file is not None:
        ligand_mol, smiles = load_mol(
            ligand_file = ligand_file,
            ref_ligand = ref_ligand_file,
            sanitize=kwargs.get('sanitize', True),
            removeHs=kwargs.get('removeHs', True),
            strictParsing=kwargs.get('strictParsing', True),
            proximityBonding=kwargs.get('proximityBonding', True),
            cleanupSubstructures=kwargs.get('cleanupSubstructures', True),
            parse_with_openbabel=kwargs.get('parse_with_openbabel', True),
        )
        if ligand_mol is None:
            logger.error(f'Error in parsing ligand file: {ligand_file}')
            if out_relax_lig_file is not None:
                shutil.copy(ligand_file, out_relax_lig_file)
            if out_relax_pdb_file:
                if out_relax_pdb_file.endswith('.pdb'):
                    export_fn = mm_app.PDBFile.writeFile
                elif out_relax_pdb_file[-3:] == 'cif':
                    export_fn = mm_app.PDBxFile.writeFile
                else:
                    suffix = Path(out_relax_pdb_file).suffix
                    raise NotImplementedError(f"{suffix} from {out_relax_pdb_file} not supported (.pdb or .cif)")
                Path(out_relax_pdb_file).parent.mkdir(parents=True, exist_ok=True)
                export_fn(
                    protein_topology,
                    protein_positions,
                    open(out_relax_pdb_file, 'w'),
                    keepIds=kwargs.get('keepIds', True),
                )
            # no complex file output to indicate this is placeholder and no minimization performed.
            return

        lig_top = ligand_mol.to_topology()
        modeller.add(lig_top.to_openmm(), to_openmm(ligand_mol.conformers[0]))
        logger.info(f'Complex System has {modeller.topology.getNumAtoms()} atoms')

    logger.info('Prepare system...')
    forcefield_kwargs = {'constraints': mm_app.HBonds, }

    try:
        # Load the ff19SB and OpenFF "Sage" force field.
        system_generator = SystemGenerator(
            forcefields=['amber/protein.ff19SB.xml'],
            small_molecule_forcefield='openff-2.2.0',
            molecules=[ligand_mol] if ligand_mol is not None else [],
            forcefield_kwargs=forcefield_kwargs,
        )
        if ligand_mol is not None:
            system = system_generator.create_system(modeller.topology, molecules=ligand_mol)
        else:
            system = system_generator.create_system(modeller.topology)
    except:
        try:
            # if sage ff fail to parameterize the ligand, use gaff to rescue
            system_generator = SystemGenerator(
                forcefields=['amber/protein.ff19SB.xml'],
                small_molecule_forcefield='gaff-2.11',
                molecules=[ligand_mol] if ligand_mol is not None else [],
                forcefield_kwargs=forcefield_kwargs,
            )
            if ligand_mol is not None:
                system = system_generator.create_system(modeller.topology, molecules=ligand_mol)
            else:
                system = system_generator.create_system(modeller.topology)
        except Exception as e:
            logger.error(f'Error when build system: {str(e)}')
            # raise No template found for residues
            # exceptions.UnassignedProperTorsionParameterException: ProperTorsionHandler was not able to find parameters for the following valence terms
            # make output placeholder
            if out_relax_lig_file is not None:
                export_openff_mol(ligand_mol, out_relax_lig_file)
            if out_relax_pdb_file:
                if out_relax_pdb_file.endswith('.pdb'):
                    export_fn = mm_app.PDBFile.writeFile
                elif out_relax_pdb_file[-3:] == 'cif':
                    export_fn = mm_app.PDBxFile.writeFile
                else:
                    suffix = Path(out_relax_pdb_file).suffix
                    raise NotImplementedError(f"{suffix} from {out_relax_pdb_file} not supported (.pdb or .cif)")
                Path(out_relax_pdb_file).parent.mkdir(parents=True, exist_ok=True)
                export_fn(
                    protein_topology,
                    protein_positions,
                    open(out_relax_pdb_file, 'w'),
                    keepIds=kwargs.get('keepIds', True),
                )
            # no complex file output to indicate this is placeholder and no minimization performed.
            return

    # give some residues (such as pocket to relex)
    # atom mask

    logger.info('Add restraint...')
    p_stiffness = float(kwargs.get('p_stiffness', 500.0))
    if p_stiffness > 0.:
        extract_res_mode = kwargs.get('extract_res_mode', 'imgt')
        flexible_range = extract_residues_from_filename(input_pdb_file, extract_res_mode)
        system = add_p_restraints(
            system,
            modeller.topology,
            modeller.positions,
            n_res=n_res,
            restraint_type=kwargs.get("p_restraint_type", 'protein'),
            flexible_range = flexible_range ,
            stiffness=p_stiffness
        )
    # Larger stiffness imposes a stronger restraint and limits displacement from the initial position.
    # Smaller stiffness imposes a weaker restraint and allows greater movement.
    l_stiffness = float(kwargs.get('l_stiffness', 1000.0))
    if ligand_mol is not None and l_stiffness > 0.:
        system = add_l_restraints(
            system,
            modeller.topology,
            modeller.positions,
            n_res=n_res,
            restraint_type=kwargs.get("l_restraint_type", 'non_H'),
            stiffness=l_stiffness,
        )

    ## Minimize energy
    logger.debug('Running Minimization...')
    ret = minimize_energy(
        modeller.topology,
        system,
        modeller.positions,
        gpu=kwargs.get('gpu', True),
        reportInterval=kwargs.get('ccd_int', 0),
        tolerance=kwargs.get('tolerance', 1),
        maxIterations=kwargs.get('maxIterations', 0),
    )

    export_obj(
        ret,
        modeller,
        protein_topology,
        smiles,
        out_relax_pdb_file,
        out_relax_lig_file if ligand_mol is not None else None,
        out_relax_complex_file if ligand_mol is not None else None,
        **kwargs,
    )

    if ligand_mol is not None:
        logger.debug(f'Complex: {input_pdb_file} with ligand {ligand_file} minimization is done!')
    else:
        logger.debug(f'Protein: {input_pdb_file} minimization is done!')

    return xs

def pipeline(input_pdb_file, ligand_file, out_dir, arg_dict):
    basename = os.path.basename(input_pdb_file)
    basename = os.path.splitext(basename)[0]
    out_fixed_pdb_file = os.path.join(out_dir, f'fixed_{basename}.pdb')
    out_relax_pdb_file = os.path.join(out_dir, f'relaxed_{basename}.pdb')
    out_relax_lig_file = os.path.join(out_dir, f'relaxed_{basename}.sdf')
    out_relax_complex_file = os.path.join(out_dir, f'relaxed_{basename}_complex.pdb')

    xs = relax_pl(
        input_pdb_file,
        ligand_file,
        out_fixed_pdb_file,
        out_relax_pdb_file,
        out_relax_lig_file,
        out_relax_complex_file,
        arg_dict,
    )

    lig_pdbqt = os.path.join(out_dir, f'lig_{basename}.pdbqt')
    prot_pqr = os.path.join(out_dir, f'prot_{basename}.pqr')
    prot_pdbqt = os.path.join(out_dir, f'prot_{basename}.pdbqt')


    lig_prep = PrepLig(out_relax_lig_file, "sdf")
    prot_prep = PrepProt(out_relax_pdb_file)

    lig_prep.get_pdbqt(lig_pdbqt)

    prot_prep.addH(prot_pqr)
    prot_prep.get_pdbqt(prot_pdbqt)

    dock = VinaDock(lig_pdbqt, prot_pdbqt)
    dock.get_box()
    score = dock.dock(mode="minimize")

    return xs, score

def minimizer(
        arg_dict
):
    
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protein-ligand complex system MM-based relaxation.')
    parser.add_argument(
        'work_dir', type = str,
        help = 'Binding Complex directory with the tree [work_dir]/[ComplexID]/[EnsembleID].'
               'Every ensemble directory has lig_final.sdf (or lig_final_ec.sdf) and prot_final.pdb',
    )

    parser.add_argument(
        '--extract_mode',
        type=str,
        default='imgt',
        help='Mode for extracting residues from filename. Defaults to "imgt".'
    )

    parser.add_argument(
        '-nb',
        '--num_workers',
        type=int,
        default=mp.cpu_count() // 2,
        help='The number of workers for multi-processing. '
             'Defaults to the half number of available cpus.'
    )
    parser.add_argument(
        '-cpu',
        '--use_cpu',
        action='store_true',
        default = False,
        help='Run OpenMM on CPU device rather than GPU acceleration.'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Whether show the progress bar.'
    )
    args = parser.parse_args()
    logger.setLevel(logging.DEBUG)
    arg_dict = dict(
        sanitize=True,
        removeHs=False,
        split_input_complex=True,
        extract_res_mode=args.extract_mode,
        parse_with_openbabel=True,
        strictParsing=True,
        proximityBonding=True,
        cleanupSubstructures=True,
        p_restraint_type='non_H',
        p_stiffness=100.,
        l_restraint_type='non_H',
        l_stiffness=0.,
        tolerance=0.01,
        maxIterations=0,
        gpu=(not args.use_cpu),
        ccd_int=0,
        keepIds=True,
        seed=None,
        num_workers=args.num_workers,
        verbose=args.verbose,
    )
    minimizer(
        arg_dict
    )