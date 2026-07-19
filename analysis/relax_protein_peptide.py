import os
import time
import io
import logging
import pdbfixer
import openmm
from openmm import app as openmm_app
import openmm.unit as mm_unit
from utils import _is_in_the_range, extract_residues_from_filename
from score_utils import check_overlapping_pairs, delete_overlap_atoms
import pyrosetta
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta import create_score_function
from pyrosetta import init, pose_from_pdb


init('-use_input_sc -input_ab_scheme AHo_Scheme -ignore_unrecognized_res \
    -ignore_zero_occupancy false -load_PDB_components true -relax:default_repeats 2 -no_fconfig')

logger = logging.getLogger('PLRelaxer')

def current_milli_time():
    return round(time.time() * 1000)


class ForceFieldMinimizer(object):

    def __init__(self, 
                 stiffness=10.0, 
                 max_iterations=0, 
                 tolerance=0.01*mm_unit.kilocalories_per_mole / mm_unit.angstroms, 
                 platform='CUDA',
                 flexible_range=[]):
        super().__init__()
        self.stiffness = stiffness
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        assert platform in ('CUDA', 'CPU')
        self.platform = platform
        self.flexible_range = flexible_range

    def _fix(self, pdb_str):
        fixer = pdbfixer.PDBFixer(pdbfile=io.StringIO(pdb_str))
        overlap_pairs = check_overlapping_pairs(fixer) 
        if overlap_pairs:
            for a1, a2 in overlap_pairs:
                r1 = a1.residue
                r2 = a2.residue
                logger.info(
                    f"Overlapping bonded atoms: {a1.name} in {r1.name} {r1.id} chain {r1.chain.id} "
                    f"<-> {a2.name} in {r2.name} {r2.id} chain {r2.chain.id}"
                )
            delete_overlap_atoms(fixer, overlap_pairs)
        
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms(seed=0)

        fixer.addMissingHydrogens()
        out_handle = io.StringIO()
        openmm_app.PDBFile.writeFile(fixer.topology, fixer.positions, out_handle, keepIds=True)
        return out_handle.getvalue()

    def _get_pdb_string(self, topology, positions):
        with io.StringIO() as f:
            openmm_app.PDBFile.writeFile(topology, positions, f, keepIds=True)
            return f.getvalue()

    def _minimize(self, pdb_str):
        pdb = openmm_app.PDBFile(io.StringIO(pdb_str))

        force_field = openmm_app.ForceField("amber/protein.ff19SB.xml")
        constraints = openmm_app.HBonds
        system = force_field.createSystem(pdb.topology, constraints=constraints)

        # Add constraints to non-generated regions
        restraint = openmm.CustomExternalForce("0.5 * k * periodicdistance(x, y, z, x0, y0, z0)^2")
        restraint.addGlobalParameter("k", self.stiffness)
        restraint.addPerParticleParameter('x0')
        restraint.addPerParticleParameter('y0')
        restraint.addPerParticleParameter('z0')
            
        if len(self.flexible_range) > 0:
            for i, a in enumerate(pdb.topology.atoms()):
                ch_rs_ic = (a.residue.chain.id, int(a.residue.id), a.residue.insertionCode)
                if not _is_in_the_range(ch_rs_ic, self.flexible_range) and a.element.name != "hydrogen":
                    restraint.addParticle(i, pdb.positions[i])
                
        system.addForce(restraint)

        # Set up the integrator and simulation
        # Temperature, friction coefficient, and step size.
        # integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
        integrator = openmm.LangevinIntegrator(300 * mm_unit.kelvin, 1 / mm_unit.picosecond, 0.002 * mm_unit.picoseconds)
        platform = openmm.Platform.getPlatformByName("CUDA")
        simulation = openmm_app.Simulation(pdb.topology, system, integrator, platform)
        simulation.context.setPositions(pdb.positions)

        # Perform minimization
        ret = {}
        ENERGY = mm_unit.kilocalories_per_mole
        LENGTH = mm_unit.angstroms
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        ret["einit"] = state.getPotentialEnergy().value_in_unit(ENERGY)
        ret["posinit"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

        # tolerance: convergence threshold.
        # maxIterations: maximum iterations; 0 means no limit and runs until convergence.
        simulation.minimizeEnergy(maxIterations=self.max_iterations, tolerance=self.tolerance)

        state = simulation.context.getState(getEnergy=True, getPositions=True)
        ret["efinal"] = state.getPotentialEnergy().value_in_unit(ENERGY)
        ret["pos"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
        ret["min_pdb"] = self._get_pdb_string(simulation.topology, state.getPositions())

        return ret['min_pdb'], ret

    def _add_energy_remarks(self, pdb_str, ret):
        pdb_lines = pdb_str.splitlines()
        pdb_lines.insert(1, "REMARK   1  FINAL ENERGY:   {:.3f} KCAL/MOL".format(ret['efinal']))
        pdb_lines.insert(1, "REMARK   1  INITIAL ENERGY: {:.3f} KCAL/MOL".format(ret['einit']))
        logger.info(f"Energy before minimization: {ret['einit']:.3f} KCAL/MOL, after minimization: {ret['efinal']:.3f} KCAL/MOL")

        return "\n".join(pdb_lines)

    def __call__(self, pdb_str, flexible_residue_first=None, flexible_residue_last=None, return_info=True):
        if '\n' not in pdb_str and pdb_str.lower().endswith(".pdb"):
            with open(pdb_str) as f:
                pdb_str = f.read()

        pdb_fixed = self._fix(pdb_str)
        pdb_min, ret = self._minimize(pdb_fixed)
        pdb_min = self._add_energy_remarks(pdb_min, ret)
        if return_info:
            return pdb_min, ret
        else:
            return pdb_min

def pyrosetta_interface_energy(pdb_path, interface):
    pose = pyrosetta.pose_from_pdb(pdb_path)
    mover = InterfaceAnalyzerMover()
    mover.set_interface(interface)
    mover.set_scorefunction(create_score_function('ref2015'))
    mover.apply(pose)
    return pose.scores['dG_separated']

def run_openmm(input_pdb_file, 
               out_relax_pdb_file,
               item_name = None,
               extract_res_mode="imgt",
               stiffness=0.0,
               max_iterations=0,
               tolerance=0.01*mm_unit.kilocalories_per_mole / mm_unit.angstroms,
               platform='CUDA',
               cal_dg = False):
    if stiffness > 0:
        flexible_range = extract_residues_from_filename(input_pdb_file, extract_res_mode)
    else:
        flexible_range = []
    if not os.path.exists(out_relax_pdb_file):
        #try:
            logger.info(f"Running OpenMM relaxation on {input_pdb_file} with stiffness={stiffness}, max_iterations={max_iterations}, tolerance={tolerance}, platform={platform}, flexible_range={flexible_range}")
            minimizer = ForceFieldMinimizer(stiffness=stiffness, 
                                            max_iterations=max_iterations, 
                                            tolerance=tolerance, 
                                            platform=platform, 
                                            flexible_range=flexible_range)
            with open(input_pdb_file, 'r') as f:
                pdb_str = f.read()

            pdb_min = minimizer(
                pdb_str = pdb_str,
                return_info = False,
            )
            with open(out_relax_pdb_file, 'w') as f:
                f.write(pdb_min)

        #except ValueError as e:
        #    import traceback
        #    logger.error(f"[SKIPPED] {input_pdb_file} failed with error: {str(e)}")
        #    logger.debug(traceback.format_exc())
        #    return
    
    if cal_dg:
        if item_name is not None:
            ligand_chain = item_name.split("_")[1]
            heavy_chain = item_name.split("_")[2]
            light_chain = item_name.split("_")[3]
        else:
            basename = os.path.basename(input_pdb_file)
            ligand_chain = basename.split("_")[1]
            heavy_chain = basename.split("_")[2]
            light_chain = basename.split("_")[3]

        interface = ligand_chain + "_" + heavy_chain + light_chain
        dg = pyrosetta_interface_energy(out_relax_pdb_file, interface)
        logger.info(f"Interface energy (dG) after relaxation: {dg:.3f} REU")
        return dg
    
    return None

    


