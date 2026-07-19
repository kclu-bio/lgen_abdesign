from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.kinematics import FoldTree
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task import operation
from pyrosetta.rosetta.core.simple_metrics import metrics
from pyrosetta.rosetta.core.select import residue_selector as selections
from pyrosetta.rosetta.core import select
from pyrosetta.rosetta.core.select.movemap import *
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector, AndResidueSelector, OrResidueSelector
from pyrosetta import create_score_function

#Protocol Includes
from pyrosetta.rosetta.protocols import minimization_packing as pack_min
from pyrosetta.rosetta.protocols import relax as rel
from pyrosetta.rosetta.protocols.antibody.residue_selector import CDRResidueSelector
from pyrosetta.rosetta.protocols.antibody import *
from pyrosetta.rosetta.protocols.loops import *
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.select.movemap import MoveMapFactory, move_map_action
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover
import pyrosetta
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

from pyrosetta import init, pose_from_pdb

import re
import os
import logging
import argparse
import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm
from utils import extract_residues_from_filename

init('-use_input_sc -input_ab_scheme AHo_Scheme -ignore_unrecognized_res \
    -ignore_zero_occupancy false -load_PDB_components true -relax:default_repeats 2 -no_fconfig')

    
def pyrosetta_interface_energy(pdb_path, interface):
    pose = pyrosetta.pose_from_pdb(pdb_path)
    mover = InterfaceAnalyzerMover()
    mover.set_interface(interface)
    mover.set_scorefunction(create_score_function('ref2015'))
    mover.apply(pose)
    return pose.scores['dG_separated']

def relax(pdb_file, pocket_residues, output_dir):
    logging.info(f'Rosetta processing {pdb_file} for Relax')
    pose = pose_from_pdb(pdb_file)
    scorefxn = create_score_function('ref2015')
    output = os.path.splitext(os.path.basename(pdb_file))[0]
    output_file = os.path.join(output_dir, f'{output}_relaxed_rosetta.pdb')
    
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    # Prevent packing from changing amino-acid identities.
    tf.push_back(operation.RestrictToRepacking()) 
    # Prevent repacking outside the selected region.
    tf.push_back(operation.PreventRepacking()) 
   
    gen_selector = selections.ResidueIndexSelector('1')
    # print(f"generate_area: {flexible_dict}")
    # Create a ResidueIndexSelector for each CDR and combine them into gen_selector.
    for range in pocket_residues:
        flexible_residue_first = range[0]
        flexible_residue_last = range[1]
        gen_selector1 = selections.ResidueIndexSelector()
        gen_selector1.set_index_range(
            pose.pdb_info().pdb2pose(*flexible_residue_first), 
            pose.pdb_info().pdb2pose(*flexible_residue_last), 
        )
        # Combine gen_selector and gen_selector1.
        gen_selector = OrResidueSelector(gen_selector, gen_selector1)
    nbr_selector = selections.NeighborhoodResidueSelector()
    nbr_selector.set_focus_selector(gen_selector)
    # Select CDRs and their neighboring residues.
    nbr_selector.set_include_focus_in_subset(True) 
    subset_selector = nbr_selector
    prevent_repacking_rlt = operation.PreventRepackingRLT()
    prevent_subset_repacking = operation.OperateOnResidueSubset(
        prevent_repacking_rlt, 
        subset_selector,
        flip_subset=True,
    )
    tf.push_back(prevent_subset_repacking)

    movemap = MoveMapFactory()
    movemap.add_bb_action(move_map_action.mm_enable, gen_selector)  # Allow backbone movement for selected residues.
    movemap.add_chi_action(move_map_action.mm_enable, subset_selector) # Allow sidechain movement for selected residues.
    mm = movemap.create_movemap_from_pose(pose)
    fastrelax = FastRelax()
    fastrelax.set_scorefxn(scorefxn)
    fastrelax.set_movemap(mm) # Use the default MoveMap.
    fastrelax.set_task_factory(tf)
    fastrelax.apply(pose)
    
    pose.dump_pdb(f'{output_file}')
    logging.info(f'Save relaxed structure to {output_file}')
    return output_file

def find_original_pdb(pdb_file_name, ppdbench_dir):
    base_name = os.path.basename(pdb_file_name).split('.comp_')[0]
    original_pdb_name = f"{base_name}.comp.pdb"
    original_pdb_path = os.path.join(ppdbench_dir, original_pdb_name)
    return original_pdb_path

# nohup python relax.py -i /home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-new/2026-01-16_01-02_peptide_only_swap40_block4/last/pocket/run_2026-01-19_18-04-53  > pyrosetta_1.log 2>&1 &
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, required=True)
    args = parser.parse_args()


    output_dir_original = os.path.join(args.input_dir, 'relaxed_original') 
    output_dir_designed = os.path.join(args.input_dir, 'relaxed_designed')
    os.makedirs(output_dir_original, exist_ok=True)
    os.makedirs(output_dir_designed, exist_ok=True)
    metrics_path = os.path.join(args.input_dir, 'interface_energy_metrics.csv')

    result = {"item":[], "dg_original": [], "dg_designed": []}
    ppdbench_dir = "/home/kechen/peppocketgen/dataset/pdb/PPDBench/complexes113"

    pdbparser = PDBParser(QUIET=True)
    for pdb_file in tqdm(os.listdir(args.input_dir)):
        if not pdb_file.endswith('.pdb'):
            continue
        pdb_file_path_designed = os.path.join(args.input_dir, pdb_file)
        pdb_file_path_orginal = find_original_pdb(pdb_file, ppdbench_dir)

        structure = pdbparser.get_structure('pdb', pdb_file_path_orginal)
        all_chains = list(structure[0].get_chains())
        assert len(all_chains) == 2, f"Expected 2 chains in the original PDB, but found {len(all_chains)}"
        interface = f"{all_chains[0].id}_{all_chains[1].id}"
        
        pocket_residues = extract_residues_from_filename(pdb_file)
        relaxed_pdb_orginal = relax(pdb_file_path_orginal, pocket_residues, output_dir_original)
        relaxed_pdb_designed = relax(pdb_file_path_designed, pocket_residues, output_dir_designed)
        dg_original = pyrosetta_interface_energy(relaxed_pdb_orginal, interface)
        result["item"].append(pdb_file)
        result["dg_original"].append(dg_original)
        dg_designed = pyrosetta_interface_energy(relaxed_pdb_designed, interface)
        result["dg_designed"].append(dg_designed)

    df = pd.DataFrame(result)
    df.to_csv(metrics_path, index=False)
    print(f'Saved interface energy metrics to {metrics_path}')
    print(df)