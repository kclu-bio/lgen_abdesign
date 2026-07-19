# original version: https://github.com/zaixizhang/PocketGen/blob/37d826cb1622b33b5411bdf0b0d08737233d7328/utils/evaluation/docking_vina.py

from openbabel import pybel
from vina import Vina
import subprocess
import tempfile
import AutoDockTools
import os
import contextlib


def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper


class PrepLig(object):
    def __init__(self, input_mol, mol_format):
        self.rdkit_mol = None
        if mol_format == 'smi':
            self.ob_mol = pybel.readstring('smi', input_mol)
        elif mol_format == 'sdf' or mol_format == 'pdb':
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))
        else:
            raise ValueError(f'mol_format {mol_format} not supported')
        
    def addH(self, polaronly=False, correctforph=True, PH=7): 
        self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)
        self.rdkit_mol = None
        self.ob_mol.write('sdf', 'tmp_h.sdf', overwrite=True)

    def get_pdbqt(self, lig_pdbqt=None):
        if lig_pdbqt is not None: 
            self.ob_mol.write("pdbqt", lig_pdbqt, overwrite=True)
            return 
        else: 
            return self.ob_mol.write("pdbqt", lig_pdbqt, overwrite=True)
        

class PrepProt(object): 
    def __init__(self, pdb_file): 
        self.prot = pdb_file
    
    def del_water(self, dry_pdb_file): # optional
        with open(self.prot) as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')] 
            dry_lines = [l for l in lines if not 'HOH' in l]
        
        with open(dry_pdb_file, 'w') as f:
            f.write(''.join(dry_lines))
        self.prot = dry_pdb_file
        
    def addH(self, prot_pqr):  # call pdb2pqr
        self.prot_pqr = prot_pqr
        subprocess.Popen(['pdb2pqr30','--ff=AMBER',self.prot, self.prot_pqr],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()

    def get_pdbqt(self, prot_pdbqt):
        prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
        subprocess.Popen(['python3', prepare_receptor, '-r', self.prot_pqr, '-o', prot_pdbqt],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()


class VinaDock(object): 
    def __init__(self, lig_pdbqt, prot_pdbqt): 
        self.lig_pdbqt = lig_pdbqt
        self.prot_pdbqt = prot_pdbqt
    
    def _max_min_pdb(self, pdb, buffer):
        with open(pdb, 'r') as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HEATATM')]
            xs = [float(l[31:39]) for l in lines]
            ys = [float(l[39:47]) for l in lines]
            zs = [float(l[47:55]) for l in lines]
            print(max(xs), min(xs))
            print(max(ys), min(ys))
            print(max(zs), min(zs))
            pocket_center = [(max(xs) + min(xs))/2, (max(ys) + min(ys))/2, (max(zs) + min(zs))/2]
            box_size = [(max(xs) - min(xs)) + buffer, (max(ys) - min(ys)) + buffer, (max(zs) - min(zs)) + buffer]
            return pocket_center, box_size
    
    def get_box(self, ref=None, buffer=0):
        '''
        ref: reference pdb to define pocket. 
        buffer: buffer size to add 

        if ref is not None: 
            get the max and min on x, y, z axis in ref pdb and add buffer to each dimension 
        else: 
            use the entire protein to define pocket 
        '''
        if ref is None: 
            ref = self.prot_pdbqt
        self.pocket_center, self.box_size = self._max_min_pdb(ref, buffer)
        print(self.pocket_center, self.box_size)

    def dock(self, score_func='vina', seed=0, mode='dock', exhaustiveness=8, save_pose=False, **kwargs):  # seed=0 mean random seed
        v = Vina(sf_name=score_func, seed=seed, verbosity=0, **kwargs)
        v.set_receptor(self.prot_pdbqt)
        v.set_ligand_from_file(self.lig_pdbqt)
        v.compute_vina_maps(center=self.pocket_center, box_size=self.box_size)
        if mode == 'score_only': 
            score = v.score()[0]
        elif mode == 'minimize':
            score = v.optimize()[0]
        elif mode == 'dock':
            v.dock(exhaustiveness=exhaustiveness, n_poses=1)
            score = v.energies(n_poses=1)[0][0]
        else:
            raise ValueError
        
        if not save_pose: 
            return score
        else: 
            if mode == 'score_only': 
                pose = None 
            elif mode == 'minimize': 
                tmp = tempfile.NamedTemporaryFile()
                with open(tmp.name, 'w') as f: 
                    v.write_pose(tmp.name, overwrite=True)             
                with open(tmp.name, 'r') as f: 
                    pose = f.read()
   
            elif mode == 'dock': 
                pose = v.poses(n_poses=1)
            else:
                raise ValueError
            return score, pose



# if __name__ == '__main__':
#     lig_pdbqt = 'data/lig.pdbqt'
#     mol_file = 'data/1a4k_ligand.sdf'
#     a = PrepLig(mol_file, 'sdf')
#     # mol_file = 'CC(=C)C(=O)OCCN(C)C'
#     # a = PrepLig(mol_file, 'smi')
#     a.addH()
#     a.gen_conf()
#     a.get_pdbqt(lig_pdbqt)
#
#     prot_file = 'data/1a4k_protein_chainAB.pdb'
#     prot_dry = 'data/protein_dry.pdb'
#     prot_pqr = 'data/protein.pqr'
#     prot_pdbqt = 'data/protein.pdbqt'
#     b = PrepProt(prot_file)
#     b.del_water(prot_dry)
#     b.addH(prot_pqr)
#     b.get_pdbqt(prot_pdbqt)
#
#     dock = VinaDock(lig_pdbqt, prot_pdbqt)
#     dock.get_box()
#     dock.dock()