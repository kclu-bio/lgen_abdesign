import abc
import os
import numpy as np
import pandas as pd
import logging
import tree
import torch
import random
import pickle
import warnings

from glob import glob
from torch.utils.data import Dataset
from data import utils as du
from data import dataset_utils as dsu
from openfold.data import data_transforms
from openfold.utils import rigid_utils
from data import protein

class PrepareCSV(abc.ABC):
    def __init__(self, 
                dataset_cfg,
                task = None,
                finetune = False,
                cluster_file_multichain = True):
        self._log = logging.getLogger(__name__)
        self._dataset_cfg = dataset_cfg
        self.task = task
        self.raw_csv = pd.read_csv(self._dataset_cfg.csv_path)
        self.fixed_test = getattr(self._dataset_cfg, "fixed_test", "")
        self.finetune = finetune
        # Convert to lowercase.
        self.raw_csv["pdb"] = self.raw_csv["item_name"].apply(lambda x: x.split("_")[0])

        self.ppdbench_list = get_ppdbench(self._dataset_cfg.ppdbench_path)
        self.antibody_list = pickle.load(open(self._dataset_cfg.antibody_list_path, "rb"))

        self._log.info(f"items before filter: {len(self.raw_csv)}")
        metadata_csv = self._filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values(
            'modeled_seq_len', ascending=False)
        self._log.info(f"items after filter: {len(metadata_csv)}")
        
        if self._dataset_cfg.cluster_path is not None:
            # Key: uppercase PDB ID.
            self._pdb_to_cluster = dsu._read_clusters(self._dataset_cfg.cluster_path, multichain=cluster_file_multichain)
            self._max_cluster = max(sum(self._pdb_to_cluster.values(), []))
            self._missing_pdbs = 0
            metadata_csv['cluster'] = metadata_csv['item_name'].map(self._cluster_lookup)
            if self.fixed_test != "":
                fixed_df = pd.read_csv(self.fixed_test)
                fixed_test_clusters = metadata_csv[metadata_csv['item_name'].isin(fixed_df['item_name'])]['cluster']
                fixed_test_cluster_set = set(np.concatenate(fixed_test_clusters.values))
                metadata_csv = metadata_csv[~metadata_csv['cluster'].apply(self.has_overlap, cluster_set=fixed_test_cluster_set)].reset_index(drop=True)
                self._log.info(f"items after filtering out clusters overlapping with fixed test set ({self.fixed_test}): {len(metadata_csv)}")
        self.metadata_csv = metadata_csv

    def _cluster_lookup(self, pdb) -> list:
        pdb = pdb.upper()
        if pdb not in self._pdb_to_cluster:
            #warnings.warn(f'Cluster not found for {pdb}', category=UserWarning)
            self._pdb_to_cluster[pdb] = [self._max_cluster + 1]
            self._max_cluster += 1
            self._missing_pdbs += 1
        return self._pdb_to_cluster[pdb]
    
    def _filter_metadata(self, raw_csv: pd.DataFrame) -> pd.DataFrame:
        self._log.info(
            f'Raw data has: {len(raw_csv)} examples')
        
        resolution_max = getattr(self._dataset_cfg.filter, "max_resolution", None)
        if resolution_max is not None and "resolution" in raw_csv.columns:
            raw_csv = _resolution_filter(raw_csv, resolution_max)
            self._log.info(
                f'Dataset has: {len(raw_csv)} examples after filtering by resolution <= {resolution_max}')
            
        filter_cfg = self._dataset_cfg.filter
        if ("type" in raw_csv.columns) and self._dataset_cfg.ligand_only:
            raw_csv = raw_csv[raw_csv['pocket_len'] > 0]
            self._log.info(
                f'Dataset has: {len(raw_csv)} examples after filtering out non-ligand examples')
            if getattr(self._dataset_cfg, "peptide_only", False):
                raw_csv = raw_csv[raw_csv['type'] == "merged_propedia_5.5"]
                self._log.info(
                    f'Dataset has: {len(raw_csv)} examples after filtering out non-peptide ligand examples')
        
        data_csv = _peptide_length_filter(raw_csv,
                                            filter_cfg.min_peptide_len,
                                            filter_cfg.max_peptide_len)
        self._log.info(
            f'Dataset has: {len(data_csv)} examples after filtering by peptide length')
        data_csv =  _pocket_size_filter(data_csv, 
                                        filter_cfg.min_pocket_size,)
        self._log.info(
            f'Dataset has: {len(data_csv)} examples after filtering by pocket size')
        data_csv = _non_standard_filter(data_csv)
        self._log.info(
            f'Dataset has: {len(data_csv)} examples after filtering by nonstandard residue in pocket size.')

        data_csv = _antibody_filter(data_csv,
                                    antibody_list = self.antibody_list,
                                    retain = filter_cfg.retain_antibodies)
        self._log.info(
            f'Dataset has: {len(data_csv)} examples after filtering by excluding antibodies')
        
        data_csv = _ligand_atom_filter(data_csv,
                                    min_atom_num = filter_cfg.min_atom_num,
                                    max_atom_num = filter_cfg.max_atom_num)
        self._log.info(
            f'Dataset has: {len(data_csv)} examples after filtering by ligand atom numbers.')
        data_csv = dsu._length_filter(data_csv,
                                        filter_cfg.min_num_res,
                                        filter_cfg.max_num_res)
        self._log.info(
            f'Dataset has: {len(data_csv)} examples after filtering by protein length')

        use_charge = getattr(self._dataset_cfg, 'use_charge', False)
        if use_charge:
            assert "has_charge" in data_csv.columns, "Metadata must contain 'has_charge' column when using charge features."
            data_csv = data_csv[data_csv["has_charge"] == True]

            self._log.info(f'Dataset has: {len(data_csv)} examples after filtering by charge presence')
        return data_csv
    
    def _ppdbench_filter(self, data_csv: pd.DataFrame, ppdbench_list: list, retain = False):
        if retain:
            return data_csv
        else:
            ppdbench_clusters = data_csv[data_csv['pdb'].isin(ppdbench_list)]['cluster']
            ppdbench_clusters_set = set(np.concatenate(ppdbench_clusters.values))
            print("ppdbench cluster number:", len(ppdbench_clusters_set))
            mask = ~data_csv['cluster'].apply(self.has_overlap, cluster_set=ppdbench_clusters_set)
            data_csv = data_csv[mask].reset_index(drop=True)
            return data_csv

    def has_overlap(self, cluster_list, cluster_set):
            return bool(set(cluster_list) & cluster_set)
    
    def create_split(self, diff_cluster = False):
        random.seed(self._dataset_cfg.seed)
        data_csv = self.metadata_csv.reset_index(drop=True)

        data_csv['index'] = list(range(len(data_csv)))

        #(1) Filter by max eval length.
        if self._dataset_cfg.max_eval_length is None:
            eval_lengths = data_csv.modeled_seq_len
        else:
            eval_lengths = data_csv.modeled_seq_len[
                data_csv.modeled_seq_len <= self._dataset_cfg.max_eval_length 
            ]
        
        # (2) Get length for evaluation
        all_lengths = np.sort(eval_lengths.unique())
        # Generate num_eval_length evenly spaced values in [0, 1] and scale by the number of lengths.
        length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self._dataset_cfg.num_eval_lengths)
        length_indices = length_indices.astype(int) # Generate evenly distributed indices.
        eval_lengths = all_lengths[length_indices] # Lengths used during validation.

        #(3) sample validation indices (based on eval length and minimum cluster)
        val_indices = []        
        # Fix a random seed to get the same split each time.
        for length in eval_lengths:
            length_samples = data_csv[data_csv.modeled_seq_len == length]
            if len(length_samples) > 0:
                samples = length_samples.sample(
                    self._dataset_cfg.samples_per_eval_length,
                    replace=True, 
                    random_state=self._dataset_cfg.seed
                )
                val_indices.extend(samples['index'].tolist())
        val_indices = list(set(val_indices))
        val_csv = data_csv.loc[val_indices].reset_index(drop=True)
        
        # Remove half of the monomers.
        if (not self.finetune) and ("type" in val_csv.columns) and ("monomer_no_ligand" in val_csv["type"].values):
            monomer_indices = val_csv[val_csv["type"] == "monomer_no_ligand"].index.tolist()
            num_to_remove = len(monomer_indices) * (1 - self._dataset_cfg.monomer_ratio) // 1
            indices_to_remove = random.sample(monomer_indices, int(num_to_remove))
            val_csv = val_csv.drop(indices_to_remove).reset_index(drop=True)
            val_indices = val_csv['index'].tolist()

        #(4) Avoid validation set having the same cluster as training set.
        # One item will be in multiple clusters.
        # Example: for one item in validation set, val_cluster = [1,2]
        # Any other items in cluster 1 or 2 will be removed.   e.g. [2, 5] [1, 100]
        if diff_cluster:
            train_csv = self.exclude_cluster(data_csv, val_indices)
            self._log.info(f"{len(train_csv)}, {len(val_csv)}, after filter by validation set clusters")
        else:
            val_pdbs = val_csv['pdb'].unique()
            train_csv = data_csv[~data_csv['pdb'].isin(val_pdbs)].reset_index(drop=True)
            self._log.info(f"{len(train_csv)}, {len(val_csv)}, without filtering by validation set clusters")
        #(5) Avoid training set having the same cluster as the ppdbench set.
        train_csv = self._ppdbench_filter(train_csv,
                                ppdbench_list = self.ppdbench_list,
                                retain = self._dataset_cfg.filter.retain_ppdbench)

        train_csv["index"] = list(range(len(train_csv)))
        val_csv["index"] =  list(range(len(val_csv)))
        self._log.info(
            f'Train dataset has: {len(train_csv)} examples, validation dataset has {len(val_csv)} examples')
        return train_csv, val_csv

    def create_split_by_agtype(self, diff_cluster = False):
        random.seed(self._dataset_cfg.seed)
        data_csv = self.metadata_csv.reset_index(drop=True)
        data_csv['index'] = list(range(len(data_csv)))
        sample_sizes = {'peptide': self._dataset_cfg.pep_val_num, 'sugar': self._dataset_cfg.sugar_val_num, 'hapten': self._dataset_cfg.hapten_val_num, 'protein': self._dataset_cfg.protein_val_num}
        val_csv = data_csv.groupby('agtype', group_keys=True).apply(
            lambda x: x.sample(n=sample_sizes[x.name], random_state=self._dataset_cfg.seed)
        ).reset_index(drop=True)
        val_indices = val_csv["index"].to_list()
        if diff_cluster:
            train_csv = self.exclude_cluster(data_csv, val_indices)
            self._log.info(f"{len(train_csv)}, {len(val_csv)}, after filter by validation set clusters")
        else:
            train_csv = data_csv.drop(val_indices).reset_index(drop=True)
            self._log.info(f"{len(train_csv)}, {len(val_csv)}, without filtering by validation set clusters")
        
        train_csv["index"] = list(range(len(train_csv)))
        val_csv["index"] =  list(range(len(val_csv)))

        return train_csv, val_csv

    def exclude_cluster(self, data_csv, val_indices):
        val_clusters = data_csv.loc[val_indices, 'cluster']
        val_cluster_set = set(np.concatenate(val_clusters.values))
    
        train_mask = ~data_csv['cluster'].apply(self.has_overlap,  cluster_set = val_cluster_set)
        train_csv = data_csv[train_mask].reset_index(drop=True)
        return train_csv
    
def _peptide_length_filter(data_csv, min_peptide_len, max_peptide_len):
    """
    Filter the dataset based on the length of the peptide.
    """
    return data_csv[ (data_csv.peptide_len == 0) |
        (data_csv.peptide_len >= min_peptide_len)
        & (data_csv.peptide_len <= max_peptide_len)
    ]

def _pocket_size_filter(data_csv, min_pocket_size):
    if "type" in data_csv.columns:
        # For ligand dataset, we want to keep all examples with pocket_len = 0 (non-ligand examples) for negative sampling.
        return data_csv[(data_csv.pocket_len == 0) | (data_csv.pocket_len >= min_pocket_size)]
    else:
        return data_csv[data_csv.pocket_len >= min_pocket_size]
    
def _resolution_filter(data_csv, max_resolution):
    return data_csv[data_csv.resolution <= max_resolution]
    
def _non_standard_filter(data_csv):
    return data_csv[data_csv.non_standard_in_pocket == False]

def _antibody_filter(data_csv:pd.DataFrame, antibody_list:list, retain = True):
    
    if retain:
        return data_csv
    else:
        return data_csv[~data_csv.pdb.isin(antibody_list)]

def _ligand_atom_filter(data_csv, min_atom_num, max_atom_num):
    return data_csv[ (data_csv.num_ligand_atoms == 0) |
        (data_csv.num_ligand_atoms >= min_atom_num)
        & (data_csv.num_ligand_atoms <= max_atom_num)
    ]

def get_ppdbench(ppdbench_path: str)->list:
    # Return a list contianing ppdbbench pdb id lists
    ppdbench_list = []
    for filename in os.listdir(ppdbench_path):
        if filename.endswith("comp.pdb"):
            pdb_id = filename.split(".")[0]
            ppdbench_list.append(pdb_id)
        else:
            continue
    return ppdbench_list

def _prepare_peptide(ligand_dict):
        return {
            "ligand_elements": torch.tensor(ligand_dict["element"]),
            "ligand_pos": torch.tensor(ligand_dict["ligand_pos"]),
            "atom_residue": torch.tensor(ligand_dict["atom_residue"])
        }        


def _process_csv_row(processed_file_path,
                     predict_sidechain:bool, 
                     multichain:bool):
    processed_feats = du.read_pkl(processed_file_path)

    #---------------parse receptor features----------------
    receptor_feats = processed_feats["receptor"]
    # dict_keys(['atom_positions', 'aatype', 'atom_mask', 'residue_index', 'chain_index', 'b_factors', 'bb_center', 'bb_positions', 'pocket_mask', 'nonhetatm_idx', 'nonhetatm_mask'])
    # Only take modeled residues.
    nonhetatm_idx = receptor_feats['nonhetatm_idx']

    #*This can lead to a difference in the number of residues between the input and output files.
    min_idx = np.min(nonhetatm_idx)
    max_idx = np.max(nonhetatm_idx)
    del receptor_feats['nonhetatm_idx']
    receptor_feats = tree.map_structure(
        lambda x: x[min_idx:(max_idx+1)], receptor_feats) #only retain elements from min_idx to max_idx

    # Run through OpenFold data transforms.
    chain_feats = {
        'aatype': torch.tensor(receptor_feats['aatype']).long(),
        'all_atom_positions': torch.tensor(receptor_feats['atom_positions']).double(),
        'all_atom_mask': torch.tensor(receptor_feats['atom_mask']).double()
    }

    # new key: {'rigidgroups_gt_frames', 'rigidgroups_gt_exists', 'rigidgroups_group_is_ambiguous', 'rigidgroups_group_exists', 'rigidgroups_alt_gt_frames'}
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_gt_frames'])[:, 0] #get RigidGroup0 (bb group)
    rotmats_1 = rigids_1.get_rots().get_rot_mats()
    trans_1 = rigids_1.get_trans()
    #res_plddt = receptor_feats['b_factors'][:, 1]
    res_mask = torch.tensor(receptor_feats['nonhetatm_mask']).int()
    pocket_mask = torch.tensor(receptor_feats['pocket_mask']).int()
    if "diffuse_mask" in receptor_feats:
        diffuse_mask = torch.tensor(receptor_feats['diffuse_mask']).int()
    else:
        diffuse_mask = None
    if multichain:
    # Re-number residue indices for each chain such that it starts from 1.
    # Original PDB chain IDs may differ; renumber chains sequentially from 1.
    # protein.PDB_CHAIN_IDS：'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
        old_chain_idx = receptor_feats['chain_index']
        old_chain_idx = np.searchsorted(np.array(list(protein.PDB_CHAIN_IDS)), old_chain_idx) #turn letter to number
        all_old_chain_idx, inverse = np.unique(old_chain_idx, return_inverse=True)
        new_values = np.random.permutation(np.arange(0, len(all_old_chain_idx) ))
        chain_idx = new_values[inverse]
        '''
        old_chain_idx:    A  A  A  B  B  B  C
                        ↓  ↓  ↓  ↓  ↓  ↓  ↓
        inverse:          0  0  0  1  1  1  2
                        ↓  ↓  ↓  ↓  ↓  ↓  ↓
        new_values:            [2, 0, 1]
                        ↓  ↓  ↓  ↓  ↓  ↓  ↓
        chain_idx:        2  2  2  0  0  0  1
        '''
        old_res_idx = receptor_feats['residue_index']
        # Number residues in each chain starting from 1.
        res_idx = np.zeros_like(old_res_idx)
        for i, chain_id in enumerate(new_values):
            chain_mask = (chain_idx == chain_id)# Mask for the current chain.
            current_res_indices = old_res_idx[chain_mask]

            if len(current_res_indices) > 0:
                chain_min = np.min(current_res_indices)
                res_idx[chain_mask] = current_res_indices - chain_min + 1
    else:
        # Temporary behavior: increment all residue indices from 1 regardless of chain.
        chain_idx = receptor_feats['chain_index']
        old_res_idx = receptor_feats['residue_index']
        res_idx = np.arange(1, len(old_res_idx)+1)
        chain_idx = np.searchsorted(np.array(list(protein.PDB_CHAIN_IDS)), chain_idx) #turn letter to number
        old_chain_idx = chain_idx
    if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
        raise ValueError(f'Found NaNs in {processed_file_path}')

    # -----------------parse ligand features----------------
    ligand_dict = processed_feats["ligand"]
    ligand_feats = _prepare_peptide(ligand_dict)
    ligand_object = processed_feats["ligand_object"]
    # -----------------all features----------------
    feats = {
        #'res_plddt': res_plddt,
        'aatypes_1': chain_feats['aatype'],
        'rotmats_1': rotmats_1,
        'trans_1': trans_1,
        'res_mask': res_mask,
        'pocket_mask': pocket_mask,
        'diffuse_mask': diffuse_mask,
        'old_chain_idx': old_chain_idx,
        'chain_idx': chain_idx,
        'res_idx': res_idx,
        "old_res_idx": old_res_idx,
        "atom37_gt_positions": chain_feats['all_atom_positions'],
        "insertion_code": receptor_feats["insertion_code"] if "insertion_code" in receptor_feats else None,
        'ligand_object': ligand_object
    }
    feats.update(ligand_feats)
    #---- add feats for sidechain prediction----
    if predict_sidechain:
        all_atom_keys = ['rigidgroups_gt_frames', 'rigidgroups_gt_exists', 'rigidgroups_group_is_ambiguous', 'rigidgroups_group_exists', 'rigidgroups_alt_gt_frames', \
                     'residx_atom14_to_atom37', 'atom37_atom_exists', 'atom14_atom_exists', 'residx_atom37_to_atom14', \
                     'atom14_gt_exists', 'atom14_atom_is_ambiguous', 'atom14_gt_positions', 'atom14_alt_gt_positions', 'atom14_alt_gt_exists', \
                     'torsion_angles_sin_cos', 'alt_torsion_angles_sin_cos', 'torsion_angles_mask']
        # new key: {'residx_atom14_to_atom37', 'atom37_atom_exists', 'atom14_atom_exists', 'residx_atom37_to_atom14'}
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        # new_key: {'atom14_gt_exists', 'atom14_atom_is_ambiguous', 'atom14_gt_positions', 'atom14_alt_gt_positions', 'atom14_alt_gt_exists'}
        chain_feats = data_transforms.make_atom14_positions(chain_feats)
        # new_key: {'torsion_angles_sin_cos', 'alt_torsion_angles_sin_cos', 'torsion_angles_mask'}
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)
        all_atom_dict = {}
        for key in all_atom_keys:
            all_atom_dict[key] = chain_feats[key]
        feats.update(all_atom_dict)
    
    return feats


class ComplexDataset(Dataset):
    def __init__(self, 
                 csv,
                 dataset_cfg,
                 is_predict = False):
        self._log = logging.getLogger(__name__)
        self._dataset_cfg = dataset_cfg
        self.csv = csv
        self.task = self.dataset_cfg.task
        self._cache = {}
        self.load_charge = getattr(self._dataset_cfg, 'use_charge', False)
        if is_predict:
            self._dataset_cfg.csv_path = self._dataset_cfg.predict_csv_path
        self.save_csv(os.path.join(os.path.dirname(self._dataset_cfg.csv_path), f'dataset_{len(self.csv)}.csv'))
        if self.load_charge:
            self._log.info(f"Dataset will load charge features. Make sure the processed files contain charge features and the metadata csv has 'has_charge' column.")
            self.charge_inter_dict = du.read_pkl(self._dataset_cfg.charge_inter_path)

        #self._rng = np.random.default_rng(seed=self._dataset_cfg.seed)
    
    @property
    def is_training(self):
        return self._is_training

    @property
    def dataset_cfg(self):
        return self._dataset_cfg
    
    def save_csv(self, save_path):
        """
        Save the dataset csv to a file.
        """
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        self.csv.to_csv(save_path, index=False)
        self._log.info(f'Saved dataset csv to {save_path}')

    def __len__(self):
        return len(self.csv)
    
    def process_csv_row(self, csv_row):
        path = csv_row['processed_path']
        seq_len = csv_row['modeled_seq_len']
        # Large protein files are slow to read. Cache them.
        use_cache = seq_len > self._dataset_cfg.cache_num_res
        if use_cache and path in self._cache:
            return self._cache[path]
        
        use_charge = getattr(self._dataset_cfg, 'use_charge', False)

        processed_row = _process_csv_row(path, 
                                         predict_sidechain=self.dataset_cfg.predict_sidechain,
                                         multichain = getattr(self.dataset_cfg, 'multichain', False),
        )
        processed_row['item_name'] = csv_row['item_name']
        aatypes_1 = du.to_numpy(processed_row['aatypes_1'])
        if len(set(aatypes_1)) == 1:
            raise ValueError(f'Example {path} has only one amino acid.')
        if use_cache:
            self._cache[path] = processed_row
        if self.load_charge:
            charge_tensor = torch.tensor(self.charge_inter_dict[csv_row["item_name"]]["charge"])
            processed_row["ligand_charge"] = torch.nan_to_num(charge_tensor, nan=0.0).clamp(self._dataset_cfg.charge_min, self._dataset_cfg.charge_max)
        return processed_row
        
    def __getitem__(self, row_idx):
        # Process data example.
        csv_row = self.csv.iloc[row_idx]
        feats = self.process_csv_row(csv_row)

        # Hallucination: Given peptide, design the whole protein.
        # Inpainting: Given peptide and protein scaffold, design the procket.
        if feats["diffuse_mask"] is None:
            if self.task == 'hallucination':
                feats['diffuse_mask'] = torch.ones_like(feats['res_mask']).bool()
            elif self.task == 'inpainting':
                feats['diffuse_mask'] = feats['pocket_mask'].bool()
            else:
                raise ValueError(f'Unknown task {self.task}')
            feats['diffuse_mask'] = feats['diffuse_mask'].int()

        # Storing the csv index is helpful for debugging.
        feats['csv_idx'] = torch.ones(1, dtype=torch.long) * row_idx

        return feats