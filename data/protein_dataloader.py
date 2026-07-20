"""Protein data loader."""
import math
import torch
import torch
import logging
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler, dist
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

class Collator:
    def __init__(
        self, 
        crop: bool = True, 
        dynamic_mask:bool = True,
        crop_threshold: int = 350, 
        predict_sidechain: bool = True,
        swap_prob: float = 0.1, 
        sidechain_prob: float = 0.15, 
        max_seg_len: int = 15,
        ligand_only: bool = False,
        atom_residue_padding_val: int = 21
    ):
        self.crop = crop
        self.crop_threshold = crop_threshold
        self.predict_sidechain = predict_sidechain
        self.swap_prob = swap_prob
        self.sidechain_prob = sidechain_prob
        self.max_seg_len = max_seg_len
        self.dynamic_mask = dynamic_mask
        self.ligand_only = ligand_only
        self.atom_residue_padding_val = atom_residue_padding_val

    def __call__(self, batch: list[dict]) -> dict:
        special_keys = ['ligand_pos', 'ligand_elements', "atom_residue"]
        if "ligand_charge" in batch[0]:
            special_keys.append("ligand_charge")
        normal_keys = [key for key in batch[0].keys() if key not in special_keys]
        collated_batch = {}
        padding_val = {'ligand_pos': (0.0, float), 'ligand_elements':(0, int), "atom_residue":(self.atom_residue_padding_val, int), "ligand_charge":(0.0, float)}

        # handle ligand features
        for key in special_keys:
            sequences = [sample[key] for sample in batch]
            padded = pad_sequence(sequences, batch_first=True, padding_value=padding_val[key][0]).to(dtype=getattr(torch, f"{padding_val[key][1].__name__}"))
            
            if key == 'ligand_elements':
                collated_batch['ligand_mask'] = (padded != 0).float()
            collated_batch[key] = padded

        # handle receptor features
        for key in normal_keys:
            values = [sample[key] for sample in batch]
            if isinstance(values[0], torch.Tensor):
                collated_batch[key] = torch.stack(values, dim=0)
            else:
                try:
                    collated_batch[key] = torch.as_tensor(np.array(values))
                except:
                    collated_batch[key] = values
        
        # crop logic. Must after the ligand padding logic
        N_res = collated_batch["aatypes_1"].shape[1] 
        if self.crop and N_res > self.crop_threshold:
            # [B, k]
            crop_indices = self.get_crop_idx(collated_batch["trans_1"], collated_batch["ligand_pos"], collated_batch["ligand_mask"])
            B, k = crop_indices.shape
            assert k<=self.crop_threshold, f"k:{k} must be smaller than crop threshold: {self.crop_threshold}"
            for key in normal_keys:
                data = collated_batch[key]
                if isinstance(collated_batch[key], torch.Tensor) and data.shape[1] == N_res:
                    # Always gather on the first dimension
                    # aatype: [B, N] -> 1
                    # trans: [B, N, 3] -> 2
                    # rotmats: [B, N, 3, 3] -> 3
                    # rigidgroups_gt_frames: [B, N, 8, 4, 4] ->4
                    dims_to_append = data.shape[2:]
                    # [B, k, 1, ..., 1]
                    crop_indices_feat = crop_indices.clone().view(B, k, *([1]*len(dims_to_append)))
                    # [B, k, feat_dim]
                    crop_indices_feat = crop_indices_feat.expand(B, k, *dims_to_append)
                    # [B, k, feat_dim]
                    collated_batch[key] = torch.gather(data, dim = 1, index = crop_indices_feat)
        # dynamic diffuse_mask and sidechain mask
        if self.dynamic_mask:
            collated_batch = self.apply_dynamic_mask(collated_batch)
        return collated_batch

    def get_crop_idx(self, trans_1, ligand_pos, ligand_mask, k_range=(250, 350)):
        """
            trans1:  Tensor of receptor CA atom coordinates
                (B, N, 3)
            ligand_coord: Tensor of ligand atom coordinates.
                (B, M, 3) 
            ligand_mask: Tensor of ligand atom coordinates.
                (B, M) 
        """

        B, N, _ = trans_1.shape
        device = trans_1.device
        # select a random k for each batch within the specified range
        k = torch.randint(k_range[0], k_range[1] + 1, (1,)).item()

        # 1. Handle cases with a ligand: compute ligand centers (B, 3)
        # (B, M, 3) * (B, M, 1) -> sum(dim=1) -> (B, 3)
        ligand_sum = (ligand_pos * ligand_mask[..., None]).sum(dim=1)
        # [B, 1]
        ligand_count = ligand_mask.sum(dim=1, keepdim=True).clamp(min=1e-6)
        # [B, 3] /[B, 1] =[B, 3]
        ligand_centers = ligand_sum / ligand_count 

        # 2. Handle ligand-free cases: randomly select a residue as center
        # [B, ]
        random_res_idx = torch.randint(0, N, (B,), device=device)
        # Use gather to get the randomly selected coordinates for each batch
        # Argument 1 (trans_1): data source, shape (B, N, 3).
        # Argument 2 (dim=1): which dimension to index? Here it's N (residue dimension).
        # Argument 3 (Index): the prepared index random_res_idx of shape (B, 1, 3): [B,] -> [B, 1, 1] -> [B, 1, 3]
        # Output dimensions depend on the index
        random_centers = torch.gather(trans_1, 1, random_res_idx.view(B, 1, 1).expand(B, 1, 3)).squeeze(1)

        # 3. Unified logic: determine which batches have ligands
        # has_ligand: (B,)
        has_ligand = (ligand_mask.sum(dim=1) > 0)
        
        # 4. Combine centers: choose ligand_centers if ligand exists, otherwise random_centers
        centers = torch.where(has_ligand.unsqueeze(-1), ligand_centers, random_centers).float()

        # 4. Compute distances from all residues to the center (B, N)
        # trans_1: (B, N, 3), centers: (B, 1, 3)
        # dist: [B, N, 1]
        dist = torch.cdist(trans_1, centers.unsqueeze(1), p=2).squeeze(-1)

        # 6. Take TopK closest residues
        # indices: (B, k)
        values, indices = torch.topk(dist, k, largest=False, sorted=False)
        indices, _ = torch.sort(indices, dim=-1)
        return indices


    def apply_dynamic_mask(self, collated_batch):
        # diffuse_mask: [B, N], res_mask: [B, N]
        # Initially, diffuse_mask equals pocket_mask
        diffuse_mask = collated_batch.get("diffuse_mask")
        res_mask = collated_batch["res_mask"]
        B, N = res_mask.shape
        device = res_mask.device

        if self.predict_sidechain:
            # For all samples, randomly select sidechain_prob * len residues to predict sidechains
            # [B, N]
            sidechain_mask = (torch.rand(B, N, device=device) < self.sidechain_prob).float()

            # Sample which batches require swapping [B, 1]
            swap_batch = (torch.rand(B, 1, device=device) < self.swap_prob).float()
            
            # If swap_batch is 1 and the region was originally diffuse: set diffuse to 0 and sidechain to 1
            # For swap samples (swap_batch=1): clear the original sidechain_mask
            sidechain_mask = sidechain_mask * (1.0 - swap_batch)
            # Record the original diffuse area where value is 1
            pocket_area = (diffuse_mask == 1).float()
            # Perform swap: if training includes ligand-free structures, clear diffuse_mask in pocket area for swap samples
            if not self.ligand_only:
                diffuse_mask = diffuse_mask * (1 - swap_batch * pocket_area)
            # sidechain_mask: force to 1 in pocket area for swap samples
            sidechain_mask = torch.maximum(sidechain_mask, swap_batch * pocket_area)
            collated_batch["sidechain_mask"] = sidechain_mask * res_mask
        # Enhance diffuse_mask regions
        
        if self.max_seg_len > 0 and not self.ligand_only:
            # For each sample, sample 2 start points and 2 lengths
            # [B, N] -> [B, 1] -> [B, 2]
            existing_diffuse_len = torch.sum(diffuse_mask, dim=-1, keepdim= True).repeat(1,2)
            # [B, 2]
            starts = torch.randint(0, N-self.max_seg_len, (B, 2), device=device)
            # [B, 2]
            lens = torch.randint(3, self.max_seg_len + 1, (B, 2), device=device)- existing_diffuse_len//2
            lens = lens.clamp(min = 0)
            # Check which segments have length 0
            # [B, 2]
            zero_len_mask = (lens == 0)
            
            # If all segments in the batch have length 0, skip directly
            if zero_len_mask.all():
                collated_batch["diffuse_mask"] = diffuse_mask * res_mask
                return collated_batch
            # [B, 2]
            ends = starts + lens

            # Create an index matrix using broadcasting [1, 1, N]
            indices = torch.arange(N, device=device).view(1, 1, N)
            
            # indices: [1, 1, N], starts/ends: [B, 2, N]
            # seg_masks: [B, 2, N]
            # Explanation: for each B, the first [1,N] indicates mask in the first interval, the second [1,N] indicates mask in the second interval
            seg_masks = (indices >= starts.unsqueeze(-1)) & (indices < ends.unsqueeze(-1))
            # [B, N]
            seg_masks = seg_masks.any(dim=1).float()
            
            # Apply to the original mask, and only affect active samples in the batch
            # [B,N]
            diffuse_mask = torch.maximum(diffuse_mask, seg_masks)

            # current_counts: [B, 1]
            current_counts = torch.sum(diffuse_mask * res_mask, dim=-1, keepdim=True)

            if not current_counts.ge(1).all():
                # Ensure at least 1 valid point, otherwise _batch_ot will error
                min_points = 1 
                # Generate random scores per position, but give extra score only to positions with res_mask=1
                # score: [B, N]
                # res_mask * 100 ensures positions with res_mask=0 have very low scores and won't be selected
                random_scores = torch.rand(B, N, device=device) + (res_mask * 100.0)
                # Select the top min_points indices per row
                # _, topk_indices: [B, min_points]
                _, topk_indices = torch.topk(random_scores, k=min_points, dim=-1)
                # Convert these indices to a mask [B, N]
                # Use scatter_ to set the indexed positions to 1.0
                extra_mask = torch.zeros(B, N, device=device).scatter_(1, topk_indices, 1.0)
                # Only fill rows that lack enough points
                # need_fix: [B, 1] (boolean to float)
                need_fix = (current_counts < min_points).float()
                # If filling is needed, take union of original mask and extra_mask; otherwise keep original
                diffuse_mask = torch.where(need_fix > 0, torch.maximum(diffuse_mask, extra_mask), diffuse_mask)

        collated_batch["diffuse_mask"] = diffuse_mask * res_mask
        return collated_batch


class ProteinData(LightningDataModule):

    def __init__(self, *, 
                 data_cfg, 
                 train_dataset, 
                 valid_dataset, 
                 dataset_cfg, 
                 predict_dataset=None):
        super().__init__()
        self.data_cfg = data_cfg
        self.loader_cfg = data_cfg.loader
        self.sampler_cfg = data_cfg.sampler
        self.dataset_cfg = dataset_cfg
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._predict_dataset = predict_dataset
        
        self.atom_residue_padding_val = 21
    
    def train_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers

        batch_sampler = LengthBatcher(
            sampler_cfg=self.sampler_cfg,
            metadata_csv=self._train_dataset.csv,
            rank=rank,
            num_replicas=num_replicas,
        )
        collator = Collator(
            crop=getattr(self.sampler_cfg, 'crop', False),
            dynamic_mask=getattr(self.sampler_cfg, 'dynamic_mask', False),
            crop_threshold=getattr(self.sampler_cfg, 'crop_threshold', 0),
            predict_sidechain=self.sampler_cfg.predict_sidechain,
            swap_prob=getattr(self.sampler_cfg, 'swap_prob', 0),
            sidechain_prob=getattr(self.sampler_cfg, 'sidechain_prob', 0),
            max_seg_len=getattr(self.sampler_cfg, 'max_seg_len', 0),
            ligand_only=getattr(self.sampler_cfg, 'ligand_only', True),
            atom_residue_padding_val=self.atom_residue_padding_val
        )
        # about pin_memory: https://blog.csdn.net/Caesar6666/article/details/127283965
        return DataLoader(
            self._train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
            collate_fn = collator
        )

    def val_dataloader(self):
        collator = Collator(
            crop=False,
            dynamic_mask=False,
            crop_threshold=0,
            predict_sidechain=self.sampler_cfg.predict_sidechain,
            swap_prob=0,
            sidechain_prob=0,
            max_seg_len=0,
            ligand_only=True,
            atom_residue_padding_val=self.atom_residue_padding_val
        )
        return DataLoader(
            self._valid_dataset,
            sampler=DistributedSampler(self._valid_dataset, shuffle=False),
            # Split the full dataset into multiple shards so each GPU process (or node) handles a disjoint subset.
            # Without a batch_sampler, batch_size defaults to 1
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True,
            collate_fn = collator
        )

    def predict_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.loader_cfg.num_workers
        collator = Collator(
            crop=False,
            dynamic_mask=False,
            crop_threshold=10000,
            predict_sidechain=True,
            swap_prob=0,
            sidechain_prob=0,
            max_seg_len=0,
            atom_residue_padding_val=self.atom_residue_padding_val
        )

        batch_sampler = LengthBatcher(
            sampler_cfg=self.sampler_cfg,
            metadata_csv=self._predict_dataset.csv,
            rank=rank,
            shuffle=False,
            num_replicas=num_replicas,
            max_batch_repeats=1,
            is_predict=True
        )

        return DataLoader(
            self._predict_dataset,
            #sampler=DistributedSampler(self._predict_dataset, shuffle=False),
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.loader_cfg.prefetch_factor,
            persistent_workers=True,
            collate_fn = collator
        )

class LengthBatcher:

    def __init__(
            self,
            sampler_cfg,
            metadata_csv,
            seed=123,
            shuffle=True,
            num_replicas=None,
            rank=None,
            max_batch_repeats=None,
            is_predict: bool = False
        ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank()
        else:
            self.rank = rank
        
        self._sampler_cfg = sampler_cfg
        self.is_predict = is_predict
        if self.is_predict:
            self._sampler_cfg.max_num_res_squared = 2_000_000
            self._sampler_cfg.max_batch_size = 30
        self._data_csv = metadata_csv
        self.ligand_only = getattr(self._sampler_cfg, 'ligand_only', True)
        # num batches of each replica
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size =  self._sampler_cfg.max_batch_size
        if max_batch_repeats is not None:
            self.max_batch_repeats = max_batch_repeats
        else:
            self.max_batch_repeats = self._sampler_cfg.max_batch_repeats

        # Estimate number of batches each replica should handle in advance
        self._num_batches = self._estimate_num_batches()
        self._log.info(f'Estimated {self._num_batches} batches for {len(self._data_csv)} data points.')
        self._log.info(f'Created dataloader rank {self.rank+1} out of {self.num_replicas}')

    def _estimate_num_batches(self):
        indices = self._sample_indices()
        full_epoch_csv = self._data_csv.loc[indices]
        
        total_batches_global = 0
    
        for seq_len, len_df in full_epoch_csv.groupby('modeled_seq_len'):
            if getattr(self._sampler_cfg, 'crop', False):
                seq_len = min(seq_len, self._sampler_cfg.crop_threshold)
            if getattr(self._sampler_cfg, 'no_cipa', False):
                num_ligand_atoms = len_df['num_ligand_atoms'].max()
                seq_len += num_ligand_atoms
            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // seq_len**2 + 1,
            )
            total_batches_global += math.ceil(len(len_df) / max_batch_size)
        per_replica_batches = math.ceil(total_batches_global / self.num_replicas)
        return per_replica_batches

    def _sample_indices(self):
        if not self.ligand_only:
            # Define masks for three sample categories
            is_monomer = self._data_csv['type'] == 'monomer_no_ligand'
            is_small_mol = self._data_csv['type'] == 'protein_small_molecule_5.5'
            # Other types (neither monomer nor 5.5 small molecule)
            is_others = ~(is_monomer | is_small_mol)
            # Handle monomer_no_ligand (sample by monomer_ratio)
            monomer_df = self._data_csv[is_monomer]
            sampled_monomer = monomer_df.sample(
                frac=self._sampler_cfg.monomer_ratio, 
                random_state=self.seed + self.epoch
            )
            # Handle protein_small_molecule_5.5 (sample by small_molecule_ratio)
            small_mol_df = self._data_csv[is_small_mol]
            sampled_small_mol = small_mol_df.sample(
                frac=self._sampler_cfg.small_molecule_ratio, 
                random_state=self.seed + self.epoch
            )
            # Extract other samples that don't require sampling
            others_df = self._data_csv[is_others]
            working_df = pd.concat([sampled_monomer, sampled_small_mol, others_df], axis=0)
        
        else:
            # Only consider structures that contain ligands
            if "type" in self._data_csv:
                working_df = self._data_csv[(self._data_csv["type"] == 'protein_small_molecule_5.5') | (self._data_csv["type"] == "merged_propedia_5.5")]
            else:
                working_df = self._data_csv

        if 'cluster' in self._data_csv and (not self.is_predict):
            working_df['cluster'] = working_df['cluster'].apply(tuple)
            cluster_sample = working_df.groupby('cluster', group_keys = False).apply(
            lambda x: x.sample(n=max(1, int(len(x)*0.5)),  
                       random_state=self.seed + self.epoch),
                       include_groups=False
        )
            return cluster_sample['index'].tolist()
        else:
            return working_df['index'].tolist()
        
    def _replica_epoch_batches(self):
        # Make sure all replicas share the same seed on each epoch.
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        indices = self._sample_indices()
        if self.shuffle:
            new_order = torch.randperm(len(indices), generator=rng).numpy().tolist() # generate a random permutation from 0 to n-1
            indices = [indices[i] for i in new_order] # shuffle indices according to the random permutation

        if len(self._data_csv) > self.num_replicas:
            replica_csv = self._data_csv.iloc[indices[self.rank::self.num_replicas]]
        else:
            replica_csv = self._data_csv
        
        # Each batch contains multiple proteins of the same length.
        sample_order = []
        for seq_len, len_df in replica_csv.groupby('modeled_seq_len'):
            if getattr(self._sampler_cfg, 'crop', False):
                seq_len = min(seq_len, self._sampler_cfg.crop_threshold)
            if getattr(self._sampler_cfg, 'no_cipa', False):
                num_ligand_atoms = len_df["num_ligand_atoms"].max()
                seq_len += num_ligand_atoms
            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // seq_len**2 + 1,
            )
            num_batches = math.ceil(len(len_df) / max_batch_size)
            len_df = len_df.sort_values("num_ligand_atoms")
            for i in range(num_batches):
                batch_df = len_df.iloc[i*max_batch_size:(i+1)*max_batch_size]
                # avoid atom number order bias within the same length batch
                batch_df_shuffled = batch_df.sample(frac=1.0, random_state=self.seed + self.epoch)
                batch_indices = batch_df_shuffled['index'].tolist()
                batch_repeats = math.floor(max_batch_size / len(batch_indices))
                sample_order.append(batch_indices * min(batch_repeats, self.max_batch_repeats)) # if samples for this length are insufficient, fill to max batch size
        
        # Remove any length bias.
        if self.shuffle:
            new_order = torch.randperm(len(sample_order), generator=rng).numpy().tolist()
            return [sample_order[i] for i in new_order]
        return sample_order

    def _create_batches(self):
        # Make sure all replicas have the same number of batches Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947
        all_batches = []
        num_augments = -1
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches())
            num_augments += 1
            if num_augments > 1000:
                raise ValueError('Exceeded number of augmentations.')
        if len(all_batches) >= self._num_batches:
            all_batches = all_batches[:self._num_batches]
        self.sample_order = all_batches

    def __iter__(self):
        # Execute at the start of each epoch
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return self._num_batches
