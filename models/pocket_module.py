from typing import Any
import torch
import time
import os
import random
import numpy as np
import pandas as pd
import logging
import torch.distributed as dist
from glob import glob
from pytorch_lightning import LightningModule
from analysis import utils as au
from models.pocket_model import PepPocketNet
from models.pocket_model_ipa import PepPocketNetIPA
from models import utils as mu
from data.interpolant import Interpolant 
from data import utils as du
from data import all_atom, so3_utils
from data import residue_constants
from experiments import utils as eu
from openfold.utils.loss import compute_renamed_ground_truth
from models import loss
import gc


class PocketModule(LightningModule):

    def __init__(self, cfg, dataset_cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._dataset_cfg = dataset_cfg
        self._interpolant_cfg = cfg.interpolant
        
        no_cipa = getattr(self._model_cfg, 'no_cipa', False)
        # Set-up vector field prediction model
        if no_cipa:
            self.model = PepPocketNetIPA(cfg.model)
        else:
            self.model = PepPocketNet(cfg.model)

        # Set-up interpolant
        self.interpolant = Interpolant(cfg.interpolant)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []

        self.predict_metrics = []
        self.save_hyperparameters()

        self._checkpoint_dir = None
        self._inference_dir = None

        self.aatype_pred_num_tokens = cfg.model.aatype_pred_num_tokens

    @property
    def checkpoint_dir(self):
        if self._checkpoint_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    checkpoint_dir = [self._exp_cfg.checkpointer.dirpath]
                else:
                    checkpoint_dir = [None]
                dist.broadcast_object_list(checkpoint_dir, src=0)
                checkpoint_dir = checkpoint_dir[0]
            else:
                checkpoint_dir = self._exp_cfg.checkpointer.dirpath
            self._checkpoint_dir = checkpoint_dir
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        return self._checkpoint_dir

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self._exp_cfg.inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self._exp_cfg.inference_dir
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)
        return self._inference_dir
    
    def all_atom_loss(self, noisy_batch, model_output, sidechain_mask, diffuse_mask, pocket_mask):
        # ground truth labels for sidechain
        # The atom14 positions in the noisy batch are only modified for pocket regions,
        # so no further diffuse_mask processing is required afterwards.
        pocket_mask = pocket_mask.long()
        sidechain_mask = sidechain_mask.long()
        diffuse_mask = diffuse_mask.long()

        gt_angle = noisy_batch["torsion_angles_sin_cos"]
        gt_alt_angle = noisy_batch["alt_torsion_angles_sin_cos"]
        angle_mask = noisy_batch["torsion_angles_mask"]

        rigidgroups_gt_frames = noisy_batch["rigidgroups_gt_frames"]
        rigidgroups_alt_gt_frames = noisy_batch["rigidgroups_alt_gt_frames"]
        rigidgroups_gt_exists = noisy_batch["rigidgroups_gt_exists"]
        
        # [B, N, 14, 3]
        pred_atom14 = model_output["all_atom"]["positions"]
        # [B, N]
        pred_aatype = model_output["pred_aatypes"]
        if "renamed_atom14_gt_positions" not in model_output.keys():
            alt_results = compute_renamed_ground_truth(
                noisy_batch, # ground truth label
                model_output["all_atom"]["positions"], #[B, N, 14, 3]
            )
            renamed_atom14_gt_positions = alt_results["renamed_atom14_gt_positions"] #[B, N, 14, 3]
            renamed_atom14_gt_exists = alt_results["renamed_atom14_gt_exists"] #[B, N, 14]
            alt_naming_is_better = alt_results["alt_naming_is_better"] #[B, N]

        # Side-chain FAPE loss
        # [B,]
        side_chain_fape = loss.sidechain_loss(sidechain_frames = model_output["all_atom"]["sidechain_frames"][..., -4:,:,:].contiguous(), # [B, N, 8, 4, 4]
                                        sidechain_atom_pos = pred_atom14, # [B, N, 14, 3]
                                        rigidgroups_gt_frames=rigidgroups_gt_frames[..., -4:,:,:].contiguous(), #[B, N, 8, 4, 4]
                                        rigidgroups_alt_gt_frames = rigidgroups_alt_gt_frames[..., -4:,:,:].contiguous(), #[B, N, 8, 4, 4]
                                        rigidgroups_gt_exists = rigidgroups_gt_exists[..., -4:].contiguous(), # [B, N, 8]
                                        renamed_atom14_gt_positions = renamed_atom14_gt_positions,# [B, N, 14, 3]
                                        renamed_atom14_gt_exists = renamed_atom14_gt_exists,# [B, N, 14]
                                        alt_naming_is_better = alt_naming_is_better, # [B, N]
                                        res_mask = sidechain_mask, 
                                        ) * self._exp_cfg.training.side_chain_fape_weight
        # torison loss [B,]
        # only consider sidchain torsions
        torsion_loss = loss.torsion_angle_loss(a = model_output["all_atom"]["angles"][..., -4:, :],
                                        a_gt = gt_angle[..., -4:, :] ,
                                        a_alt_gt = gt_alt_angle[..., -4:, :] ,
                                        res_mask = sidechain_mask,
                                        a_mask = angle_mask[..., -4:] ) * self._exp_cfg.training.torsion_loss_weight
        # [B, N, 14]
        atom14_pred_mask = torch.sum(torch.abs(pred_atom14), dim=-1) > 1e-7
        violations = loss.find_structural_violations(batch = noisy_batch, 
                                                atom14_pred_positions = pred_atom14,
                                                atom14_pred_mask = atom14_pred_mask, 
                                                pred_aatype = pred_aatype,
                                                diffuse_mask= diffuse_mask,
                                                sidechain_mask= sidechain_mask,
                                                pocket_mask = pocket_mask,
                                                violation_tolerance_factor=self._exp_cfg.violation.violation_tolerance_factor,
                                                clash_overlap_tolerance=self._exp_cfg.violation.clash_overlap_tolerance)
        
        between_residue_loss, within_residue_loss, pocket_peptide_loss= loss.violation_loss(
            violations,
            atom14_atom_exists = atom14_pred_mask,
            res_mask = sidechain_mask, # Only used to count atoms when computing within_residue_loss
        )
        if "so3_t" not in noisy_batch:
            noisy_batch["so3_t"] = torch.ones_like(between_residue_loss)
            noisy_batch["cat_t"] = torch.ones_like(between_residue_loss)
        between_residue_loss = between_residue_loss * self._exp_cfg.violation.between_residue_loss_weight * noisy_batch["so3_t"].squeeze()
        within_residue_loss = within_residue_loss * self._exp_cfg.violation.within_residue_loss_weight
        pocket_peptide_loss = torch.clamp(pocket_peptide_loss * self._exp_cfg.violation.pocket_peptide_loss_weight, max = 4)
        #side_chain_fape = side_chain_fape * noisy_batch["cat_t"].squeeze()
        #torsion_loss = torsion_loss * noisy_batch["cat_t"].squeeze()

        return side_chain_fape, torsion_loss, between_residue_loss, within_residue_loss, pocket_peptide_loss

    def on_train_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            'train/epoch_time_minutes',
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )
        self._epoch_start_time = time.time()

    def model_step(self, noisy_batch: Any):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch['res_mask'] * noisy_batch['diffuse_mask']
        if "sidechain_mask" in noisy_batch:
            sidechain_mask =  noisy_batch["res_mask"] * noisy_batch["sidechain_mask"]
        else:
            sidechain_mask = loss_mask
        if "pocket_mask" in noisy_batch:
           pocket_mask = noisy_batch["pocket_mask"]
        else:
            pocket_mask = loss_mask

        loss_denom = torch.sum(loss_mask, dim=-1).clamp(min = 1.0) * 3.
        if (not self._model_cfg.predict_sidechain) and torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError('Empty batch encountered')
        num_batch, num_res = loss_mask.shape

        # Ground truth labels for bb
        gt_trans_1 = noisy_batch['trans_1']
        gt_rotmats_1 = noisy_batch['rotmats_1']
        gt_aatypes_1 = noisy_batch['aatypes_1']
        rotmats_t = noisy_batch['rotmats_t']
        gt_rot_vf = so3_utils.calc_rot_vf(
            rotmats_t, gt_rotmats_1.type(torch.float32))
        gt_bb_atoms = noisy_batch["atom37_gt_positions"][:, :, :3]
        del noisy_batch["atom37_gt_positions"]
        #gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3] 
        
        # Timestep used for normalization.
        r3_t = noisy_batch['r3_t'] # (B, 1)
        so3_t = noisy_batch['so3_t'] # (B, 1)
        cat_t = noisy_batch['cat_t'] # (B, 1)
        r3_norm_scale = 1 - torch.min(
            r3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)) # (B, 1, 1)
        so3_norm_scale = 1 - torch.min(
            so3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)) # (B, 1, 1)
        if training_cfg.aatypes_loss_use_likelihood_weighting:
            cat_norm_scale = 1 - torch.min(
                cat_t, torch.tensor(training_cfg.t_normalize_clip)) # (B, 1)
            assert cat_norm_scale.shape == (num_batch, 1)
        else:
            cat_norm_scale = 1.0
        
        # Model output predictions.
        model_output = self.model(noisy_batch)

        pred_trans_1 = model_output['pred_trans']
        pred_rotmats_1 = model_output['pred_rotmats']
        pred_logits = model_output['pred_logits'] # (B, N, aatype_pred_num_tokens)
        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        if torch.any(torch.isnan(pred_rots_vf)):
            pred_rots_vf = torch.nan_to_num(pred_rots_vf, nan=0.0)
            #raise ValueError('NaN encountered in pred_rots_vf')
        
        # aatypes loss
        aatypes_loss = loss.aa_ce_loss(pred_logits, gt_aatypes_1, cat_norm_scale, loss_mask, num_batch, training_cfg, self.aatype_pred_num_tokens)
        
        # Backbone atom loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        bb_atom_loss = loss.bb_atom_loss(gt_bb_atoms, pred_bb_atoms, loss_mask, r3_norm_scale, training_cfg)
        # Translation VF loss
        trans_loss = loss.trans_vf_loss(gt_trans_1, pred_trans_1, loss_mask, r3_norm_scale, training_cfg)
        # Rotation VF loss
        rots_vf_loss = loss.rots_vf_loss(gt_rot_vf, pred_rots_vf, loss_mask, so3_norm_scale, training_cfg)
        # Pairwise distance loss
        dist_mat_loss = loss.pairwise_distance_loss(gt_bb_atoms, pred_bb_atoms, num_batch, num_res, loss_mask)
        side_chain_fape = torsion_loss = between_residue_loss = within_residue_loss =  pocket_peptide_loss = torch.zeros_like(trans_loss)
        if self._model_cfg.predict_sidechain:
            side_chain_fape, torsion_loss, between_residue_loss, within_residue_loss, pocket_peptide_loss = self.all_atom_loss(noisy_batch, model_output, sidechain_mask, loss_mask, pocket_mask)

        aar = mu.calc_aar(pred_aa = pred_logits.detach().cpu().float().numpy(),
                          aa = gt_aatypes_1.cpu().numpy(),
                          mask = loss_mask.cpu().numpy())
        auxiliary_loss = (
            bb_atom_loss * training_cfg.aux_loss_use_bb_loss
            + dist_mat_loss * training_cfg.aux_loss_use_pair_loss
        )
        auxiliary_loss *= (
            (r3_t[:, 0] > training_cfg.aux_loss_t_pass)
            & (so3_t[:, 0] > training_cfg.aux_loss_t_pass)
        )
        auxiliary_loss *= self._exp_cfg.training.aux_loss_weight
        auxiliary_loss = torch.clamp(auxiliary_loss, max=5)

        train_loss = trans_loss + rots_vf_loss + auxiliary_loss + aatypes_loss + side_chain_fape + torsion_loss + between_residue_loss + within_residue_loss + pocket_peptide_loss
        if torch.any(torch.isnan(train_loss)):
            raise ValueError('NaN loss encountered')
        self._prev_batch = noisy_batch
        self._prev_loss_denom = loss_denom
        self._prev_loss = {
            "trans_loss": trans_loss,
            "auxiliary_loss": auxiliary_loss,
            "rots_vf_loss": rots_vf_loss,
            "train_loss": train_loss,
            'aatypes_loss': aatypes_loss,
            "aar": torch.tensor(aar),
            "side_chain_fape":side_chain_fape,
            "torsion_loss": torsion_loss,
            "between_residue_loss":between_residue_loss,
            "within_residue_loss":within_residue_loss,
            "pocket_peptide_loss":pocket_peptide_loss

        }
        return self._prev_loss

    def validation_step(self, batch: Any, batch_idx: int):
        # batch_size = 1
        res_mask = batch['res_mask']
        item_name = batch['item_name']
        ligand_object = batch["ligand_object"]
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        diffuse_mask = batch['diffuse_mask'] #[B, N]

        if "sidechain_mask" in batch:
            sidechain_mask =  batch["res_mask"] * batch["sidechain_mask"]
        else:
            sidechain_mask = diffuse_mask
        if "pocket_mask" in batch:
           pocket_mask = batch["pocket_mask"]
        else:
            pocket_mask = diffuse_mask

        ligand_mask = (batch["ligand_elements"] != 0).float()
        pocket_len = torch.sum(diffuse_mask, dim = -1) #[B,]
        condition = {
            "ligand_pos": batch["ligand_pos"],
            "ligand_elements": batch["ligand_elements"],
            "ligand_mask": ligand_mask,
            "atom_residue": batch["atom_residue"],
        }

        all_atom_dict = {}
        if self._model_cfg.predict_sidechain:
            all_atom_keys = ['rigidgroups_gt_frames', 'rigidgroups_gt_exists', 'rigidgroups_group_is_ambiguous', 'rigidgroups_group_exists', 'rigidgroups_alt_gt_frames', \
                     'residx_atom14_to_atom37', 'atom37_atom_exists', 'atom14_atom_exists', 'residx_atom37_to_atom14', \
                     'atom14_gt_exists', 'atom14_atom_is_ambiguous', 'atom14_gt_positions', 'atom14_alt_gt_positions', 'atom14_alt_gt_exists', \
                     'torsion_angles_sin_cos', 'alt_torsion_angles_sin_cos', 'torsion_angles_mask']
            for key in all_atom_keys:
                all_atom_dict[key] = batch[key]
        if "sidechain_mask" in batch:
            all_atom_dict["sidechain_mask"] = batch["res_mask"] * batch["sidechain_mask"]
        else:
            all_atom_dict["sidechain_mask"] = diffuse_mask
        prot_traj, model_traj, model_out = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
            trans_1=batch['trans_1'],
            rotmats_1=batch['rotmats_1'],
            aatypes_1=batch['aatypes_1'],
            diffuse_mask=diffuse_mask,
            chain_idx=batch['chain_idx'],
            res_idx=batch['res_idx'],
            condition = condition,
            predict_sidechain = self._model_cfg.predict_sidechain,
            sidechain_dict = all_atom_dict
        )
        # prot_traj: list[torch.tensor([B, N, 37, 3]), torch.tensor([B, N])]
        # pred atom37 [B, N, 37, 3]
        gt_atom37 = batch["atom37_gt_positions"]
        samples_tensor = prot_traj[-1][0].to(gt_atom37.device)
        # Keep original full atom37 for non-pocket regions; for pocket regions keep backbone only
        samples = gt_atom37 * (1-diffuse_mask[...,None,None]) + samples_tensor*diffuse_mask[...,None,None]
        samples = samples.cpu().numpy()
        assert samples.shape == (num_batch, num_res, 37, 3)
        # pred_aatype [B, N]
        generated_aatypes = prot_traj[-1][1].numpy() 
        assert generated_aatypes.shape == (num_batch, num_res)

        aar_metrics = mu.calc_aar(pred_aa = generated_aatypes,
                         aa = batch['aatypes_1'].cpu().numpy(),
                         mask = (diffuse_mask*res_mask).cpu().numpy())
        
        # [B, num_res, 37, 3]
        #gt_bb_atoms = all_atom.to_atom37(batch['trans_1'], batch['rotmats_1'])
        side_chain_fape, torsion_loss = None, None
        if self._model_cfg.predict_sidechain:
            side_chain_fape, torsion_loss, between_residue_loss, within_residue_loss, pocket_peptide_loss= self.all_atom_loss(batch, model_out, sidechain_mask, diffuse_mask, pocket_mask)
        del model_out
        batch_metrics = []
        for i in range(num_batch):
            length = pocket_len[i]
            residue_idx = batch['old_res_idx'][i].cpu().numpy()
            chain_idx = batch['old_chain_idx'][i].cpu().numpy()
            pocket_range = du.find_pocket_range(diffuse_mask[i], residue_idx, chain_idx)
            sample_dir = os.path.join(
                self.checkpoint_dir,
                f'sample_epoch{self.current_epoch}'
            )
            os.makedirs(sample_dir, exist_ok=True)
            
            # Write out sample to PDB file
            # atom 37 (num_res, 37, 3)
            final_pos = samples[i]
            gt_pos = gt_atom37[i]
            assert gt_pos.shape == (num_res, 37, 3)
            aatype = generated_aatypes[i]
            ligand = ligand_object[i]
            residue_index = batch["old_res_idx"][i].cpu().numpy()
            rmsd = {"rmsd" : mu.calc_rmsd(mask = np.repeat(diffuse_mask[i].cpu(),3),
                               sample_bb_pos = final_pos[:, :3].reshape(-1, 3), # only compute CA, C, N atoms
                               folded_bb_pos = gt_pos[:, :3].reshape(-1, 3),
                               )}
            
            if side_chain_fape is not None:
                rmsd["side_chain_fape"] = side_chain_fape[i].cpu().numpy()
                rmsd["torsion_loss"] = torsion_loss[i].cpu().numpy()
                rmsd["between_residue_loss"] = between_residue_loss[i].cpu().numpy()
                rmsd["within_residue_loss"] = within_residue_loss[i].cpu().numpy()
                rmsd["pocket_peptide_loss"] = pocket_peptide_loss[i].cpu().numpy()

            aar = {"aar": aar_metrics[i]}

            save_name = os.path.join(sample_dir, f'{item_name[i]}_gen_len_{length}_{pocket_range}.pdb')
            if len(save_name) > 255:
                save_name = save_name[:240] + ".pdb"
            saved_path = au.write_prot_to_pdb(
                prot_pos = final_pos,
                file_path = save_name,
                aatype = aatype,
                no_indexing=True,
                residue_index = residue_index,
                chain_index = chain_idx,
                insertion_code = batch["insertion_code"][i] if "insertion_code" in batch else None,
                ligand = ligand,
            )

            try: # metrics must be numpy types
                mdtraj_metrics = mu.calc_mdtraj_metrics(saved_path)
                ca_ca_metrics = mu.calc_ca_ca_metrics(final_pos[:, residue_constants.atom_order['CA']])
                batch_metrics.append((mdtraj_metrics | ca_ca_metrics | aar | rmsd))
            except Exception as e:
                print(e)
                continue

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)
        
    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key='valid/samples',
                columns=["sample_path", "global_step", "Protein"],
                data=self.validation_epoch_samples)
            self.validation_epoch_samples.clear()
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        for metric_name,metric_val in val_epoch_metrics.mean().to_dict().items():
            self._log_scalar(
                f'valid/{metric_name}',
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        self.validation_epoch_metrics.clear()

    def _log_scalar(
            self,
            key,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=None,
            sync_dist=False,
            rank_zero_only=True
        ):
        if sync_dist and rank_zero_only:
            raise ValueError('Unable to sync dist when rank_zero_only=True')
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only
        )
    
    def training_step(self, batch: Any, stage: int):
        if self.global_step % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        # delete unused keys
        del batch["ligand_object"]
        if "insertion_code" in batch:
            del batch["insertion_code"]
        step_start_time = time.time()
        self.interpolant.set_device(batch['res_mask'].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)
        if self._interpolant_cfg.self_condition and random.random() > 0.5:
            with torch.no_grad():
                model_sc = self.model(noisy_batch)
                noisy_batch['trans_sc'] = (
                    model_sc['pred_trans'] * noisy_batch['diffuse_mask'][..., None]
                    + noisy_batch['trans_1'] * (1 - noisy_batch['diffuse_mask'][..., None])
                )
                logits_1 = torch.nn.functional.one_hot(
                    batch['aatypes_1'].long(), num_classes=self.aatype_pred_num_tokens).float()
                noisy_batch['aatypes_sc'] = (
                    model_sc['pred_logits'] * noisy_batch['diffuse_mask'][..., None]
                    + logits_1 * (1 - noisy_batch['diffuse_mask'][..., None])
                )
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses['trans_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        # Losses to track. Stratified across t.
        so3_t = torch.squeeze(noisy_batch['so3_t'])
        self._log_scalar(
            "train/so3_t",
            np.mean(du.to_numpy(so3_t)),
            prog_bar=False, batch_size=num_batch)
        r3_t = torch.squeeze(noisy_batch['r3_t'])
        self._log_scalar(
            "train/r3_t",
            np.mean(du.to_numpy(r3_t)),
            prog_bar=False, batch_size=num_batch)
        cat_t = torch.squeeze(noisy_batch['cat_t'])
        self._log_scalar(
            "train/cat_t",
            np.mean(du.to_numpy(cat_t)),
            prog_bar=False, batch_size=num_batch)
        for loss_name, loss_dict in batch_losses.items():
            if loss_name == 'rots_vf_loss':
                batch_t = so3_t
            elif loss_name == 'train_loss':
                continue
            elif loss_name == 'aatypes_loss':
                batch_t = cat_t
            else:
                batch_t = r3_t
            stratified_losses = mu.t_stratified_loss(
                batch_t, loss_dict, loss_name=loss_name)
            for k,v in stratified_losses.items():
                self._log_scalar(
                    f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Training throughput
        self._log_scalar(
            "train/length", batch['res_mask'].shape[1], prog_bar=False, batch_size=num_batch)
        self._log_scalar(
            "train/batch_size", num_batch, prog_bar=False)
        step_time = time.time() - step_start_time
        self._log_scalar(
            "train/examples_per_second", num_batch / step_time)
        train_loss = total_losses['train_loss']
        self._log_scalar(
            "train/loss", train_loss, batch_size=num_batch)
        return train_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self._exp_cfg.optimizer
        )
    def on_predict_start(self):
        self.generated_num={}

    def predict_step(self, batch, batch_idx):
        # batch_size = 1
        # the number of sampling times is specified by config
        del batch_idx # Unused
        device = f'cuda:{torch.cuda.current_device()}'
        interpolant = Interpolant(self._infer_cfg.interpolant) 
        interpolant.set_device(device)
        sample_num = batch['res_mask'].shape[0]

        res_mask = batch['res_mask']
        item_name = batch['item_name']
        ligand_object = batch["ligand_object"]
        self.interpolant.set_device(res_mask.device)
        num_batch, sample_length = res_mask.shape

        ligand_mask = (batch["ligand_elements"] != 0).float()
        diffuse_mask = batch['diffuse_mask'] #[B, N]

        if "sidechain_mask" in batch:
            sidechain_mask =  batch["res_mask"] * batch["sidechain_mask"]
        else:
            sidechain_mask = diffuse_mask
        if "pocket_mask" in batch:
           pocket_mask = batch["pocket_mask"]
        else:
            pocket_mask = diffuse_mask

        pocket_len = torch.sum(diffuse_mask, dim = -1) #[B,]
        condition = {
            "ligand_pos": batch["ligand_pos"],
            "ligand_elements": batch["ligand_elements"],
            "ligand_mask": ligand_mask,
            "atom_residue": batch["atom_residue"],
        }

        all_atom_dict = {}
        if self._model_cfg.predict_sidechain:
            all_atom_keys = ['rigidgroups_gt_frames', 'rigidgroups_gt_exists', 'rigidgroups_group_is_ambiguous', 'rigidgroups_group_exists', 'rigidgroups_alt_gt_frames', \
                     'residx_atom14_to_atom37', 'atom37_atom_exists', 'atom14_atom_exists', 'residx_atom37_to_atom14', \
                     'atom14_gt_exists', 'atom14_atom_is_ambiguous', 'atom14_gt_positions', 'atom14_alt_gt_positions', 'atom14_alt_gt_exists', \
                     'torsion_angles_sin_cos', 'alt_torsion_angles_sin_cos', 'torsion_angles_mask']
            for key in all_atom_keys:
                all_atom_dict[key] = batch[key]
        if "sidechain_mask" in batch:
            all_atom_dict["sidechain_mask"] = batch["res_mask"] * batch["sidechain_mask"]
        else:
            all_atom_dict["sidechain_mask"] = diffuse_mask
            
        # Sample batch
        prot_traj, model_traj, model_out = interpolant.sample(
            sample_num, 
            sample_length, 
            self.model,
            trans_1 = batch['trans_1'],
            rotmats_1 = batch['rotmats_1'],
            aatypes_1 = batch['aatypes_1'],
            diffuse_mask = diffuse_mask,
            chain_idx = batch['chain_idx'],
            res_idx = batch['res_idx'],
            separate_t = self._infer_cfg.interpolant.codesign_separate_t,
            condition = condition,
            predict_sidechain = self._model_cfg.predict_sidechain,
            sidechain_dict = all_atom_dict
        )
        # prot_traj: intermediate states x_t at each step of the entire generation process
        # model_traj: model predictions at each timestep t (i.e., pred_x_1 at that t)
        write_traj = self._infer_cfg.write_traj
        # [B, num_res, 37, 3]
        gt_atom37 = batch["atom37_gt_positions"]
        
        if write_traj:
            atom37_traj = [x[0] for x in prot_traj]
            atom37_model_traj = [x[0] for x in model_traj]

            bb_trajs = du.to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1))
            noisy_traj_length = bb_trajs.shape[1]
            assert bb_trajs.shape == (num_batch, noisy_traj_length, sample_length, 37, 3)

            model_trajs = du.to_numpy(torch.stack(atom37_model_traj, dim=0).transpose(0, 1))
            clean_traj_length = model_trajs.shape[1]
            assert model_trajs.shape == (num_batch, clean_traj_length, sample_length, 37, 3)

            aa_traj = [x[1] for x in prot_traj]
            clean_aa_traj = [x[1] for x in model_traj]

            aa_trajs = du.to_numpy(torch.stack(aa_traj, dim=0).transpose(0, 1).long())
            assert aa_trajs.shape == (num_batch, noisy_traj_length, sample_length)

            for i in range(aa_trajs.shape[0]):
                for j in range(aa_trajs.shape[2]):
                    if aa_trajs[i, -1, j] == du.MASK_TOKEN_INDEX:
                        print("WARNING mask in predicted AA")
                        aa_trajs[i, -1, j] = 0
            clean_aa_trajs = du.to_numpy(torch.stack(clean_aa_traj, dim=0).transpose(0, 1).long())
            assert clean_aa_trajs.shape == (num_batch, clean_traj_length, sample_length)
            for i in range(num_batch):
                bb_traj = bb_trajs[i]
                x0_traj = model_trajs[i]
                ligand = ligand_object[i]
                residue_idx = batch['old_res_idx'][i].cpu().numpy()
                chain_idx = batch['old_chain_idx'][i].cpu().numpy()
                _ = eu.save_traj(
                    sample = bb_traj[-1],
                    bb_prot_traj = bb_traj,
                    x0_traj = x0_traj,
                    diffuse_mask =du.to_numpy(diffuse_mask)[i],
                    output_dir = os.path.join(self.inference_dir, f'{item_name[i]}_{i}_traj'),
                    aa_traj = aa_trajs[i],
                    clean_aa_traj = clean_aa_trajs[i],
                    ligand = ligand,
                    residue_idx = residue_idx,
                    chain_idx = chain_idx,
                    insertion_code = batch["insertion_code"][i] if "insertion_code" in batch else None
                )
        # [B, N, 37, 3]
        samples_tensor = prot_traj[-1][0].to(gt_atom37.device)
        # Keep original full atom37 for non-pocket regions; for pocket regions keep backbone only
        samples = gt_atom37 * (1-diffuse_mask[...,None,None]) + samples_tensor*diffuse_mask[...,None,None]
        samples = samples.cpu().numpy()
        assert samples.shape == (sample_num, sample_length, 37, 3)
        # assert False, "need to separate aatypes from atom37_traj"

        generated_aatypes = prot_traj[-1][1].numpy() #pred aatype
        assert generated_aatypes.shape == (sample_num, sample_length)

        aar_metrics = mu.calc_aar(pred_aa = generated_aatypes,
                        aa = batch['aatypes_1'].cpu().numpy(),
                        mask = (diffuse_mask*res_mask).cpu().numpy())
        side_chain_fape = None
        if self._model_cfg.predict_sidechain:
            side_chain_fape, torsion_loss, between_residue_loss, within_residue_loss, pocket_peptide_loss= self.all_atom_loss(batch, model_out, sidechain_mask, diffuse_mask, pocket_mask)
        del model_out
        batch_metrics = []
        # list[dict[str],...]
        for i in range(num_batch):
            length = pocket_len[i]
            residue_idx = batch['old_res_idx'][i].cpu().numpy()
            chain_idx = batch['old_chain_idx'][i].cpu().numpy()
            pocket_range = du.find_pocket_range(diffuse_mask[i], residue_idx, chain_idx)

            os.makedirs(self.inference_dir, exist_ok=True)
            
            # Write out sample to PDB file
            final_pos = samples[i]
            gt_pos = gt_atom37[i]

            aatype = generated_aatypes[i]
            ligand = ligand_object[i]
            rmsd = {"rmsd" : mu.calc_rmsd(mask = np.repeat(diffuse_mask[i].cpu(),3),
                            sample_bb_pos = final_pos[:, :3].reshape(-1, 3),
                            folded_bb_pos = gt_pos[:, :3].reshape(-1, 3)
                            )}
            if side_chain_fape is not None:
                rmsd["side_chain_fape"] = side_chain_fape[i].cpu().numpy()
                rmsd["torsion_loss"] = torsion_loss[i].cpu().numpy()
                rmsd["between_residue_loss"] = between_residue_loss[i].cpu().numpy()
                rmsd["within_residue_loss"] = within_residue_loss[i].cpu().numpy()
                rmsd["pocket_peptide_loss"] = pocket_peptide_loss[i].cpu().numpy()
            aar = {"aar": aar_metrics[i]}
            if item_name[i] not in self.generated_num:
                self.generated_num[item_name[i]] = 0
            else:
                self.generated_num[item_name[i]]+=1
            num = self.generated_num[item_name[i]]
            saved_path = au.write_prot_to_pdb(
                prot_pos = final_pos,
                file_path = os.path.join(self.inference_dir, f'{item_name[i]}_{num}_gen_len_{length}_{pocket_range}.pdb'),
                aatype = aatype,
                no_indexing=True,
                residue_index = residue_idx,
                chain_index = chain_idx,
                insertion_code = batch["insertion_code"][i] if "insertion_code" in batch else None,
                ligand = ligand,
            )

            try:
                mdtraj_metrics = mu.calc_mdtraj_metrics(saved_path)
                ca_ca_metrics = mu.calc_ca_ca_metrics(final_pos[:, residue_constants.atom_order['CA']])
                batch_metrics.append(({"name": f"{item_name[i]}_{num}"}| mdtraj_metrics | ca_ca_metrics | aar | rmsd))
            except Exception as e:
                print(e)
                continue

        self.predict_metrics.extend(batch_metrics)
        
    def on_predict_end(self):
        predict_result = pd.DataFrame(self.predict_metrics)
        rank = self.trainer.global_rank
        
        temp_file = os.path.join(self.inference_dir, f"metrics_temp_rank{rank}.csv")
        predict_result.to_csv(temp_file)
        print(f"rank {rank} result saved to {temp_file}")

        if self.trainer.is_global_zero:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            
            all_files = glob(os.path.join(self.inference_dir, "metrics_temp_rank*.csv"))
            combined_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
            
            final_file = os.path.join(self.inference_dir, "metrics.csv")
            print(combined_df.aar.mean())
            combined_df.to_csv(final_file, index=False)
            print(f"Final merged result saved to {final_file}")
            
            for f in all_files:
                os.remove(f)
        else:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()