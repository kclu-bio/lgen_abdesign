from typing import Any
import torch
import time
import os
import pandas as pd
import logging
import torch.distributed as dist
from glob import glob
from pytorch_lightning import LightningModule
from analysis import utils as au
from sidechain.sidechain_model import SideChainPrediction
from models import utils as mu
from data import utils as du
from openfold.utils.loss import compute_renamed_ground_truth
from models import loss
import gc
from data.all_atom import atom14_to_atom37

# TODO: 
# How to sample the sidechain mask.
# Include proteins without ligands in the training set.
class SideChainModule(LightningModule):
    def __init__(self, cfg, dataset_cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.sc_model
        self._data_cfg = cfg.data
        self._dataset_cfg = dataset_cfg

        # Set-up vector field prediction model
        self.model = SideChainPrediction(cfg.sc_model)

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []

        self.predict_metrics = []
        self.save_hyperparameters()

        self._checkpoint_dir = None
        self._inference_dir = None

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
    
    def all_atom_loss(self, noisy_batch, model_output, loss_mask):
        # ground truth labels for sidechain
        # atom14 in the noisy batch changes only the pocket, so no later diffuse_mask step is needed.
        gt_angle = noisy_batch["torsion_angles_sin_cos"]
        gt_alt_angle = noisy_batch["alt_torsion_angles_sin_cos"]
        angle_mask = noisy_batch["torsion_angles_mask"]

        rigidgroups_gt_frames = noisy_batch["rigidgroups_gt_frames"]
        rigidgroups_alt_gt_frames = noisy_batch["rigidgroups_alt_gt_frames"]
        rigidgroups_gt_exists = noisy_batch["rigidgroups_gt_exists"]
        
        # [B, N, 14, 3]
        pred_atom14 = model_output["positions"]
        # [B, N]
        pred_aatype = noisy_batch["aatypes_1"]
        if "renamed_atom14_gt_positions" not in model_output.keys():
            alt_results = compute_renamed_ground_truth(
                noisy_batch, # ground truth label
                model_output["positions"], #[B, N, 14, 3]
            )
            renamed_atom14_gt_positions = alt_results["renamed_atom14_gt_positions"] #[B, N, 14, 3]
            renamed_atom14_gt_exists = alt_results["renamed_atom14_gt_exists"] #[B, N, 14]
            alt_naming_is_better = alt_results["alt_naming_is_better"] #[B, N]

        # Side-chain FAPE loss
        # [B,]
        side_chain_fape = loss.sidechain_loss(sidechain_frames = model_output["sidechain_frames"][..., -4:,:,:].contiguous(), # [B, N, 8, 4, 4]
                                        sidechain_atom_pos = pred_atom14, # [B, N, 14, 3]
                                        rigidgroups_gt_frames=rigidgroups_gt_frames[..., -4:,:,:].contiguous(), #[B, N, 8, 4, 4]
                                        rigidgroups_alt_gt_frames = rigidgroups_alt_gt_frames[..., -4:,:,:].contiguous(), #[B, N, 8, 4, 4]
                                        rigidgroups_gt_exists = rigidgroups_gt_exists[..., -4:].contiguous(), # [B, N, 8]
                                        renamed_atom14_gt_positions = renamed_atom14_gt_positions,# [B, N, 14, 3]
                                        renamed_atom14_gt_exists = renamed_atom14_gt_exists,# [B, N, 14]
                                        alt_naming_is_better = alt_naming_is_better, # [B, N]
                                        res_mask = loss_mask, 
                                        ) * self._exp_cfg.training.side_chain_fape_weight
        # torison loss [B,]
        # only consider sidchain torsions
        torsion_loss = loss.torsion_angle_loss(a = model_output["angles"][..., -4:, :],
                                        a_gt = gt_angle[..., -4:, :] ,
                                        a_alt_gt = gt_alt_angle[..., -4:, :] ,
                                        res_mask = loss_mask,
                                        a_mask = angle_mask[..., -4:] ) * self._exp_cfg.training.torsion_loss_weight
        # [B, N, 14]
        atom14_pred_mask = torch.sum(torch.abs(pred_atom14), dim=-1) > 1e-7
        violations = loss.find_structural_violations(batch = noisy_batch, 
                                                atom14_pred_positions = pred_atom14,
                                                atom14_pred_mask = atom14_pred_mask, 
                                                pred_aatype = pred_aatype,
                                                res_mask = loss_mask,
                                                violation_tolerance_factor=self._exp_cfg.violation.violation_tolerance_factor,
                                                clash_overlap_tolerance=self._exp_cfg.violation.clash_overlap_tolerance)
        
        between_residue_loss, within_residue_loss, pocket_peptide_loss= loss.violation_loss(
            violations,
            atom14_atom_exists = atom14_pred_mask,
            res_mask = loss_mask
        )

        pocket_peptide_loss = torch.clamp(pocket_peptide_loss * self._exp_cfg.violation.pocket_peptide_loss_weight, max = 4)
        return side_chain_fape, torsion_loss, between_residue_loss, within_residue_loss, pocket_peptide_loss
    
    def load_shared_GNN(self):
        weight_path = self._model_cfg.gnn_ckpt_path
        ckpt_weight = torch.load(weight_path,
                          weights_only=False)["state_dict"]
        weight_for_load = {}
        for k, v in ckpt_weight.items():
            if "mol_embedding_layer" in k:
                weight_path[k] = v
        
        self.model.load_state_dict(weight_for_load, strict = False)
        print(f"load mol embedder gnn from {weight_path}")

    def configure_lrs(self):
        pass

    def on_train_start(self):
        self._epoch_start_time = time.time()
        if self._model_cfg.load_shared_GNN:
            self.load_shared_GNN()

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
        if "sidechain_mask" not in noisy_batch:
            noisy_batch['sidechain_mask'] = noisy_batch['diffuse_mask'].copy()
        loss_mask = noisy_batch['sidechain_mask']
        if training_cfg.mask_plddt:
            loss_mask *= noisy_batch['plddt_mask']
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError('Empty batch encountered')
        
        # Model output predictions.
        '''
        {
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(), # Eight sidechain frames [*, N, 8].
                "angles": torsion_angles, #[*, N, 7, 2]
                "positions": pred_xyz, #[*, N, 14, 3]
            }
        '''
        model_output = self.model(noisy_batch)

        #pred_atom14 = model_output['positions']        
        #pred_frames = model_output['sidechain_frames']
        #pred_angles = model_output['positions'] 
        #gt_atom14 = noisy_batch["atom14_gt_positions"]
        # dist_mat_loss = self.pairwise_distance_loss(gt_atom14, pred_atom14, num_batch, num_res, loss_mask)

        side_chain_fape, torsion_loss, between_residue_loss, within_residue_loss, pocket_peptide_loss = self.all_atom_loss(noisy_batch, model_output, loss_mask)


        train_loss = side_chain_fape + torsion_loss + between_residue_loss + within_residue_loss + pocket_peptide_loss
        if torch.any(torch.isnan(train_loss)):
            raise ValueError('NaN loss encountered')
        self._prev_batch = noisy_batch
        self._prev_loss_denom = loss_denom
        self._prev_loss = {
            "train_loss": train_loss,
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
        num_batch, num_res = res_mask.shape
        diffuse_mask = batch['diffuse_mask'] #[B, N]
        pocket_len = torch.sum(diffuse_mask, dim = -1) #[B,]
        if "sidechain_mask" not in batch:
            sidechain_mask = batch['diffuse_mask'].copy()
        else:
            sidechain_mask = batch['sidechain_mask']

        gt_atom37 = batch["atom37_gt_positions"]
        model_out = self.model(batch)
        pred_atom14, pred_frames, pred_angles = model_out["positions"], model_out["sidechain_frames"], model_out["angles"]
        assert pred_atom14.shape == (num_batch, num_res, 14, 3)

        # [B, num_res, 37, 3]
        pred_atom37, _ = atom14_to_atom37(atom14_data = pred_atom14,
                                         aatype = batch["aatypes_1"])
        final_atom37 = gt_atom37 * (1-sidechain_mask[...,None,None]) + pred_atom37*sidechain_mask[...,None,None]
        side_chain_fape, torsion_loss, between_residue_loss, within_residue_loss, pocket_peptide_loss= self.all_atom_loss(batch, model_out, diffuse_mask)
        del model_out
        batch_metrics = []

        for i in range(num_batch):
            length = pocket_len[i]
            residue_idx = batch['old_res_idx'][i].cpu().numpy()
            chain_idx = batch['chain_idx'][i].cpu().numpy()
            pocket_range = du.find_pocket_range(diffuse_mask[i], residue_idx, chain_idx)
            sample_dir = os.path.join(
                self.checkpoint_dir,
                f'sample_epoch{self.current_epoch}'
            )
            os.makedirs(sample_dir, exist_ok=True)
            
            # Write out sample to PDB file
            # atom 37 (num_res, 37, 3)
            final_pos = final_atom37[i]
            gt_pos = gt_atom37[i]
            assert gt_pos.shape == (num_res, 37, 3)
            aatype = batch["aatypes_1"][i]
            ligand = ligand_object[i]
            residue_index = batch["old_res_idx"][i].cpu().numpy()
            # TODO: 
            rmsd = {"allatom_rmsd" : mu.calc_bb_rmsd(mask = diffuse_mask[i],
                               sample_bb_pos = final_pos[:, :3].reshape(-1, 3), # Compute CA, C, and N only.
                               folded_bb_pos = gt_pos[:, :3].reshape(-1, 3),
                               )}
            if side_chain_fape is not None:
                rmsd["side_chain_fape"] = side_chain_fape[i].cpu().numpy()
                rmsd["torsion_loss"] = torsion_loss[i].cpu().numpy()
                rmsd["between_residue_loss"] = between_residue_loss[i].cpu().numpy()
                rmsd["within_residue_loss"] = within_residue_loss[i].cpu().numpy()
                rmsd["pocket_peptide_loss"] = pocket_peptide_loss[i].cpu().numpy()

            saved_path = au.write_prot_to_pdb(
                prot_pos = final_pos,
                file_path = os.path.join(sample_dir, f'{item_name[i]}_pocket_len_{length}_{pocket_range}.pdb'),
                aatype = aatype,
                no_indexing=True,
                residue_index = residue_index,
                chain_index = chain_idx,
                insertion_code = batch["insertion_code"][i] if "insertion_code" in batch else None,
                ligand = ligand,
            )

            try:
                batch_metrics.append(rmsd)
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
        if not self._model_cfg.mol_embedder.embed_residue and "atom_residue" in batch:
            del batch["atom_residue"]
        del batch["ligand_object"]
        del batch["flow_mask"]
        del batch["atom37_gt_positions"]
        if "insertion_code" in batch:
            del batch["insertion_code"]
        step_start_time = time.time()

        batch_losses = self.model_step(batch)
        num_batch = batch_losses['trans_loss'].shape[0]
        total_losses = {
            k: torch.mean(v) for k,v in batch_losses.items()
        }
        for k,v in total_losses.items():
            self._log_scalar(
                f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        
        for loss_name, loss_dict in batch_losses.items():
            self._log_scalar(
                f"train/{loss_name}", loss_dict, prog_bar=False, batch_size=num_batch)

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

    def predict_step(self, batch, batch_idx):
        # batch_size = 1
        # the number of sampling times is specified by config
        del batch_idx # Unused
        res_mask = batch['res_mask']
        item_name = batch['item_name']
        ligand_object = batch["ligand_object"]
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape

        diffuse_mask = batch['diffuse_mask'] #[B, N]
        pocket_len = torch.sum(diffuse_mask, dim = -1) #[B,]
        if "sidechain_mask" not in batch:
            sidechain_mask = batch['diffuse_mask'].copy()
        else:
            sidechain_mask = batch['sidechain_mask']

        # [B, num_res, 37, 3]
        gt_atom37 = batch["atom37_gt_positions"]
        model_out = self.model(batch)
        pred_atom14, pred_frames, pred_angles = model_out["positions"], model_out["sidechain_frames"], model_out["angles"]
        assert pred_atom14.shape == (num_batch, num_res, 14, 3)
        # [B, num_res, 37, 3]
        pred_atom37, _ = atom14_to_atom37(atom14_data = pred_atom14,
                                         aatype = batch["aatypes_1"])
        final_atom37 = gt_atom37 * (1-sidechain_mask[...,None,None]) + pred_atom37*sidechain_mask[...,None,None]
        side_chain_fape, torsion_loss, between_residue_loss, within_residue_loss, pocket_peptide_loss= self.all_atom_loss(batch, model_out, diffuse_mask)
        del model_out
        batch_metrics = []

        # list[dict[str],...]
        for i in range(num_batch):
            length = pocket_len[i]
            residue_idx = batch['old_res_idx'][i].cpu().numpy()
            chain_idx = batch['chain_idx'][i].cpu().numpy()
            pocket_range = du.find_pocket_range(diffuse_mask[i], residue_idx, chain_idx)

            os.makedirs(self.inference_dir, exist_ok=True)
            
            # Write out sample to PDB file
            final_pos = final_atom37[i]
            gt_pos = gt_atom37[i]
            aatype = batch["aatypes_1"][i]
            ligand = ligand_object[i]

            rmsd = {"allatom_rmsd" : mu.calc_bb_rmsd(mask = diffuse_mask[i],
                               sample_bb_pos = final_pos[:, :3].reshape(-1, 3), # Compute CA, C, and N only.
                               folded_bb_pos = gt_pos[:, :3].reshape(-1, 3),
                               )}
            if side_chain_fape is not None:
                rmsd["side_chain_fape"] = side_chain_fape[i].cpu().numpy()
                rmsd["torsion_loss"] = torsion_loss[i].cpu().numpy()
                rmsd["between_residue_loss"] = between_residue_loss[i].cpu().numpy()
                rmsd["within_residue_loss"] = within_residue_loss[i].cpu().numpy()
                rmsd["pocket_peptide_loss"] = pocket_peptide_loss[i].cpu().numpy()
            saved_path = au.write_prot_to_pdb(
                prot_pos = final_pos,
                file_path = os.path.join(self.inference_dir, f'{item_name[i]}_{i}_pocket_len_{length}_{pocket_range}.pdb'),
                aatype = aatype,
                no_indexing=True,
                residue_index = residue_idx,
                chain_index = chain_idx,
                insertion_code = batch["insertion_code"][i] if "insertion_code" in batch else None,
                ligand = ligand,
            )

            try:
                batch_metrics.append(({"name": f"{item_name[i]}_{i}"}| rmsd))
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