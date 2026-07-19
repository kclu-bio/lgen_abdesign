import os
import torch
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from data.complexdataset import PrepareCSV, ComplexDataset
from data.protein_dataloader import ProteinData
from models.pocket_module import PocketModule
from experiments import utils as eu
import swanlab

os.environ["WANDB_MODE"] = "offline"
# ------ Pretrain ------
# CUDA_VISIBLE_DEVICES=2 python train_pockets.py ++experiment.identifier=full_edge_bf16_preln_ligand ++experiment.warm_start=/home/kechen/peppocketgen/experiments/ckpt/se3-fm-new/2026-03-19_12-17_full_edge_bf16_preln_ligand/last.ckpt ++experiment.num_devices=1 ++experiment.finetune=False > full_edge_bf16_preln_ligand_0320_continue.log 2>&1 &

# ------ finetune ------
# (1) Pep-addpro finetune
# CUDA_VISIBLE_DEVICES=1 nohup python train_pockets.py ++experiment.identifier=pep_addpro_finetune ++experiment.finetune=True ++experiment.finetune_start="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-new/2026-02-04_22-11_full_edge_bf16_preln_peptide/epoch\=509-step\=642600.ckpt" > finetune_pep_addpro_0318.log 2>&1 &
# (2) Lig-addpro finetune
# CUDA_VISIBLE_DEVICES=2 nohup python train_pockets.py ++experiment.identifier=ligand_addpro_finetune ++experiment.finetune=True ++experiment.finetune_start="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-new/2026-03-20_16-06_full_edge_bf16_preln_ligand/epoch\=398-step\=1047004.ckpt" > finetune_lig_addpro_0321.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python train_pockets.py ++experiment.identifier=ligand_addpro_finetune ++experiment.finetune=True ++experiment.warm_start="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-abfinetune/2026-03-21_01-23_pep_addpro_finetune/last.ckpt" > finetune_lig_addpro_continue.log 2>&1 &
# (3) pep -finetune (set use_protein_ag to False in finetune.yaml)
# CUDA_VISIBLE_DEVICES=3 nohup python train_pockets.py ++experiment.identifier=pep_finetune ++experiment.finetune=True ++experiment.finetune_start="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-new/2026-02-04_22-11_full_edge_bf16_preln_peptide/epoch\=509-step\=642600.ckpt" > finetune_pep_addpro_0318.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python train_pockets.py ++experiment.identifier=pep_finetune_continue ++experiment.finetune=True ++experiment.warm_start="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-abfinetune/2026-03-19_12-33_pep_finetune/last.ckpt" > finetune_pep_finetune_continue.log 2>&1 &
# (4) ligand-finetune (set use_protein_ag to False in finetune.yaml)
# CUDA_VISIBLE_DEVICES=3 nohup python train_pockets.py ++experiment.identifier=ligand_finetune ++experiment.finetune=True ++experiment.finetune_start="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-new/2026-03-20_16-06_full_edge_bf16_preln_ligand/epoch\=398-step\=1047004.ckpt" > finetune_lig_0322.log 2>&1 &
# (5) ligand finetune nocipa
# CUDA_VISIBLE_DEVICES=3 nohup python train_pockets.py ++experiment.identifier=ligand_finetune_nocipa ++experiment.finetune=True ++experiment.finetune_start="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-new/2026-02-23_22-35_nocipa_preln_ligand/epoch\=680-step\=2789376.ckpt" > finetune_lig_nocipa_0322.log 2>&1 &
# ----- ablation study ------
# no finetune (set use_protein_ag to True in finetune.yaml)
# CUDA_VISIBLE_DEVICES=0 nohup python train_pockets.py ++experiment.identifier=addpro_ablation ++experiment.finetune=True > ablation_addpro_0318.log 2>&1 &

# Solve error: RuntimeError: received 0 items of ancdata
# See: https://zhuanlan.zhihu.com/p/585186356
torch.multiprocessing.set_sharing_strategy('file_system')

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision('high')

swanlab.sync_wandb()

class Experiment:

    def __init__(self, cfg: DictConfig, ckpt_path = None, start_mode = None):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment
        self._dataset_cfg = self._setup_dataset()
        self._datamodule: LightningDataModule = ProteinData(
            data_cfg=self._data_cfg,
            dataset_cfg=self._dataset_cfg,
            train_dataset=self._train_dataset,
            valid_dataset=self._valid_dataset
        )
        self._ckpt_path = None
        self._start_mode = start_mode
        total_devices = self._exp_cfg.num_devices
        device_ids = eu.get_available_device(total_devices)
        self._train_device_ids = device_ids
        log.info(f"Training with devices: {self._train_device_ids}")
        if self._start_mode == 'finetune_start':
            log.info(f"Finetune start from checkpoint: {ckpt_path}")
            self._module = PocketModule.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                cfg=self._cfg,
                dataset_cfg=self._dataset_cfg
            )
        else:
            self._module: LightningModule = PocketModule(
                self._cfg,
                self._dataset_cfg
            )
            self._ckpt_path = ckpt_path
        if self._exp_cfg.raw_state_dict_reload is not None:
            self._module.load_state_dict(torch.load(self._exp_cfg.raw_state_dict_reload)['state_dict'])

        # Give model access to datamodule for post DDP setup processing.
        self._module._datamodule = self._datamodule

    def _setup_dataset(self):

        dataset_cfg = self._cfg.complex_dataset
        cluster_file_multichain = getattr(dataset_cfg, "cluster_file_multichain", True)
        print(cluster_file_multichain)
        if self._exp_cfg.finetune:
            csv_helper = PrepareCSV(dataset_cfg, finetune = True, cluster_file_multichain = cluster_file_multichain)
            train_csv, val_csv =  csv_helper.create_split_by_agtype()
            use_protein_ag = getattr(dataset_cfg, "use_protein_ag", False)
            if not use_protein_ag:
                train_csv = train_csv[train_csv['agtype'] != "protein"].reset_index(drop=True)
                val_csv = val_csv[val_csv['agtype'] != "protein"].reset_index(drop=True)
                train_csv["index"] = list(range(len(train_csv)))
                val_csv["index"] =  list(range(len(val_csv)))
                print(f"items after filtering out protein-ag examples for finetuning: {len(train_csv)}")
        else:
            csv_helper = PrepareCSV(dataset_cfg, finetune = False, cluster_file_multichain = cluster_file_multichain)
            train_csv, val_csv =  csv_helper.create_split()

        self._train_dataset = ComplexDataset(train_csv, dataset_cfg)
        self._valid_dataset = ComplexDataset(val_csv, dataset_cfg)
        
        return dataset_cfg
    
    def train(self):
        callbacks = []
        if self._exp_cfg.debug:
            log.info("Debug mode.")
            logger = None
            self._train_device_ids = [self._train_device_ids[0]]
            self._data_cfg.loader.num_workers = 0
        else:
            logger = WandbLogger(
                **self._exp_cfg.wandb,
            )
            
            # Model checkpoints
            callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))
            
            # Save config only for main process.
            local_rank = os.environ.get('LOCAL_RANK', 0)
            if local_rank == 0:
                ckpt_dir = self._exp_cfg.checkpointer.dirpath
                log.info(f"Checkpoints saved to {ckpt_dir}")
                os.makedirs(ckpt_dir, exist_ok=True)
                cfg_path = os.path.join(ckpt_dir, 'config.yaml')
                with open(cfg_path, 'w') as f:
                    OmegaConf.save(config=self._cfg, f=f.name)
                cfg_dict = OmegaConf.to_container(self._cfg, resolve=True)
                flat_cfg = dict(eu.flatten_dict(cfg_dict))
                if isinstance(logger.experiment.config, wandb.sdk.wandb_config.Config):
                    logger.experiment.config.update(flat_cfg)
        device_num = len(self._train_device_ids)
        if device_num ==0 :
            device_num = 1
        
        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=device_num,
            precision = "bf16"
        )
        trainer.fit(
            model=self._module,
            datamodule=self._datamodule,
            ckpt_path=self._ckpt_path # Resume training from checkpoint
        )


@hydra.main(version_base=None, config_path="../configs", config_name="base.yaml")
def main(cfg: DictConfig):
    # Start training from a checkpoint
    ckpt_path, start_mode, start_cfg = None, None, None
    if ((cfg.experiment.warm_start is not None) or (cfg.experiment.finetune_start is not None)) and cfg.experiment.warm_start_cfg_override:
        # Loads warm start config.
        # Warm start config may not have latest fields in the base config.
        # Add these fields to the warm start config.
        # Rules of OmegaConf.merge(A, B):
        # Fields in B override fields with the same name in A
        # Fields present in A but not in B remain unchanged
        # Fields present in B but not in A will be added

        # warm_start: continue training from a checkpoint
        # finetune_start: start from a checkpoint and fine-tune on the antibody dataset
        identifier = cfg.experiment.identifier
        if cfg.experiment.warm_start is not None:
            ckpt_path = cfg.experiment.warm_start
            start_mode = 'warm_start'
        elif cfg.experiment.finetune_start is not None:
            ckpt_path = cfg.experiment.finetune_start
            start_mode = 'finetune_start'
        else:
            raise ValueError("Either warm_start or finetune_start must be provided when warm_start_cfg_override is True.")
        start_cfg_path = os.path.join(
            os.path.dirname(ckpt_path), 'config.yaml')
        start_cfg = OmegaConf.load(start_cfg_path)
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(start_cfg, False)
        if cfg.experiment.finetune:
            log.info('finetune mode')
            finetune_cfg = OmegaConf.load("../configs/finetune.yaml")
            OmegaConf.set_struct(finetune_cfg, False)
            cfg = OmegaConf.merge(start_cfg, finetune_cfg)
            cfg.experiment.finetune=True
        else:
            cfg = OmegaConf.merge(cfg, start_cfg)
            OmegaConf.set_struct(cfg.model, True)
            log.info(f'Loaded warm start config from {start_cfg_path}')
        cfg.experiment.identifier = identifier
    else:
        # deigned for ablation study
        if cfg.experiment.finetune:
            identifier = cfg.experiment.identifier
            log.info('finetune mode without ckpt warm start')
            finetune_cfg = OmegaConf.load("../configs/finetune.yaml")
            OmegaConf.set_struct(cfg, False)
            OmegaConf.set_struct(finetune_cfg, False)
            finetune_cfg = OmegaConf.load("../configs/finetune.yaml")
            OmegaConf.set_struct(finetune_cfg, False)
            cfg = OmegaConf.merge(cfg, finetune_cfg)
            cfg.experiment.finetune=True
            cfg.experiment.identifier = identifier

    exp = Experiment(cfg=cfg, ckpt_path=ckpt_path, start_mode=start_mode)
    exp.train()

if __name__ == "__main__":
    main()
