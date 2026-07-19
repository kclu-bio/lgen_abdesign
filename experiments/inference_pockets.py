
import os
import time
import numpy as np
import hydra
import torch
from torch.utils.data.distributed import dist
from data.protein_dataloader import ProteinData
import pandas as pd
import GPUtil
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from omegaconf import DictConfig, OmegaConf
from experiments import utils as eu
from models.pocket_module import PocketModule
from data.complexdataset import ComplexDataset
from data.protein_dataloader import Collator
import torch.distributed as dist

torch.set_float32_matmul_precision('high')
log = eu.get_pylogger(__name__)

# abeta
# CUDA_VISIBLE_DEVICES=3 nohup python inference_pockets.py ++inference.predict_csv_path="../dataset/design_new/abeta/processed_docked_complex_imgt/metadata.csv" ++inference.ckpt_path="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-abfinetune/2026-03-22_00-27_ligand_finetune/epoch\=1403-step\=175500.ckpt" ++inference.sample_num=50 ++inference.task=abeta> inference_abeta.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python inference_pockets.py ++inference.predict_csv_path="../dataset/processed_finetune/dataset_58.csv" ++inference.ckpt_path="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-abfinetune/2026-03-18_14-01_pep_addpro_finetune/epoch\=830-step\=503586.ckpt" ++inference.sample_num=8 ++inference.task=all_CDR> inference_pep_addpro.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python inference_pockets.py ++inference.predict_csv_path="../dataset/processed_finetune/dataset_58.csv" ++inference.ckpt_path="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-abfinetune/2026-03-18_14-01_addpro_ablation/epoch\=833-step\=505404.ckpt" ++inference.sample_num=8 ++inference.task=all_CDR > inference_ablation.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python inference_pockets.py ++inference.predict_csv_path="../dataset/processed_finetune/dataset_58.csv" ++inference.ckpt_path="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-abfinetune/2026-03-22_00-27_ligand_finetune/epoch\=1403-step\=175500.ckpt" > inference_ligand_only.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python inference_pockets.py ++inference.predict_csv_path="../dataset/processed_finetune/dataset_58.csv" ++inference.ckpt_path="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-abfinetune/2026-03-24_10-08_pep_finetune_continue/epoch\=1391-step\=174000.ckpt" > inference_pep_only.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python inference_pockets.py ++inference.predict_csv_path="../dataset/processed_finetune/dataset_58.csv" ++inference.ckpt_path="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-abfinetune/2026-03-24_09-57_ligand_addpro_finetune/epoch\=896-step\=543582.ckpt" > inference_ligand_addpro.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python inference_pockets.py ++inference.predict_csv_path="../dataset/processed_finetune/dataset_58.csv" ++inference.ckpt_path="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-abfinetune/2026-03-25_15-29_ligand_finetune_nocipa/epoch\=1385-step\=500346.ckpt" > inference_nocipa.log 2>&1 &

# design_new
# CUDA_VISIBLE_DEVICES=1 nohup python inference_pockets.py ++inference.predict_csv_path="/home/kechen/peppocketgen/dataset/design_new/peg/peg_metadata.csv" ++inference.ckpt_path=\"/home/kechen/peppocketgen/experiments/ckpt/se3-fm-abfinetune/2026-01-28_04-11_peptide_only_block4/last.ckpt\" ++inference.sample_num=1 > peg_inference.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python inference_pockets.py ++inference.predict_csv_path="/home/kechen/peppocketgen/dataset/design_new/competition/complex_processed_imgt/metadata.csv" ++inference.ckpt_path="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-abfinetune/2026-02-17_00-31_full_edge_bf16_preln_ligand/epoch\=1304-step\=951364.ckpt" ++inference.sample_num=500 > competition_inference.log 2>&1 &

# H3 inference
# CUDA_VISIBLE_DEVICES=1 nohup python inference_pockets.py ++inference.predict_csv_path="/home/kechen/antibody_design/testset/0217/processed_full_imgt_H3/metadata.csv" ++inference.ckpt_path="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-abfinetune/2026-03-22_00-27_ligand_finetune/epoch\=1403-step\=175500.ckpt" ++inference.sample_num=8 ++inference.task=CDR_H3 > testset_inference_ligand.log 2>&1 &

# tau
# CUDA_VISIBLE_DEVICES=1 nohup python inference_pockets.py ++inference.predict_csv_path="../dataset/design_new/tau/processed_imgt/metadata.csv" ++inference.ckpt_path="./ckpt/se3-fm-abfinetune/2026-03-22_00-27_ligand_finetune/epoch\=1403-step\=175500.ckpt" ++inference.sample_num=1 ++inference.task=tau > testset_inference_tau1.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python inference_pockets.py ++inference.predict_csv_path="/home/kechen/peppocketgen/dataset/design_new/tau/processed_imgt/metadata.csv" ++inference.ckpt_path="./ckpt/se3-fm-abfinetune/2026-03-18_14-01_pep_addpro_finetune/epoch\=830-step\=503586.ckpt" ++inference.sample_num=1000 ++inference.task=tau > testset_inference_tau2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python inference_pockets.py ++inference.predict_csv_path="/home/kechen/peppocketgen/dataset/design_new/tau/processed_imgt/metadata.csv" ++inference.ckpt_path="./ckpt/se3-fm-abfinetune/2026-03-18_14-01_addpro_ablation/epoch\=833-step\=505404.ckpt" ++inference.sample_num=1000 ++inference.task=tau > testset_inference_tau3.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python inference_pockets.py ++inference.predict_csv_path="/home/kechen/peppocketgen/dataset/design_new/tau/processed_imgt/metadata.csv" ++inference.ckpt_path="./ckpt/se3-fm-abfinetune/2026-03-25_15-29_ligand_finetune_nocipa/epoch\=1385-step\=500346.ckpt" ++inference.sample_num=1000 ++inference.task=tau > testset_inference_tau4.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python inference_pockets.py ++inference.predict_csv_path="/home/kechen/peppocketgen/dataset/design_new/tau/processed_imgt/metadata.csv" ++inference.ckpt_path="./ckpt/se3-fm-abfinetune/2026-03-24_10-08_pep_finetune_continue/epoch\=1391-step\=174000.ckpt" ++inference.sample_num=1000 ++inference.task=tau > testset_inference_tau5.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python inference_pockets.py ++inference.predict_csv_path="/home/kechen/peppocketgen/dataset/design_new/tau/processed_imgt/metadata.csv" ++inference.ckpt_path="./ckpt/se3-fm-abfinetune/2026-03-24_09-57_ligand_addpro_finetune/epoch\=896-step\=543582.ckpt" ++inference.sample_num=1000 ++inference.task=tau > testset_inference_tau6.log 2>&1 &

class EvalRunner:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """
        ckpt_path = cfg.inference.ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))

        # Set-up config.
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        print(cfg.keys())
        cfg = OmegaConf.merge(cfg, ckpt_cfg) # ckpt_cfg has higher priority.
        cfg.experiment.checkpointer.dirpath = './'
        self._cfg = cfg
        self._exp_cfg = cfg.experiment
        self._infer_cfg = cfg.inference
        self._samples_cfg = self._infer_cfg.samples
        self._rng = np.random.default_rng(self._infer_cfg.seed)

        # Set-up output directory only on rank 0
        local_rank = os.environ.get('LOCAL_RANK', 0)
        if local_rank == 0:
            inference_dir = self.setup_inference_dir(ckpt_path)
            self._exp_cfg.inference_dir = inference_dir
            config_path = os.path.join(inference_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                OmegaConf.save(config=self._cfg, f=f)
            log.info(f'Saving inference config to {config_path}')

        # Read checkpoint and initialize module.
        if "predict_sidechain" not in self._cfg.complex_dataset:
            self._cfg.complex_dataset.predict_sidechain = False
            self._cfg.model.predict_sidechain = False

        self._flow_module = PocketModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            cfg=self._cfg,
            dataset_cfg=self._cfg.complex_dataset,
        )
        log.info(pl.utilities.model_summary.ModelSummary(self._flow_module))
        self._flow_module.eval()
        self._flow_module._infer_cfg = self._infer_cfg
        self._flow_module._samples_cfg = self._samples_cfg

    @property
    def inference_dir(self):
        return self._flow_module.inference_dir

    def setup_inference_dir(self, ckpt_path):
        self._ckpt_name = '/'.join(ckpt_path.replace('.ckpt', '').split('/')[-3:])
        output_dir = os.path.join(
            self._infer_cfg.predict_dir,
            self._ckpt_name,
            self._infer_cfg.task,
            self._infer_cfg.inference_subdir,
        )
        os.makedirs(output_dir, exist_ok=True)
        log.info(f'Saving results to {output_dir}')
        return output_dir

    def run_sampling(self):
        devices = GPUtil.getAvailable(
            order='memory', limit = 8)[:self._infer_cfg.num_gpus]
        log.info(f"Using devices: {devices}")
        log.info(f'Evaluating {self._infer_cfg.task}')
        eval_csv = pd.read_csv(self._cfg.inference.predict_csv_path)
        eval_csv_expanded = eu.duplicate_csv_rows(eval_csv, self._cfg.inference.sample_num)
        eval_csv_expanded['index'] = list(range(len(eval_csv_expanded)))
        log.info(f'expanded each item according to sample num: {self._cfg.inference.sample_num}')
        eval_dataset = ComplexDataset(eval_csv_expanded, 
                                        self._cfg.complex_dataset,
                                        is_predict = True)

        dataloader = ProteinData(
            data_cfg = self._cfg.data,
            dataset_cfg = self._cfg.complex_dataset,
            train_dataset = None,
            valid_dataset = None,
            predict_dataset = eval_dataset
        )

        trainer = Trainer(
            accelerator="gpu",
            strategy="ddp",
            devices=len(devices),
        )
        trainer.predict(self._flow_module, dataloaders=dataloader)



@hydra.main(version_base=None, config_path="../configs", config_name="inference_pocket")
# Do not reference base in Inference_pocket.
def run(cfg: DictConfig) -> None:
    # Read model checkpoint.
    log.info(f'Starting inference with {cfg.inference.num_gpus} GPUs')
    start_time = time.time()
    sampler = EvalRunner(cfg)
    sampler.run_sampling()
    
    elapsed_time = time.time() - start_time
    log.info(f'Finished in {elapsed_time:.2f}s')

if __name__ == '__main__':
    run()