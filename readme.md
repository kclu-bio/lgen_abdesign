
# An Antigen-Aware Deep Generative Model for Antibody Design via Protein–Ligand Pocket Pretraining

<p align="center">
  <img src="combined_vertical_highres_2x.gif" alt="PepPocketGen overview animation">
</p>

PepPocketGen generates antibody CDR structures and sequences for peptide,
protein, hapten, and saccharide antigens. This repository contains data preprocessing,
pretraining, antibody fine-tuning, inference, and structure-based evaluation pipelines.

All commands below assume that the current directory is the repository root unless a
different directory is shown explicitly.

## Environment Setup

```bash
conda env create -f conda_env_yamls/ppg.yml
conda activate ppg
```


```bash
python -m pip install ./torch_scatter-2.1.2+pt26cu126-cp312-cp312-linux_x86_64.whl
# Or you can download the `torch_scatter` wheel from <https://data.pyg.org/whl/> that matches your own cuda versions
python -m pip install -e .
```


## Checkpoints and Data

The examples below use the following downloadable artifacts fron zenodo:

- `LGenLig.ckpt`: Checkpoint for main model.
- `finetune_ckpt.tar.gz`: Checkpoint and configureation file for all finetuned model.
- `non_finetune_ckpt.tar.gz`: Checkpoint and configureation file for all finetuned model, including pretrained model and LGenLig w/o pretrain.
- `hapten_peptide_saccharide.tar.gz`: SAbDab hapten, peptide, and saccharide complexes.
- `protein.tar.gz`: SAbDab protein-antigen complexes, required only when protein antigens
  are included in fine-tuning.
- `testset.tar.gz`: test PDB files.
- `CCD_sdf.tar.gz`: CCD reference ligands used for hapten and saccharide relaxation.
- `propedia_pep_sequences_clu_id0.5_c0.8.clusters` and `antibody_list.pkl`: pretraining
  split/filter files.
- `antibody_ag_CDRH3_clu_id0.5_c0.8.clusters`: antibody fine-tuning cluster file.

For inference, you could only download `LGenLig.ckpt`, `config.yaml` and `testset.tar.gz`.
For training, you should download other files, and extract them under `dataset/`, and
override the corresponding paths in `configs/datasets.yaml` or `configs/finetune.yaml` when you use them.

## Inference

Inference reads `config.yaml` from the same directory as the checkpoint. Move `experiments/config.yaml` to `path/to/checkpoint/`. Keep the checkpoint and its training configuration together:

```text
path/to/checkpoint/
|-- LGenLig.ckpt
`-- config.yaml
```

If you use other checkpoints and yaml files from finetune_ckpt.tar.gz, please rename the corresponding yaml file to `config.yaml`.
Results are written beneath:

```text
experiments/inference_outputs/<project>/<run>/<checkpoint>/<task>/<timestamp>/
```

### Test set inference

Download and extract `testset.tar.gz`, put all input PDB files in one directory. The preprocessing output directory receives a
suffix based on the selected mode, for example `_imgt` or `_imgt_H3`.

Full CDR generation:

```bash
python data/pocket_parsing.py \
  -i path/to/testset/pdb \
  -o path/to/processed/testset \
  --mode imgt
```

CDR-H3-only generation:

```bash
python data/pocket_parsing.py \
  -i path/to/testset/pdb \
  -o path/to/processed/testset \
  --mode imgt_H3
```

In `imgt` mode, the shortest chain is treated as the antigen. In `imgt_H3` mode, each
filename must follow the convention expected by the parser: the antigen chain is the
second underscore-delimited field and the heavy chain is the third field.

In this process, a `metadata.csv` will be generated in your output directory (`path/to/processed/testset`).
Run inference from `experiments/` because the script and checkpoint configuration use
paths relative to this directory:

```bash
cd experiments
python inference_pockets.py \
  ++inference.predict_csv_path=../path/to/processed/testset_imgt/metadata.csv \
  ++inference.ckpt_path=../path/to/checkpoint/LGenLig.ckpt \
  ++inference.sample_num=8 \
  ++inference.task=testset
cd ..
```

### Prepare a custom antibody

Input pdb file should contain the antibody variable region.
For IMGT-numbered structures, use the predefined full-CDR ranges:

```bash
python data/pocket_parsing.py \
  -i path/to/input_pdbs \
  -o path/to/processed/custom_antibody \
  --mode imgt
```

For custom residue ranges, use `new` mode. Ranges use the syntax shown below:

```bash
python data/pocket_parsing.py \
  -i path/to/input_pdbs \
  -o path/to/processed/custom_antibody \
  --mode new \
  --pocket_idx "H:27-38,56-65,105-117;L:27-38,56-65,105-117"
```

In `new` mode, the shortest chain is treated as the antigen by default. Specify one or
more antigen chains explicitly with `--peptide_chain`; separate multiple chains with
`|`:

```bash
python data/pocket_parsing.py \
  -i path/to/input_pdbs \
  -o path/to/processed/custom_antibody \
  --mode new \
  --peptide_chain P \
  --pocket_idx "H:27-38,56-65,105-117;L:27-38,56-65,105-117"
```

Then pass the generated `metadata.csv` to `inference_pockets.py` as shown above.

### Abeta and Tau examples

After extracting the supplied `abeta.tar.gz` and `tau.tar.gz`, and put them in the directory `./dataset/design_new/`. 
Then run:

```bash
cd experiments
python inference_pockets.py \
  ++inference.predict_csv_path=../path/to/abeta/processed_docked_complex_imgt/metadata.csv \
  ++inference.ckpt_path=../path/to/checkpoint/LGenLig.ckpt \
  ++inference.sample_num=50 \
  ++inference.task=abeta

python inference_pockets.py \
  ++inference.predict_csv_path=../path/to/tau/processed_imgt/metadata.csv \
  ++inference.ckpt_path=../path/to/checkpoint/LGenLig.ckpt \
  ++inference.sample_num=1000 \
  ++inference.task=tau
cd ..
```
Errors often come from the processed_path column in metadata.csv does not match your real path. You can manually modify this column, or run `data/pocket_parsing.py` to generate metadata.csv with correct path:


```bash
cd data
python pocket_parsing.py -i ../dataset/design_new/abeta/docked_complex -o ../dataset/design_new/abeta/processed_docked_complex --mode imgt
python pocket_parsing.py -i ../dataset/design_new/tau -o ../dataset/design_new/tau/processed --mode imgt
```


## Reproduce Training

### Pretraining data

The pretraining set combines ProPedia peptide complexes and Q-BioLiP small-molecule
complexes. The initialization script is:

```bash
cd data/scripts
bash pretrain_data_initialization.sh
```

It downloads and processes both sources, writes their processed features below
`dataset/processed_training/`, and merges their metadata into
`dataset/processed_training/training_metadata.csv`
Before running it:

1. Place the resolution-list text files in `dataset/resolution/`.
2. Check the Q-BioLiP URLs because they are maintained by an external service.


```yaml
complex_dataset:
  csv_path: ../dataset/processed_training/training_metadata.csv
  cluster_path: ../dataset/propedia_pep_sequences_clu_id0.5_c0.8.clusters
  ppdbench_path: ../dataset/pdb/PPDBench/complexes113
  antibody_list_path: ../dataset/antibody_list.pkl
```
Please download file for cluster_path and antibody_list_path from zenodo, and download ppdbench data from https://webs.iiitd.edu.in/raghava/ppdbench/Complexes.tar. Put all ppdbench pdb file in  a directory and do not alter the file name.

### Pretraining

Review `configs/base.yaml`, `configs/datasets.yaml`, and `configs/model.yaml`, especially
the dataset paths, batch limits, device count, checkpoint directory, and model options.

The training script sets W&B to offline mode and calls `swanlab.sync_wandb()`. Ensure that you have a swanlab account and configure
SwanLab before launching a tracked run:

```bash
export SWANLAB_API_KEY=your_key
```

Start pretraining from `experiments/`:

```bash
cd experiments
python train_pockets.py \
  ++experiment.identifier=pretrain \
  ++experiment.num_devices=1 \
  ++experiment.finetune=false
cd ..
```

Resume an interrupted pretraining run with `warm_start`. The checkpoint directory must
also contain the original `config.yaml`:

```bash
cd experiments
python train_pockets.py \
  ++experiment.identifier=pretrain_resume \
  ++experiment.num_devices=1 \
  ++experiment.finetune=false \
  ++experiment.warm_start=path/to/checkpoint/last.ckpt
cd ..
```

### Fine-tuning data

Extract `hapten_peptide_saccharide.tar.gz` under `dataset/`. If protein antigens are
enabled with `complex_dataset.use_protein_ag: true`, extract `protein.tar.gz` as well.

Process each antigen class with the filename-aware IMGT mode:

```bash
python data/pocket_parsing.py \
  -i dataset/sabdab/Ab-pep/split_imgt_clean_fixed_fv \
  -o dataset/processed_finetune/Ab_pep_fv \
  --mode imgt_filename

python data/pocket_parsing.py \
  -i dataset/sabdab/Ab-sugar/split_imgt_clean_fixed_fv \
  -o dataset/processed_finetune/Ab_sugar_fv \
  --mode imgt_filename

python data/pocket_parsing.py \
  -i dataset/sabdab/Ab-hapten/split_imgt_clean_fixed_fv \
  -o dataset/processed_finetune/Ab_hapten_fv \
  --mode imgt_filename
```

This mode expects filenames whose second underscore-delimited field identifies the
antigen chain or chains. Concatenate the generated metadata files into one CSV without
changing the `processed_path` values. Configure `configs/finetune.yaml`:

```yaml
complex_dataset:
  csv_path: ../dataset/processed_finetune/combined_ab_metadata.csv
  fixed_test: ../path/to/testset.csv
  cluster_path: ../dataset/sabdab/antibody_ag_CDRH3_clu_id0.5_c0.8.clusters
  use_protein_ag: false
```

`fixed_test` may point to the supplied `testset.csv` in zenodo or the metadata csv produced by test-set
preprocessing. Its `item_name` column is used to remove overlapping clusters.

### Fine-tuning

Start fine-tuning from a pretrained checkpoint with `finetune_start`:

```bash
cd experiments
python train_pockets.py \
  ++experiment.identifier=antibody_finetune \
  ++experiment.num_devices=1 \
  ++experiment.finetune=true \
  ++experiment.finetune_start=path/to/pretrain/checkpoint.ckpt
cd ..
```

When `experiment.finetune=true`, `train_pockets.py` merges
`configs/finetune.yaml` into the checkpoint configuration. Use `warm_start` instead of
`finetune_start` only when resuming an already-started fine-tuning run.

## Evaluation

### Sequence recovery and RMSD

The benchmark script supports PepPocketGen (`ppg`) and several baselines:

```bash
python analysis/aar_rmsd_benchmark.py \
  --model ppg \
  --input_dir path/to/inference/run_directory
```

The current script is not fully portable: `gt_path` in
`analysis/aar_rmsd_benchmark.py` contains absolute ground-truth directories. Update
those paths before running it on another machine. `analysis/benchmark.sh` is an
experiment-specific example and also contains absolute paths.

### Relaxation and energetic scoring

Use a separate bioinformatics environment for PyRosetta, OpenMM, Open Babel, PDB2PQR,
AutoDockTools, Meeko, and Vina. A starting environment is supplied in
`bioinfo.yml`:

```bash
conda env create -f bioinfo.yml
conda activate bioinfo
conda install -c https://conda.graylab.jhu.edu -c conda-forge pyrosetta
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
python -m pip install git+https://github.com/forlilab/meeko
```

The ligand pipeline also invokes the `pdb2pqr30` executable. 
Run peptide-antigen relaxation and interface-energy calculation:

```bash
cd analysis
python relax_pipeline.py \
  --mode pep_ppg \
  --path path/to/inference/run_directory
cd ..
```

This writes relaxed structures under `peptide_standard_relax_output/` and scores to
`pep_cdrs_dg.pkl` in the inference run directory.

For hapten and saccharide antigens, extract `CCD_sdf.tar.gz`. Each SDF filename stem must
match the ligand residue name in the corresponding PDB file:

```bash
cd analysis
python relax_pipeline.py \
  --mode ligand_ppg \
  --path path/to/inference/run_directory \
  --sdf_dir path/to/CCD_sdf
cd ..
```

This writes antigen-specific relaxation directories and saves Vina scores to
`lig_cdrs_vina.pkl`. If `--sdf_dir` is omitted, the script uses its built-in legacy
default path, which is machine-specific; passing the option explicitly is recommended.

## Configuration Reference

- `configs/base.yaml`: training, optimizer, sampler, interpolant, logging, and checkpoint
  settings.
- `configs/datasets.yaml`: pretraining dataset paths and filtering.
- `configs/finetune.yaml`: antibody fine-tuning paths, validation sizes, and filters.
- `configs/model.yaml`: model architecture.
- `configs/inference_pocket.yaml`: inference sampling and output settings.

Hydra overrides use the `++section.key=value` form in the repository scripts. Relative
paths in training configs are interpreted from `experiments/` in the documented launch
commands; use absolute paths when there is any ambiguity.