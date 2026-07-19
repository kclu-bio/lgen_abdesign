$PATH1="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-abfinetune/2026-03-24_10-08_pep_finetune_continue/epoch=1391-step=174000.ckpt"
$PATH2="/home/kechen/peppocketgen/experiments/ckpt/se3-fm-abfinetune/2026-03-22_00-27_ligand_finetune/epoch=1403-step=175500.ckpt"

python relax_pipeline.py --mode ligand_ppg --path $PATH1
python relax_pipeline.py --mode pep_ppg --path $PATH1
python relax_pipeline.py --mode ligand_ppg --path $PATH2
python relax_pipeline.py --mode pep_ppg --path $PATH2 
