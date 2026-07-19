
PATH1="/home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-24_10-08_pep_finetune_continue/epoch=1391-step=174000/pocket/run_2026-03-25_13-44-24"
PATH2="/home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-22_00-27_ligand_finetune/epoch=1403-step=175500/pocket/run_2026-03-25_13-44-24"
PATH3="/home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-18_14-01_pep_addpro_finetune/epoch=830-step=503586/all_CDR/run_2026-03-24_11-06-15"
PATH4="/home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-18_14-01_addpro_ablation/epoch=833-step=505404/all_CDR/run_2026-03-24_11-06-11"
PATH5="/home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-24_09-57_ligand_addpro_finetune/epoch=764-step=463590/pocket/run_2026-03-26_15-21-09"
PATH_CDR="/home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-22_00-27_ligand_finetune/epoch=1403-step=175500/CDR_H3/run_2026-04-03_12-25-26"
nocipa_PATH="/home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-25_15-29_ligand_finetune_nocipa/epoch=1385-step=500346/pocket/run_2026-04-07_13-25-42"

#python aar_rmsd_benchmark.py --model ppg --input_dir $PATH1
#python aar_rmsd_benchmark.py --input_dir $PATH2
#python aar_rmsd_benchmark.py --model ppg --input_dir $PATH3
#python aar_rmsd_benchmark.py --model ppg --input_dir $PATH4
python aar_rmsd_benchmark.py --model ppg --input_dir $PATH5
python aar_rmsd_benchmark.py --model ppg --input_dir $PATH_CDR
python aar_rmsd_benchmark.py --model ppg --input_dir $nocipa_PATH

#python aar_rmsd_benchmark.py --model diffab --mode h3 --input_dir /home/kechen/antibody_design/diffab/results
#python aar_rmsd_benchmark.py --model diffab --mode cdrs --input_dir /home/kechen/antibody_design/diffab/results
#python aar_rmsd_benchmark.py --model abeg --input_dir /home/kechen/antibody_design/AbEgDiffuser/results