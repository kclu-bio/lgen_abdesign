from tqdm import tqdm
from pathlib import Path
from relax_protein_ligand import *
from vina_score import *
from relax_protein_peptide import *
from score_utils import get_ligand_name
import argparse
import pickle
import logging
import re

# peptide 
# CUDA_VISIBLE_DEVICES=1 nohup python relax_pipeline.py --mode pep_baseline --path /home/kechen/antibody_design/diffab/results/peptide_standard/codesign_multicdrs --save_path /home/kechen/antibody_design/testset/0217/diffab_pep_cdrs_dg.pkl --subdir_name MultipleCDRs > relax_pep_cdrs_diffab.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python relax_pipeline.py --mode pep_baseline --path /home/kechen/antibody_design/AbEgDiffuser/results/peptide_standard/codesign_single --save_path /home/kechen/antibody_design/testset/0217/abeg_pep_h3_dg.pkl --subdir_name H_CDR3 > relax_pep_h3_abeg.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python relax_pipeline.py --mode pep_abx --path /home/kechen/antibody_design/AbX/output --save_path /home/kechen/antibody_design/testset/0217/abx_pep_cdrs_dg.pkl > relax_pep_cdrs_abx.log 2>&1 &

# peptide_H3
# CUDA_VISIBLE_DEVICES=1 nohup python relax_pipeline.py --mode pep_baseline --path /home/kechen/antibody_design/diffab/results/peptide_standard/codesign_single --save_path /home/kechen/antibody_design/testset/0217/diffab_pep_h3_dg.pkl --subdir_name H_CDR3 > relax_pep_h3_diffab.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python relax_pipeline.py --mode pep_abx --path /home/kechen/antibody_design/AbX/output_H3 --save_path /home/kechen/antibody_design/testset/0217/abx_pep_h3_dg.pkl > relax_pep_h3_abx.log 2>&1 &

#ligand
# CUDA_VISIBLE_DEVICES=7 nohup python relax_pipeline.py --mode ligand_diffab --save_path /home/kechen/antibody_design/testset/0217/diffab_lig_cdrs_vina.pkl --subdir_name MultipleCDRs_with_ligand > relax_lig_cdrs_diffab.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python relax_pipeline.py --mode ligand_abeg --save_path /home/kechen/antibody_design/testset/0217/abeg_lig_h3_vina.pkl  --subdir_name H_CDR3_with_ligand > relax_lig_h3_abeg.log 2>&1 &

#ligand_H3
# CUDA_VISIBLE_DEVICES=7 nohup python relax_pipeline.py --mode ligand_diffab_h3 --save_path /home/kechen/antibody_design/testset/0217/diffab_lig_h3_vina.pkl --subdir_name H_CDR3_with_ligand > relax_lig_cdrs_diffab.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup python relax_pipeline.py --mode ligand_ppg --path /home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-02-22_11-27_full_edge_bf16_preln_peptide/epoch=1472-step=764902/pocket/run_2026-03-10_01-02-24 > relax_lig_cdrs_ppg_peponly.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python relax_pipeline.py --mode pep_ppg --path /home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-02-22_11-27_full_edge_bf16_preln_peptide/epoch=1472-step=764902/pocket/run_2026-03-10_01-02-24  > relax_pep_cdrs_ppg_peponly.log 2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup python relax_pipeline.py --mode ligand_ppg --path /home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-02-17_00-31_full_edge_bf16_preln_ligand/epoch=1304-step=951364/pocket/run_2026-03-10_01-02-24 > relax_lig_cdrs_ppg_lig.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python relax_pipeline.py --mode pep_ppg --path /home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-02-17_00-31_full_edge_bf16_preln_ligand/epoch=1304-step=951364/pocket/run_2026-03-10_01-02-24  > relax_pep_cdrs_ppg_lig.log 2>&1 &


# CUDA_VISIBLE_DEVICES=2 nohup python relax_pipeline.py --mode ligand_ppg --path /home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-02_14-20_nocipa_preln_peptide/epoch=1454-step=1846612/pocket/run_2026-03-10_01-12-58 > relax_lig_cdrs_ppg_peponly_nocipa.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python relax_pipeline.py --mode pep_ppg --path /home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-02_14-20_nocipa_preln_peptide/epoch=1454-step=1846612/pocket/run_2026-03-10_01-12-58  > relax_pep_cdrs_ppg_peponly_nocipa.log 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup python relax_pipeline.py --mode ligand_ppg --path /home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-18_14-01_addpro_ablation/epoch=833-step=505404/all_CDR/run_2026-03-24_11-06-11 > relax_abl_cdrs_ppg_ligand.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python relax_pipeline.py --mode pep_ppg --path /home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-18_14-01_addpro_ablation/epoch=833-step=505404/all_CDR/run_2026-03-24_11-06-11 > relax_abl_cdrs_ppg_ligand.log 2>&1 &

# CUDA_VISIBLE_DEVICES=7 nohup python relax_pipeline.py --mode ligand_ppg --path /home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-18_14-01_pep_addpro_finetune/epoch=830-step=503586/all_CDR/run_2026-03-24_11-06-15 > relax_pep_addpro_cdrs_ppg_peponly.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python relax_pipeline.py --mode pep_ppg --path /home/kechen/peppocketgen/experiments/inference_outputs/se3-fm-abfinetune/2026-03-18_14-01_pep_addpro_finetune/epoch=830-step=503586/all_CDR/run_2026-03-24_11-06-15 > relax_pep_addpro_ppg_peponly.log 2>&1 &
CCD_SDF_DIR = "/home/kechen/peppocketgen/dataset/pdb/CCD_sdf"

def remove_trailing_number(text):
    pattern = r'_\d+$'
    return re.sub(pattern, '', text)

item_type_mapping = {'6d01_J_G_H': 'peptide_standard',
 '8qy9_A_H_L': 'peptide_standard',
 '7rm0_Q_C_D': 'peptide_standard',
 '8f9s_P_H_L': 'peptide_standard',
 '5kzp_D_G_K': 'peptide_standard',
 '6wfz_C_A_B': 'peptide_standard',
 '4wht_k_K_L': 'peptide_standard',
 '7uym_P_H_L': 'peptide_standard',
 '8b8i_N_F_': 'peptide_standard',
 '5mp3_C_A_B': 'peptide_standard',
 '8fb5_O_A_B': 'peptide_standard',
 '3ifo_Q_A_B': 'peptide_standard',
 '4wht_u_U_V': 'peptide_standard',
 '1qkz_P_H_L': 'peptide_standard',
 '6o29_C_A_B': 'peptide_standard',
 '3mls_P_H_L': 'peptide_standard',
 '1u8i_C_B_A': 'peptide_standard',
 '6b5l_A_H_L': 'peptide_standard',
 '8che_D_A_B': 'peptide_standard',
 '8ek1_P_A_B': 'peptide_standard',
 '7sl5_C_A_B': 'peptide_standard',
 '8ux6_E_B_A': 'peptide_standard',
 '3h0t_C_B_A': 'peptide_standard',
 '4xxd_F_E_D': 'peptide_standard',
 '5ea0_P_H_L': 'peptide_standard',
 '6vi1_Q_J_I': 'peptide_standard',
 '9b0a_C_B_': 'peptide_standard',
 '1xf5_Q_D_C': 'peptide_standard',
 '5mp5_K_A_B': 'peptide_standard',
 '3idn_C_B_A': 'peptide_non_standard',
 '5ocy_C_H_L': 'peptide_non_standard',
 '8s73_N_C_D': 'peptide_non_standard',
 '5n7b_I_H_': 'peptide_non_standard',
 '1mpa_P_H_L': 'peptide_non_standard',
 '6h06_K_A_B': 'peptide_non_standard',
 '5mu0_T_G_H': 'peptide_non_standard',
 '6obd_F_H_L': 'peptide_non_standard',
 '6sf6_D_A_B': 'peptide_non_standard',
 '6xli_E_A_B': 'peptide_non_standard',
 '8us8_R_H_L': 'peptide_non_standard',
 '1yuh_X_B_A': 'hapten',
 '1y18_A_H_L': 'hapten',
 '8y57_X_A_B': 'hapten',
 '7y0g_A_H_G': 'hapten',
 '7qt0_A_E_F': 'hapten',
 '1yee_A_H_L': 'hapten',
 '1nd0_A_D_C': 'hapten',
 '1lo3_A_Y_X': 'hapten',
 '1riu_A_H_L': 'hapten',
 '7lmq_A__B': 'hapten',
 '1a6w_A_H_L': 'hapten',
 '5acm_A__B': 'hapten',
 '4hij_A_D_C': 'sugar',
 '6uuh_A_C_D': 'sugar',
 '4odv_A_H_L': 'sugar',
 '6xuk_A_H_L': 'sugar',
 '4hih_A_D_C': 'sugar',
 '3hnv_A_H_L': 'sugar'}

logger = logging.getLogger('PPRelaxer')
logging.basicConfig(
    level=logging.INFO,  # Usually set to INFO.
    format='%(asctime)s - %(levelname)s - %(message)s'
)

arg_dict = dict(
    sanitize=True,
    removeHs=True,
    split_input_complex=True,
    extract_res_mode="imgt",
    parse_with_openbabel=False,
    strictParsing=True,
    proximityBonding=True,
    cleanupSubstructures=True,
    p_restraint_type='non_H',
    p_stiffness=0.,
    l_restraint_type='non_H',
    l_stiffness=0.,
    tolerance=0.01,
    maxIterations=0,
    gpu=True,
    ccd_int=0,
    keepIds=True,
    seed=None,
    num_workers=2,
    verbose=False,
)

def vina_pipeline(out_relax_lig_file, 
                  out_relax_pdb_file, 
                  lig_pdbqt, 
                  prot_pqr, 
                  prot_pdbqt):
    lig_prep = PrepLig(out_relax_lig_file, "sdf")
    prot_prep = PrepProt(out_relax_pdb_file)

    #lig_prep.addH()
    #lig_prep.gen_conf()
    lig_prep.get_pdbqt(lig_pdbqt)

    prot_prep.addH(prot_pqr)
    prot_prep.get_pdbqt(prot_pdbqt)

    dock = VinaDock(lig_pdbqt, prot_pdbqt)
    buffer = 0
    try:
        dock.get_box(buffer=buffer)
        score = dock.dock(mode="minimize")
    except Exception as e:
        buffer = 30
        logger.error(f"Vina docking failed: {e}, retrying with larger buffer: {buffer}")
        dock.get_box(buffer=buffer)
        score = dock.dock(mode="minimize")
    return score

def relax_pl_origin(arg_dict, gt_path:list[tuple], save_path, sdf_dir=CCD_SDF_DIR):
    result = {}
    for type_dir, agtype in gt_path:
        for pdb in tqdm(os.listdir(type_dir)):
            os.makedirs(os.path.join(type_dir, "relax_output"), exist_ok=True)
            if pdb.endswith(".pdb"):
                if agtype == "hapten":
                    ligand_chain_id = os.path.basename(pdb).split('_')[1]
                    ligand_ccd = get_ligand_name(os.path.join(type_dir, pdb), ligand_chain_id)
                    ref_ligand_file = os.path.join(sdf_dir, f"{ligand_ccd}.sdf")
                    if not os.path.exists(ref_ligand_file):
                        logger.warning(f"Reference ligand file {ref_ligand_file} not found for {pdb}. Switch to OpenBabel parsing...")
                        ref_ligand_file = None
                        arg_dict["parse_with_openbabel"] = True
                    else:
                        arg_dict["parse_with_openbabel"] = False
                else:
                    logger.info(f"No reference ligand for {pdb} since it's not a hapten. Switch to OpenBabel parsing...")
                    ref_ligand_file = None
                    arg_dict["parse_with_openbabel"] = True
                    
                input_pdb_file = os.path.join(type_dir, pdb)
                out_fixed_pdb_file = os.path.join(type_dir, "relax_output", pdb.replace(".pdb", "_fixed.pdb"))
                out_relax_pdb_file = os.path.join(type_dir, "relax_output", pdb.replace(".pdb", "_relax.pdb"))
                out_relax_lig_file = os.path.join(type_dir, "relax_output", pdb.replace(".pdb", "_lig_relax.sdf"))
                out_relax_complex_file = os.path.join(type_dir, "relax_output", pdb.replace(".pdb", "_complex_relax.pdb"))
                lig_pdbqt = os.path.join(type_dir, "relax_output", pdb.replace(".pdb", "_lig.pdbqt"))
                prot_pqr = os.path.join(type_dir, "relax_output", pdb.replace(".pdb", "_prot.pqr"))
                prot_pdbqt = os.path.join(type_dir, "relax_output", pdb.replace(".pdb", "_prot.pdbqt"))
                try:
                    relax_pl(
                        input_pdb_file=input_pdb_file,
                        ref_ligand_file = ref_ligand_file,
                        out_fixed_pdb_file=out_fixed_pdb_file,
                        out_relax_pdb_file=out_relax_pdb_file,
                        out_relax_lig_file=out_relax_lig_file,
                        out_relax_complex_file=out_relax_complex_file,
                        kwargs = arg_dict
                    )

                except Exception as e:
                    logger.error(f"Relaxation failed for {pdb} with error: {e}")
                    continue

                try:
                    score = vina_pipeline(out_relax_lig_file, 
                                        out_relax_pdb_file, 
                                        lig_pdbqt, 
                                        prot_pqr, 
                                        prot_pdbqt)
                    logger.info(f"Vina score for {pdb}: {score:.3f}")
                    result[pdb] = float(score)
                
                except Exception as e:
                    logger.error(f"Vina scoring failed for {pdb} with error: {e}")
                    continue
    if save_path is not None:
        pickle.dump(result, open(save_path, "wb"))
        logger.info(f"Relaxation results saved to {save_path}")
    return result

def relax_pl_ppg(arg_dict, gen_path, save_path, sdf_dir=CCD_SDF_DIR):
    result = {}
    for pdb_file in os.listdir(gen_path):
        if pdb_file.endswith(".pdb"):
            item_basename = remove_trailing_number(pdb_file.split("_gen_len")[0])
            agtype = item_type_mapping[item_basename]
            if agtype == "peptide_standard" or agtype == "peptide_non_standard":
                logger.info(f"Skipping relaxation for {pdb_file} since it's a peptide without ligand.")
                continue
            elif agtype == "hapten":
                ligand_chain_id = os.path.basename(pdb_file).split('_')[1]
                ligand_ccd = get_ligand_name(os.path.join(gen_path, pdb_file), ligand_chain_id)
                ref_ligand_file = os.path.join(sdf_dir, f"{ligand_ccd}.sdf")
                if not os.path.exists(ref_ligand_file):
                    logger.warning(f"Reference ligand file {ref_ligand_file} not found for {pdb_file}. Switch to OpenBabel parsing...")
                    ref_ligand_file = None
                    arg_dict["parse_with_openbabel"] = True
                else:
                    arg_dict["parse_with_openbabel"] = False
            else:
                logger.info(f"No reference ligand for {pdb_file} since it's not a hapten. Switch to OpenBabel parsing...")
                ref_ligand_file = None
                arg_dict["parse_with_openbabel"] = True

            os.makedirs(os.path.join(gen_path, f"{agtype}_relax_output"), exist_ok=True)
            input_pdb_file = os.path.join(gen_path, pdb_file)
            out_fixed_pdb_file = os.path.join(gen_path, f"{agtype}_relax_output", pdb_file.replace(".pdb", "_fixed.pdb"))
            out_relax_pdb_file = os.path.join(gen_path, f"{agtype}_relax_output", pdb_file.replace(".pdb", "_relax.pdb"))
            out_relax_lig_file = os.path.join(gen_path, f"{agtype}_relax_output", pdb_file.replace(".pdb", "_lig_relax.sdf"))
            out_relax_complex_file = os.path.join(gen_path, f"{agtype}_relax_output", pdb_file.replace(".pdb", "_complex_relax.pdb"))
            lig_pdbqt = os.path.join(gen_path, f"{agtype}_relax_output", pdb_file.replace(".pdb", "_lig.pdbqt"))
            prot_pqr = os.path.join(gen_path, f"{agtype}_relax_output", pdb_file.replace(".pdb", "_prot.pqr"))
            prot_pdbqt = os.path.join(gen_path, f"{agtype}_relax_output", pdb_file.replace(".pdb", "_prot.pdbqt"))

            try:
                relax_pl(
                    input_pdb_file=input_pdb_file,
                    ref_ligand_file = ref_ligand_file,
                    out_fixed_pdb_file=out_fixed_pdb_file,
                    out_relax_pdb_file=out_relax_pdb_file,
                    out_relax_lig_file=out_relax_lig_file,
                    out_relax_complex_file=out_relax_complex_file,
                    kwargs = arg_dict
                )
            except Exception as e:
                logger.error(f"Relaxation failed for {pdb_file} with error: {e}")
                continue
            try:
                score = vina_pipeline(out_relax_lig_file, 
                                    out_relax_pdb_file, 
                                    lig_pdbqt, 
                                    prot_pqr, 
                                    prot_pdbqt)
                logger.info(f"Vina score for {pdb_file}: {score:.3f}")
                result[pdb_file] = float(score)
            except Exception as e:
                logger.error(f"Vina scoring failed for {pdb_file} with error: {e}")
                continue
    if save_path is not None:
        pickle.dump(result, open(save_path, "wb"))
        logger.info(f"Relaxation results saved to {save_path}")
    return result


def relax_pl_baseline(arg_dict, sample_num, gen_path:list[tuple], save_path:str, subdir_name:str, sdf_dir=CCD_SDF_DIR): 
    # diffab: MultipleCDRs_with_ligand
    # abeg: H_CDR3_with_ligand
    result = {}
    for type_dir, agtype in gen_path:
        for gen_dir in tqdm(os.listdir(type_dir)):
            full_gen_dir = os.path.join(type_dir, gen_dir)
            os.makedirs(os.path.join(full_gen_dir, "relax_output"), exist_ok=True)
            ligand_chain_id = gen_dir.split('_')[1]
            if agtype == "hapten":
                ligand_ccd = get_ligand_name(os.path.join(full_gen_dir, subdir_name, "0000.pdb"), ligand_chain_id)
                ref_ligand_file = os.path.join(sdf_dir, f"{ligand_ccd}.sdf")
                if not os.path.exists(ref_ligand_file):
                    logger.warning(f"Reference ligand file {ref_ligand_file} not found for {gen_dir}. Switch to OpenBabel parsing...")
                    ref_ligand_file = None
                    arg_dict["parse_with_openbabel"] = True
                else:
                    arg_dict["parse_with_openbabel"] = False
            else:
                logger.info(f"No reference ligand for {gen_dir} since it's not a hapten. Switch to OpenBabel parsing...")
                ref_ligand_file = None
                arg_dict["parse_with_openbabel"] = True

            for i in range(sample_num):
                input_pdb_file = os.path.join(full_gen_dir, subdir_name, f"{str(i).zfill(4)}.pdb")  
                out_fixed_pdb_file = os.path.join(full_gen_dir, "relax_output", f"{str(i).zfill(4)}_fixed.pdb")
                out_relax_pdb_file = os.path.join(full_gen_dir, "relax_output", f"{str(i).zfill(4)}_relax.pdb")
                out_relax_lig_file = os.path.join(full_gen_dir, "relax_output", f"{str(i).zfill(4)}_lig_relax.sdf")
                out_relax_complex_file = os.path.join(full_gen_dir, "relax_output", f"{str(i).zfill(4)}_complex_relax.pdb")
                lig_pdbqt = os.path.join(full_gen_dir, "relax_output", f"{str(i).zfill(4)}_lig.pdbqt")
                prot_pqr = os.path.join(full_gen_dir, "relax_output", f"{str(i).zfill(4)}_prot.pqr")
                prot_pdbqt = os.path.join(full_gen_dir, "relax_output", f"{str(i).zfill(4)}_prot.pdbqt")
                if not os.path.exists(out_relax_complex_file):
                    try:
                        relax_pl(
                            input_pdb_file=input_pdb_file,
                            ref_ligand_file = ref_ligand_file,
                            out_fixed_pdb_file=out_fixed_pdb_file,
                            out_relax_pdb_file=out_relax_pdb_file,
                            out_relax_lig_file=out_relax_lig_file,
                            out_relax_complex_file=out_relax_complex_file,
                            ligand_chain_id=ligand_chain_id,
                            kwargs = arg_dict
                        )
                    except Exception as e:
                        logger.error(f"Relaxation failed for {input_pdb_file} with error: {e}")
                        result[f"{gen_dir.split('.pdb')[0]}_{i}"] = e
                        continue
                    
                try:
                    score = vina_pipeline(out_relax_lig_file, 
                                        out_relax_pdb_file, 
                                        lig_pdbqt, 
                                        prot_pqr, 
                                        prot_pdbqt)
                    logger.info(f"Vina score for {gen_dir}: {score:.3f}")
                    result[f"{gen_dir.split('.pdb')[0]}_{i}"] = float(score)
                except Exception as e:
                    logger.error(f"Vina scoring failed for {gen_dir} with error: {e}")
                    result[f"{gen_dir.split('.pdb')[0]}_{i}"] = e
                    continue

    if save_path is not None:
        pickle.dump(result, open(save_path, "wb"))
        logger.info(f"Relaxation results saved to {save_path}")
    return result

def relax_pep_origin(path: str):
    result = {}
    for pdb in tqdm(os.listdir(path)):
        os.makedirs(os.path.join(path, "relax_output"), exist_ok=True)
        if pdb.endswith(".pdb"):
            out_relaxed_pdb_file = os.path.join(path, "relax_output", pdb.replace(".pdb", "_relaxed.pdb"))
            input_pdb_file = os.path.join(path, pdb)
            dg = run_openmm(input_pdb_file, out_relaxed_pdb_file, stiffness = 0, cal_dg=True)
            result[pdb] = float(dg)

def relax_pep_ppg(path: str, save_path):
    result = {}
    os.makedirs(os.path.join(path, f"peptide_standard_relax_output"), exist_ok=True)
    for pdb_file in os.listdir(path):
        if pdb_file.endswith(".pdb"):
            item_basename = remove_trailing_number(pdb_file.split("_gen_len")[0])
            agtype = item_type_mapping[item_basename]
            if agtype != "peptide_standard":
                logger.info(f"Skipping relaxation for {pdb_file} since its atigen type is not standard peptide.")
                continue

            out_relax_pdb_file = os.path.join(path, f"{agtype}_relax_output", pdb_file.replace(".pdb", "_relax.pdb"))
            input_pdb_file = os.path.join(path, pdb_file)
            try:
                dg = run_openmm(input_pdb_file, out_relax_pdb_file, stiffness = 0, cal_dg=True)
                logger.info(f"Relaxation completed for {pdb_file} with ΔG: {dg:.3f} kcal/mol")
                result[pdb_file] = float(dg)
            except Exception as e:
                logger.error(f"Relaxation failed for {pdb_file} with error: {e}")
                result[pdb_file] = e
    if save_path is not None:
        pickle.dump(result, open(save_path, "wb"))
        logger.info(f"Relaxation results saved to {save_path}")
    return result
    

def relax_pep_baseline(path:str, sample_num=8, save_path = None, subdir_name = None):
    # for diffab all cdr: subdir_name = MultipleCDRs
    # for abeg h3: subdir_name = H_CDR3
    result = {}
    for gen_dir in tqdm(os.listdir(path)):
        full_gen_dir = os.path.join(path, gen_dir)
        os.makedirs(os.path.join(full_gen_dir, "relax_output"), exist_ok=True)
        for i in tqdm(range(sample_num)):
            input_pdb_file = os.path.join(full_gen_dir, subdir_name, f"{str(i).zfill(4)}.pdb")
            out_relax_pdb_file = os.path.join(full_gen_dir, "relax_output", f"{str(i).zfill(4)}_relax.pdb")
            try:
                dg = run_openmm(input_pdb_file, out_relax_pdb_file, item_name = gen_dir, stiffness = 0, cal_dg=True)
                logger.info(f"Relaxation completed for {input_pdb_file} with ΔG: {dg:.3f} kcal/mol")
                result[f"{gen_dir}_{i}"] = float(dg)
            except Exception as e:
                logger.error(f"Relaxation failed for {input_pdb_file} with error: {e}")
                result[f"{gen_dir}_{i}"] = e
    if save_path is not None:
        pickle.dump(result, open(save_path, "wb"))
        logger.info(f"Relaxation results saved to {save_path}")
    return result

def relax_pep_abx(path:str, sample_num=8, save_path = None):
    result = {}
    for item_name in tqdm(os.listdir(path)):
     # item_name: 1qkz_H_L_P
        full_gen_dir = os.path.join(path, item_name, "design")
        H_chain = item_name.split("_")[1]
        L_chain = item_name.split("_")[2]
        ligand_chain = item_name.split("_")[3]
        original_item_name = f"{item_name.split('_')[0]}_{ligand_chain}_{H_chain}_{L_chain}"
        for i in tqdm(range(sample_num)):
            input_pdb_file = os.path.join(full_gen_dir, f"{str(i).zfill(4)}", f"{item_name}.pdb") 
            out_relax_pdb_file = os.path.join(full_gen_dir, f"{str(i).zfill(4)}", f"{item_name}_relax.pdb")
            try:
                dg = run_openmm(input_pdb_file, out_relax_pdb_file, item_name = original_item_name, stiffness = 0, cal_dg=True)
                logger.info(f"Relaxation completed for {input_pdb_file} with ΔG: {dg:.3f} kcal/mol")
                result[f"{item_name}_{i}"] = float(dg)
            except Exception as e:
                logger.error(f"Relaxation failed for {input_pdb_file} with error: {e}")
                result[f"{item_name}_{i}"] = e

    if save_path is not None:
        pickle.dump(result, open(save_path, "wb"))
        logger.info(f"Relaxation results saved to {save_path}")
    return result

def relax_new(path, save_path = None, H_chain=None, L_chain=None, ligand_chain=None):
    result = {}
    os.makedirs(os.path.join(path, "relaxed"), exist_ok=True)
    for filename in tqdm(os.listdir(path)):
        if filename.endswith(".pdb"):
            input_pdb_file = os.path.join(path, filename)
            out_relax_pdb_file = os.path.join(path, "relaxed", filename.replace(".pdb", "_relax.pdb"))
            if H_chain is not None and L_chain is not None and ligand_chain is not None:
                item_name = f"_{ligand_chain}_{H_chain}_{L_chain}"
            try:
                dg = run_openmm(input_pdb_file, out_relax_pdb_file, item_name = item_name, stiffness = 0, cal_dg=True)
                logger.info(f"Relaxation completed for {filename} with ΔG: {dg:.3f} kcal/mol")
                result[filename] = float(dg)
            except Exception as e:
                logger.error(f"Relaxation failed for {filename} with error: {e}")
                result[filename] = e
    if save_path is not None:
        pickle.dump(result, open(save_path, "wb"))
        logger.info(f"Relaxation results saved to {save_path}")
    else:
        pickle.dump(result, open(os.path.join(path, "dg_results.pkl"), "wb"))
        logger.info(f"Relaxation results saved to {os.path.join(path, 'dg_results.pkl')}")
    return result

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--mode", type=str, choices=["pep_origin", "pep_baseline", "pep_abx", "pep_ppg", "ligand_diffab", "ligand_diffab_h3", "ligand_abeg", "ligand_ppg", "new"], required=True, help="Relaxation mode to run")
    args.add_argument("--path", type=str, help="Path to the dataset for relaxation")
    args.add_argument("--sample_num",type=int, default=8, help="Number of samples to relax for each item (only applicable for baseline and abx modes)")
    args.add_argument("--save_path", type=str, default=None, help="Path to save the relaxation results (pickle file)")
    args.add_argument("--subdir_name", type=str, default=None, help="Subdirectory name where the pdb files are located (only applicable for baseline mode)")
    args.add_argument("--H_chain", type=str, default=None, help="Heavy chain identifier (only applicable for new mode)")
    args.add_argument("--L_chain", type=str, default=None, help="Light chain identifier (only applicable for new mode)")
    args.add_argument("--ligand_chain", type=str, default=None, help="Ligand chain identifier (only applicable for new mode)")
    args.add_argument("--sdf_dir", type=str, default=CCD_SDF_DIR, help="Directory containing ligand SDF files")

    args = args.parse_args()
    
    paths_diffab = [("/home/kechen/antibody_design/diffab/results/hapten_non_ligand/codesign_multicdrs", "hapten"), 
            ("/home/kechen/antibody_design/diffab/results/sugar_non_ligand/codesign_multicdrs", "sugar")]
    paths_diffab_h3 = [("/home/kechen/antibody_design/diffab/results/hapten_non_ligand/codesign_single", "hapten"), 
            ("/home/kechen/antibody_design/diffab/results/sugar_non_ligand/codesign_single", "sugar")]
    paths_abeg = [("/home/kechen/antibody_design/AbEgDiffuser/results/hapten_non_ligand/codesign_single", "hapten"), 
            ("/home/kechen/antibody_design/AbEgDiffuser/results/sugar_non_ligand/codesign_single", "sugar")]
    
    if args.mode == "pep_origin":
        assert args.path is not None, "Path to the original peptide dataset must be provided for pep_origin mode"
        relax_pep_origin(args.path)
    elif args.mode == "pep_baseline":
        assert args.path is not None and args.subdir_name is not None, "Path to the generated peptide dataset and subdirectory name must be provided for pep_baseline mode"
        relax_pep_baseline(args.path, args.sample_num, args.save_path, args.subdir_name)
    elif args.mode == "pep_abx":
        assert args.path is not None, "Path to the AbX generated dataset must be provided for pep_abx mode"
        relax_pep_abx(args.path, args.sample_num, args.save_path)
    elif args.mode == "ligand_diffab":
        relax_pl_baseline(arg_dict, args.sample_num, paths_diffab, args.save_path, args.subdir_name, args.sdf_dir)
    elif args.mode == "ligand_diffab_h3":
        relax_pl_baseline(arg_dict, args.sample_num, paths_diffab_h3, args.save_path, args.subdir_name, args.sdf_dir)
    elif args.mode == "ligand_abeg":
        relax_pl_baseline(arg_dict, args.sample_num, paths_abeg, args.save_path, args.subdir_name, args.sdf_dir)
    elif args.mode == "pep_ppg":
        assert args.path is not None, "Path to the PPG generated dataset must be provided for pep_ppg mode"
        save_path = os.path.join(args.path, "pep_cdrs_dg.pkl")
        result = relax_pep_ppg(args.path, save_path)
        print(result)
    elif args.mode == "ligand_ppg":
        assert args.path is not None, "Path to the PPG generated dataset must be provided for ligand_ppg mode"
        save_path = os.path.join(args.path, "lig_cdrs_vina.pkl")
        result = relax_pl_ppg(arg_dict, args.path, save_path, args.sdf_dir)
        print(result)
    elif args.mode == "new":
        assert args.path is not None, "Path to the dataset must be provided for new mode"
        assert args.H_chain is not None and args.L_chain is not None and args.ligand_chain is not None, "Heavy chain, light chain, and ligand chain identifiers must be provided for new mode"
        relax_new(args.path, args.save_path, args.H_chain, args.L_chain, args.ligand_chain)