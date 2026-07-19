import torch
from torch import nn
from models.node_feature_net import AllAtomNodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet, CrossEdgeFeatureNet
from models import ipa_pytorch
from models import ligand_net
from models.side_chain import SidechainModule, initialize_constants
from data import utils as du
from openfold.np import residue_constants as rc
from openfold.utils.rigid_utils import Rotation, Rigid
import math

from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)

# Residue types and backbone coordinates are fixed.

# A pretrained ligand network can be used.
# Future extensions: fine-tune backbone rigid values and peptide coordinates.

class SideChainPrediction(nn.Module):
    def __init__(self, model_conf):
        super(SideChainPrediction, self).__init__()

        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        node_embed_size = self._model_conf.node_embed_size
        edge_embed_size = self._model_conf.edge_embed_size

        self._cross_ipa_conf = model_conf.cross_ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.node_feature_net = AllAtomNodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features, embed_sc=False)
        self.mol_embedding_layer = ligand_net.MolEmbedder(model_conf.mol_embedder,
                                                          aatype_embedding_layer=None)

        self.node_fusion = nn.Sequential(
            nn.Linear(node_embed_size * 2, node_embed_size),
            nn.SiLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )
        
        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'cross_edge_feature_net_{b}'] = CrossEdgeFeatureNet(model_conf.cross_edge_features)
            self.trunk[f'bb_lig_fusion_{b}'] = ligand_net.CrossIPA(self._cross_ipa_conf)
            self.trunk[f'cipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            self.trunk[f'node_fusion_{b}'] = ligand_net.DualGLU(self._ipa_conf.c_s)
            self.trunk[f'edge_fusion_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_embed_size,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self._ipa_conf.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=self._model_conf.transformer_dropout,
                norm_first=False
            )
            self.trunk[f'seq_tfmr_{b}'] = nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)

            if b < self._ipa_conf.num_blocks-1:
                # No edge update on the last block.
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_embed_size,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )
            self.side_chain_net = SidechainModule(cfg = model_conf.sidechain_module,
                                                    aatype_embedding_layer = self.node_feature_net.aatype_embedding)
    
    def lig_feat_initialize(self, input_feats, node_mask):
        ligand_pos = input_feats['ligand_pos']
        ligand_elements = input_feats['ligand_elements']
        ligand_mask = input_feats['ligand_mask'].long() #padding mask for ligand
        # Change 1: all residues now have ligand edges, rather than pocket residues only.
        bb_lig_edge_mask = node_mask[..., None] * ligand_mask[..., None,:]
        lig_edge_mask = ligand_mask[..., None] * ligand_mask[..., None, :]
        if self._model_conf.mol_embedder.embed_residue and 'atom_residue' in input_feats:
            atom_residue = input_feats['atom_residue']
            assert atom_residue.shape == ligand_elements.shape, \
                f"atom_residue shape {atom_residue.shape} does not match ligand_elements shape {ligand_elements.shape}"
        
        # ligand node initial embeedding 
        # [b, n_lig, ligand_embed_size]
        init_lig_node_embed, _ = self.mol_embedding_layer( 
            ligand_atom = ligand_elements,
            ligand_pos = ligand_pos,
            edge_mask = lig_edge_mask,
            atom_residue = atom_residue
        )

        lig_node_embed = init_lig_node_embed * ligand_mask[..., None]

        batch_size = ligand_pos.shape[0]
        num_atom = ligand_pos.shape[1]
        ligand_rot = torch.eye(3).expand(batch_size, num_atom, 3, 3).to(ligand_pos.device)
        ligand_rigids = du.create_rigid(ligand_rot, ligand_pos)
        ligand_rigids = self.rigids_ang_to_nm(ligand_rigids)

        return ligand_pos, ligand_mask, lig_node_embed, bb_lig_edge_mask, ligand_rigids
    
    def torsion_angle_initialization(self, 
                                     torsion_angles_sin_cos, # [B, N, 7, 2]
                                     sidechain_mask,
                                     mode = "random"): 
        # [B, N, 7, 2]
        angles_corrupted = torsion_angles_sin_cos.clone()
        if mode == "all_zero":
            angles_corrupted[...,-4:, :] =  torch.tensor([0.0, 1.0], device=torsion_angles_sin_cos.device)
        elif mode == "random":
            leading_dims = torsion_angles_sin_cos.shape[:-2]
            theta = torch.rand(*leading_dims, 4, 1, device=torsion_angles_sin_cos.device) * 2 * math.pi
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            # [B, N, 4, 2]
            random_points = torch.cat([sin_theta, cos_theta], dim=-1)
            angles_corrupted[..., -4, :] = random_points
        elif mode == "rotamer":
            pass
        else:
            raise ValueError
        
        angles_corrupted = sidechain_mask[...,None, None]*angles_corrupted + (1 - sidechain_mask[...,None,None]) * torsion_angles_sin_cos
        return angles_corrupted

    def sidechain_initialization(self, 
                                 trans_1,
                                 rotmats_1,
                                 torsion_angles,
                                 aatypes,
                                 gt_atom14, #[B, N, 14]
                                 sidechain_mask):
            # [B, N, 8]
        backb_to_global = Rigid(
                Rotation(
                    rot_mats=rotmats_1, 
                    quats=None
                ),
                trans_1,
            )
        
        # initialize default_frames, groupidx, atommask, litpositions
        initialize_constants(self, torsion_angles.dtype, torsion_angles.device)

        all_frames_to_global = torsion_angles_to_frames(
            backb_to_global,
            torsion_angles,
            aatypes,
            self.default_frames
        )
        
        # [B, N, 14, 3]
        pred_xyz = frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global,
            aatypes,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions, 
        )

        
        replaced_sidechain = gt_atom14.copy()
        replaced_sidechain[...,5:,:] = pred_xyz[...,5:,:]
        initial_atom14 = sidechain_mask[...,None,None] * replaced_sidechain + (1 - sidechain_mask[...,None,None]) * gt_atom14
        return initial_atom14

    def forward(self, input_feats):
        # Reading Features
        # Attention:
        # Where sidechain_mask=1, initialize non-backbone atom14_gt_pos coordinates with all four dihedrals set to zero before model input.
        # Here diffuse_mask marks ligand-contacting residues, while sidechain_mask marks sidechains that require prediction.
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask'] #pocket mask
        chain_index = input_feats['chain_idx']
        res_index = input_feats['res_idx']
        aatypes = input_feats['aatypes_1']
        trans_1 = input_feats['trans_1']
        rotmats_1 = input_feats['rotmats_1']

        sidechain_mask = input_feats['sidechain_mask']
        atom14_gt_pos = input_feats["atom14_gt_positions"]

        # Perturb the four dihedrals of sidechains to predict, starting from an initialized conformation.
        # [B, N, 7, 2]
        angles_corrupted = self.torsion_angle_initialization(torsion_angles_sin_cos = input_feats["torsion_angles_sin_cos"],
                                                             sidechain_mask = sidechain_mask,
                                                             mode = "random")
        # [B, N, 14, 3]
        initial_atom14 = self.sidechain_initialization(self, 
                                 trans_1,
                                 rotmats_1,
                                 angles_corrupted,
                                 aatypes,
                                 atom14_gt_pos, #[B, N, 14]
                                 sidechain_mask)
        # [B, N, 14 ]
        atom14_atom_exists = input_feats["atom14_gt_exists"]
        # [B, N, 14]
        atom14_elements = rc.restype_name_to_atomic_numbers_lookup[aatypes]
        # [B, N, 14]
        atom14_res_edge_mask = atom14_atom_exists[...,:,None] * atom14_atom_exists[...,None,:]

        init_node_embed = self.node_feature_net(
            res_mask=node_mask,
            diffuse_mask=diffuse_mask,
            chain_index=chain_index,
            pos=res_index,
            aatypes=aatypes,
            atom14_gt_pos = initial_atom14 , # [B, N, 14, 3]
            atom14_elements = atom14_elements, # [B, N, 14]
            atom14_atom_exists = atom14_atom_exists, # [B, N, 14]
            atom14_res_edge_mask = atom14_res_edge_mask #[B, N, 14, 14]
        )
        bb_node_embed = init_node_embed * node_mask[..., None]

        init_edge_embed = self.edge_feature_net(
            bb_node_embed,
            trans_1,
            None,
            edge_mask,
            diffuse_mask,
            chain_index
        )

        edge_embed = init_edge_embed * edge_mask[..., None]

        # Initial bb rigids
        curr_rigids = du.create_rigid(rotmats_1, trans_1)
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)

        # initialize ligand feats
        if "ligand_pos" in input_feats:
            ligand_pos, ligand_mask, lig_node_embed, bb_lig_edge_mask, ligand_rigids = self.lig_feat_initialize(input_feats, node_mask)

        # Main trunk
        for b in range(self._ipa_conf.num_blocks):
            # ligand feat fusion
            if "ligand_pos" in input_feats:
                # Edge Feature between protein node and ligand node
                # [b, n_res, n_lig, bb_lig_edge_embed_size]
                bb_lig_edge_embed = self.trunk[f'cross_edge_feature_net_{b}'](
                    s_p = bb_node_embed,
                    s_l = lig_node_embed,
                    t_p = trans_1,
                    t_l = ligand_pos,
                    sc_t_p = None,
                    edge_mask = bb_lig_edge_mask
                )
                
                # bb-lig node embedding fusion with cross-IPA
                # employ sparse attention
                # [b, n_res, node_embed_size]
                bb_lig_rep = self.trunk[f'bb_lig_fusion_{b}'](
                    s_p = bb_node_embed,
                    s_l = lig_node_embed,
                    z = bb_lig_edge_embed,
                    r_p = curr_rigids,
                    r_l = ligand_rigids,
                    mask_p = diffuse_mask,
                    mask_l = ligand_mask,
                )
                bb_lig_rep = self.trunk[f'cipa_ln_{b}'](bb_lig_rep)
                # [b, n_res, node_embed_size]
                # bb_node_embed = bb_node_embed + gate * bb_lig_rep
                bb_node_embed = self.trunk[f'node_fusion_{b}'](bb_node_embed, bb_lig_rep)

                # edge feature fusion
                edge_embed = self.trunk[f'edge_fusion_{b}'](bb_node_embed, edge_embed)
    
            # [b, n_res, node_embed_size]
            ipa_embed = self.trunk[f'ipa_{b}'](
                bb_node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            bb_node_embed = self.trunk[f'ipa_ln_{b}'](bb_node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                bb_node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
            # src_key_padding_mask: 1 indicates padding; 0 indicates an unmasked position.
            bb_node_embed = bb_node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            bb_node_embed = self.trunk[f'node_transition_{b}'](bb_node_embed)
            bb_node_embed = bb_node_embed * node_mask[..., None]
            
            #update edge embedding with updated node embedding 
            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    bb_node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]
        # nm: 1e-9
        # angstrom: 1e-10
        curr_rigids = self.rigids_nm_to_ang(curr_rigids)

        sidechain_out = self.side_chain_net(node_embed = bb_node_embed * node_mask[..., None],
                                            pred_aa = aatypes,
                                            curr_rigids = curr_rigids,
                                            mask = node_mask)
            
        # FILL SCAFFOLD WITH GROUND TRUTH
        # diffuse_mask: [B, N] -> [B, N, 1, 1]
        # Directly output atom14 (Only modify pocket region)
        sidechain_out["positions"] = sidechain_out["positions"]*sidechain_mask[...,None,None] + atom14_gt_pos*(1 - sidechain_mask[...,None,None])
    
        return sidechain_out