import torch
from torch import nn

from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet, CrossEdgeFeatureNet
from models import ipa_pytorch
from models import ligand_net
from models.side_chain import SidechainModule
from data import utils as du

class PepPocketNet(nn.Module):
    def __init__(self, model_conf):
        super(PepPocketNet, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        node_embed_size = self._model_conf.node_embed_size
        edge_embed_size = self._model_conf.edge_embed_size
        self._cross_ipa_conf = model_conf.cross_ipa
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE) 
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)
        self.full_bb_lig_edge = getattr(self._model_conf, "full_bb_lig_edge", False)
        self.pre_ln = getattr(self._model_conf, "pre_ln", False)

        if not self._model_conf.new_version:
            self.cross_edge_feature_net = CrossEdgeFeatureNet(model_conf.cross_edge_features)
            self.bb_lig_fusion = ligand_net.CrossIPA(self._cross_ipa_conf)
            self.node_fusion = nn.Sequential(
                nn.Linear(node_embed_size * 2, node_embed_size),
                nn.SiLU(),
                nn.Linear(node_embed_size, node_embed_size),
                nn.LayerNorm(node_embed_size),
            )
        
        self.mol_embedding_layer = ligand_net.MolEmbedder(model_conf.mol_embedder)

        self.aatype_pred_net = nn.Sequential(
            nn.Linear(node_embed_size, node_embed_size),
            nn.SiLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.SiLU(),
            nn.Linear(node_embed_size, self._model_conf.aatype_pred_num_tokens),
        )
        
        # Attention trunk
        self.trunk = nn.ModuleDict()
        print(self._ipa_conf.num_blocks)
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'cross_edge_feature_net_{b}'] = CrossEdgeFeatureNet(model_conf.cross_edge_features)
            self.trunk[f'bb_lig_fusion_{b}'] = ligand_net.CrossIPA(self._cross_ipa_conf)
            self.trunk[f'cipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            if self._model_conf.new_version:
                #self.trunk[f'node_fusion_{b}'] = ligand_net.DualGLU(self._ipa_conf.c_s)
                self.trunk[f'node_fusion_{b}'] = nn.Sequential(
                    nn.Linear(node_embed_size, node_embed_size),
                    nn.SiLU(),
                    nn.LayerNorm(node_embed_size),
                )

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
                norm_first=self.pre_ln
            )
            self.trunk[f'seq_tfmr_{b}'] = nn.TransformerEncoder(
                tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(
                tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(
                c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(
                self._ipa_conf.c_s, use_rot_updates=True)

            # This value is unused.
            self.trunk[f'final_block_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)

            if b < self._ipa_conf.num_blocks-1:
                # No edge update on the last block.
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self._ipa_conf.c_s,
                    edge_embed_in=edge_embed_size,
                    edge_embed_out=self._model_conf.edge_embed_size,
                )
        if model_conf.predict_sidechain:
            self.side_chain_net = SidechainModule(cfg = model_conf.sidechain_module,
                                                  aatype_embedding_layer = self.node_feature_net.aatype_embedding)
    def forward(self, input_feats):
        # input feats: noisy batch
        node_mask = input_feats['res_mask']
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        chain_index = input_feats['chain_idx']
        res_index = input_feats['res_idx']
        so3_t = input_feats['so3_t']
        r3_t = input_feats['r3_t']
        cat_t = input_feats['cat_t']
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        aatypes_t = input_feats['aatypes_t'].long()
        trans_sc = input_feats['trans_sc']
        aatypes_sc = input_feats['aatypes_sc']
        ligand_pos = input_feats['ligand_pos']
        ligand_elements = input_feats['ligand_elements']
        ligand_mask = input_feats['ligand_mask'].long() #padding mask for ligand
        atom_residue = input_feats['atom_residue']
        ligand_charge = input_feats.get('ligand_charge', None)

        if "pocket_mask" in input_feats:
            pocket_mask = input_feats["pocket_mask"]
        else:
            pocket_mask = diffuse_mask
        
        if self._model_conf.predict_sidechain:
            gt_atom14 = input_feats["atom14_gt_positions"]
            
        # Initialize node and edge embeddings
        # BB node initial embedding 
        # [B, n_res, node_embed_size]
        init_node_embed = self.node_feature_net(
            so3_t=so3_t,
            r3_t=r3_t,
            cat_t=cat_t,
            res_mask=node_mask,
            diffuse_mask=pocket_mask,
            chain_index=chain_index,
            pos=res_index,
            aatypes=aatypes_t,
            aatypes_sc=aatypes_sc,
        )
        bb_node_embed = init_node_embed * node_mask[..., None]

        # BB edge initial embedding 
        # [B, n_res, n_res, edge_embed_size]
        init_edge_embed = self.edge_feature_net(
            bb_node_embed,
            trans_t,
            trans_sc,
            res_index,
            edge_mask,
            pocket_mask,
            chain_index
        )
        edge_embed = init_edge_embed * edge_mask[..., None]

        # ligand node initial embeedding 
        # [b, n_lig, ligand_embed_size]
        if ligand_pos.shape[1] != 0:
            if self.full_bb_lig_edge:
                bb_lig_edge_mask = node_mask[..., None] * ligand_mask[:, None, :]
            else:
                bb_lig_edge_mask = pocket_mask[..., None] * ligand_mask[..., None,:]
            lig_edge_mask = ligand_mask[..., None] * ligand_mask[..., None, :]
            init_lig_node_embed, _ = self.mol_embedding_layer( #Can be graph transformer
                ligand_atom = ligand_elements,
                ligand_pos = ligand_pos,
                edge_mask = lig_edge_mask,
                atom_residue = atom_residue,
                ligand_charge = ligand_charge
            )

            lig_node_embed = init_lig_node_embed * ligand_mask[..., None]

            # Initial ligand rigids
            batch_size = ligand_pos.shape[0]
            num_atom = ligand_pos.shape[1]
            ligand_rot = torch.eye(3).expand(batch_size, num_atom, 3, 3).to(ligand_pos.device)
            ligand_rigids = du.create_rigid(ligand_rot, ligand_pos)
            ligand_rigids = self.rigids_ang_to_nm(ligand_rigids)

        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)

        # Main trunk
        for b in range(self._ipa_conf.num_blocks):
            # Edge Feature between protein node and ligand node
            # [b, n_res, n_lig, bb_lig_edge_embed_size]
            if ligand_pos.shape[1] != 0:
                bb_lig_edge_embed = self.trunk[f'cross_edge_feature_net_{b}'](
                    s_p = bb_node_embed,
                    s_l = lig_node_embed,
                    t_p = trans_t,
                    t_l = ligand_pos,
                    sc_t_p = trans_sc,
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
                    mask_p = node_mask if self.full_bb_lig_edge else pocket_mask,
                    mask_l = ligand_mask,
                )

                if self._model_conf.new_version:
                    # [b, n_res, node_embed_size]
                    if self.full_bb_lig_edge:
                        bb_node_embed = bb_lig_rep * node_mask[...,None] + bb_node_embed
                        bb_node_embed = self.trunk[f'cipa_ln_{b}'](self.trunk[f'node_fusion_{b}'](bb_node_embed) + bb_node_embed)
                    else:
                        bb_lig_rep = self.trunk[f'cipa_ln_{b}'](bb_lig_rep)
                        bb_node_embed = bb_lig_rep * pocket_mask[...,None] + bb_node_embed
                        bb_node_embed = self.trunk[f'node_fusion_{b}'](bb_node_embed)
                        bb_node_embed = bb_lig_rep * pocket_mask[...,None] + bb_node_embed
                else:
                    bb_lig_rep = self.trunk[f'cipa_ln_{b}'](bb_lig_rep)
                    # [b, n_res, node_embed_size]
                    bb_node_embed = self.trunk[f'node_fusion_{b}'](torch.cat([bb_node_embed, bb_lig_rep], 
                                                            dim = -1))
                
            # edge feature fusion
            edge_embed = self.trunk[f'edge_fusion_{b}'](bb_node_embed, edge_embed)

            ipa_embed = self.trunk[f'ipa_{b}'](
                bb_node_embed,
                edge_embed,
                curr_rigids,
                node_mask)
            ipa_embed *= node_mask[..., None]
            bb_node_embed = self.trunk[f'ipa_ln_{b}'](bb_node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](
                bb_node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
            bb_node_embed = bb_node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)

            # Contains an internal LayerNorm and residual connection.
            bb_node_embed = self.trunk[f'node_transition_{b}'](bb_node_embed)
            bb_node_embed = bb_node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](
                bb_node_embed * node_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, (node_mask * diffuse_mask)[..., None])
            
            #update edge embedding with updated node embedding 
            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    bb_node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # nm: 1e-9
        # angstrom: 1e-10
        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()

        #pred aatype
        # [B, N, 21]
        pred_logits = self.aatype_pred_net(bb_node_embed * node_mask[..., None])
        #if has masks (i.e. residue index 21)
        if self._model_conf.aatype_pred_num_tokens == du.NUM_TOKENS + 1:
            pred_logits_wo_mask = pred_logits.clone()
            pred_logits_wo_mask[:, :, du.MASK_TOKEN_INDEX] = -1e4
            pred_aatypes = torch.argmax(pred_logits_wo_mask, dim=-1)
        else:
            pred_aatypes = torch.argmax(pred_logits, dim=-1)
        
        # FILL SCAFFOLD WITH GROUND TRUTH
        pred_aatypes = pred_aatypes * diffuse_mask + aatypes_t * (1 - diffuse_mask)
        pred_aatypes = pred_aatypes.long()
        out = {
            'pred_trans': pred_trans,
            'pred_rotmats': pred_rotmats,
            'pred_logits': pred_logits,
            'pred_aatypes': pred_aatypes,
        }
        # Pred sidechain
        if self._model_conf.predict_sidechain:
            '''
                "sidechain_frames": all_frames_to_global.to_tensor_4x4(), # Eight sidechain frames [*, N, 8].
                "unnormalized_angles": unnormalized_angles,
                "angles": angles, #[*, N, 7, 2]
                "positions": pred_xyz, #[*, N, 14, 3]}
            '''
            sidechain_out = self.side_chain_net(node_embed = bb_node_embed * node_mask[..., None],
                                                pred_aa = pred_aatypes,
                                                curr_rigids = curr_rigids,
                                                mask = node_mask)
            
            # FILL SCAFFOLD WITH GROUND TRUTH
            # diffuse_mask: [B, N] -> [B, N, 1, 1]
            # Directly output atom14
            # diffuse_mask generates backbone structure; sidechain_mask generates sidechain structure.
            sidechain_mask = input_feats["sidechain_mask"] * input_feats["res_mask"]
            pred_mask = (sidechain_mask.long() + diffuse_mask.long()).clamp(max=1)
            sidechain_out["positions"] = sidechain_out["positions"]*pred_mask[...,None,None] + gt_atom14*(1 - pred_mask[...,None,None])
            out["all_atom"] = sidechain_out
        return out