import torch
from torch import nn

from models.node_feature_net import NodeFeatureNet
from models.edge_feature_net import EdgeFeatureNet, CrossEdgeFeatureNet
from models import ipa_pytorch
from models import ligand_net
from models.side_chain import SidechainModule
from data import utils as du

class PepPocketNetIPA(nn.Module):
    def __init__(self, model_conf):
        super(PepPocketNetIPA, self).__init__()
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
        self.lig_bb_ipa = getattr(self._model_conf, "lig_bb_ipa", False)
        
        self.mol_embedding_layer = ligand_net.MolEmbedder(model_conf.mol_embedder)
        self.lig_embed_proj = nn.Linear(model_conf.mol_embedder.node_embed_size, node_embed_size)
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
        num_res = node_mask.shape[1]

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
        
        # ligand node initial embeedding 
        if ligand_pos.shape[1] != 0:
            lig_edge_mask = ligand_mask[..., None] * ligand_mask[..., None, :]
            
            # [b, n_lig, ligand_embed_size]
            init_lig_node_embed, _ = self.mol_embedding_layer( 
                ligand_atom = ligand_elements,
                ligand_pos = ligand_pos,
                edge_mask = lig_edge_mask,
                atom_residue = atom_residue
            )

            lig_node_embed = init_lig_node_embed * ligand_mask[..., None]
            #  [b, n_lig, node_embed_size]
            lig_node_embed = self.lig_embed_proj(lig_node_embed) * ligand_mask[..., None]

            # Initial ligand rigids
            batch_size = ligand_pos.shape[0]
            num_atom = ligand_pos.shape[1]
            ligand_rot = torch.eye(3).expand(batch_size, num_atom, 3, 3).to(ligand_pos.device)
            
            # Concatenate protein and ligand node embeddings and rigids
            # [B, n_lig]
            ligand_res_idx = torch.arange(num_atom).unsqueeze(0).expand(batch_size, -1).to(ligand_pos.device)
            # [B, n_res + n_lig]
            protein_mask = torch.concat([node_mask, torch.zeros_like(ligand_mask)], dim=1)
            node_mask = torch.concat([node_mask, ligand_mask], dim=1)
            # [B, n_lig]
            ligand_pocket_mask = torch.zeros_like(ligand_mask)
            # [B, n_res + n_lig]
            pocket_mask = torch.concat([pocket_mask, ligand_pocket_mask], dim=1)
            diffuse_mask = torch.concat([diffuse_mask, ligand_mask], dim=1)
            # [B, n_lig]
            ligand_chain_index = torch.full_like(ligand_mask, fill_value=100)
            # [B, n_res + n_lig]
            chain_index = torch.concat([chain_index, ligand_chain_index], dim=1)
            # [B, n_res + n_lig, n_res + n_lig]
            edge_mask = node_mask[:, None] * node_mask[:, :, None]
            # [B, n_res + n_lig]
            res_index = torch.concat([res_index, ligand_res_idx], dim=1)
            # [B, n_res + n_lig, node_embed_size]
            bb_node_embed = torch.concat([bb_node_embed, lig_node_embed], dim=1)
            # [B, n_res + n_lig, 3, 3]
            rotmats_t = torch.concat([rotmats_t, ligand_rot], dim=1)
            # [B, n_res + n_lig, 3]
            trans_t = torch.concat([trans_t, ligand_pos], dim=1)
            # [B, n_res + n_lig, 3]
            trans_sc = torch.concat([trans_sc, ligand_pos], dim=1)


        # BB edge initial embedding 
        # [B, n_res + n_lig, n_res + n_lig, edge_embed_size]
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

        curr_rigids = du.create_rigid(rotmats_t, trans_t)
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)

        # Main trunk
        for b in range(self._ipa_conf.num_blocks):
            # Edge Feature between protein node and ligand node
        
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
                bb_node_embed * protein_mask[..., None])
            curr_rigids = curr_rigids.compose_q_update_vec(
                rigid_update, (protein_mask * diffuse_mask)[...,None])
            
            #update edge embedding with updated node embedding 
            if b < self._ipa_conf.num_blocks-1:
                edge_embed = self.trunk[f'edge_transition_{b}'](
                    bb_node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # nm: 1e-9
        # angstrom: 1e-10
        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()[:, :num_res, ...]
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()[:, :num_res, ...]

        #pred aatype
        # [B, N_res, 21]
        pred_logits = self.aatype_pred_net(bb_node_embed[:,:num_res, ...] * node_mask[:,:num_res, None])
        #if has masks (i.e. residue index 21)
        if self._model_conf.aatype_pred_num_tokens == du.NUM_TOKENS + 1:
            pred_logits_wo_mask = pred_logits.clone()
            pred_logits_wo_mask[:, :, du.MASK_TOKEN_INDEX] = -1e4
            pred_aatypes = torch.argmax(pred_logits_wo_mask, dim=-1)
        else:
            pred_aatypes = torch.argmax(pred_logits, dim=-1)
        
        # FILL SCAFFOLD WITH GROUND TRUTH
        pred_aatypes = pred_aatypes * diffuse_mask[:,:num_res] + aatypes_t * (1 - diffuse_mask[:,:num_res])
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
            sidechain_out = self.side_chain_net(node_embed = bb_node_embed[:,:num_res, ...] * node_mask[:,:num_res, None],
                                                pred_aa = pred_aatypes[:,:num_res],
                                                curr_rigids = curr_rigids[:, :num_res, ...],
                                                mask = node_mask[:,:num_res])
            
            # FILL SCAFFOLD WITH GROUND TRUTH
            # diffuse_mask: [B, N] -> [B, N, 1, 1]
            # Directly output atom14
            if self._model_conf.new_version:
                # diffuse_mask generates backbone structure; sidechain_mask generates sidechain structure.
                sidechain_mask = input_feats["sidechain_mask"] * input_feats["res_mask"]
                pred_mask = (sidechain_mask.long() + diffuse_mask.long()).clamp(max=1)
                sidechain_out["positions"] = sidechain_out["positions"]*pred_mask[:,:num_res, None, None] + gt_atom14*(1 - pred_mask[:,:num_res, None, None])
            out["all_atom"] = sidechain_out
        return out