import torch
import torch.nn as nn

import torch
import math
import torch.nn as nn
from typing import Optional, Callable, List, Sequence
from openfold.utils.rigid_utils import Rigid
from models.ipa_pytorch import Linear, ipa_point_weights_init_, permute_final_dims, flatten_final_dims

# 3D-GNN
# only used for full atom representation for ligand
class MolEmbedder(nn.Module):
    def __init__(self, model_conf):
        super(MolEmbedder, self).__init__()
        torch.set_default_dtype(torch.float32)
        self._model_conf = model_conf
        self._charge_conf = getattr(model_conf, 'charge', None)
        self._distance_threshold = self._model_conf.distance_threshold
        node_embed_dims = self._model_conf.num_atom_type
        node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Embedding(node_embed_dims, node_embed_size, padding_idx=0),
            nn.SiLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        self.residue_embedder = nn.Sequential(
            nn.Embedding(self._model_conf.aatype_embed_num_tokens + 1, node_embed_size),
            nn.SiLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )
        # Fuse atom and residue embed for ligand (peptide and small molecule)    
        self.feature_fusion = DualGLU(node_embed_size)

        if self._charge_conf is not None and self._charge_conf.embed_charge:
            self.atom_feature_fusion = nn.Linear(node_embed_size + self._charge_conf.num_rbf_size, node_embed_size)
            
        self.node_aggregator = nn.Sequential(
            nn.Linear(node_embed_size + self._model_conf.edge_embed_size, node_embed_size),
            nn.SiLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.SiLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        self.dist_min = self._model_conf.ligand_rbf_d_min
        self.dist_max = self._model_conf.ligand_rbf_d_max
        self.num_rbf_size = self._model_conf.num_rbf_size
        self.edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(self.num_rbf_size + node_embed_size + node_embed_size, self.edge_embed_size),
            nn.SiLU(),
            nn.Linear(self._model_conf.edge_embed_size, self._model_conf.edge_embed_size),
            nn.SiLU(),
            nn.Linear(self._model_conf.edge_embed_size, self._model_conf.edge_embed_size),
            nn.LayerNorm(self._model_conf.edge_embed_size),
        )

        mu = torch.linspace(self.dist_min, self.dist_max, self.num_rbf_size)
        self.mu = mu.reshape([1, 1, 1, -1])
        self.sigma = (self.dist_max - self.dist_min) / self.num_rbf_size

    def coord2dist(self, coord, edge_mask):
        n_batch, n_atom = coord.size(0), coord.size(1)
        radial = torch.sum((coord.unsqueeze(1) - coord.unsqueeze(2)) ** 2, dim=-1) # squared distances
        dist = torch.sqrt(
                radial + 1e-6
            ) * edge_mask

        radial = radial * edge_mask
        return radial, dist
    
    def rbf(self, dist):
        dist_expand = torch.unsqueeze(dist, -1) #[B, N, N, 1]
        _mu = self.mu.to(dist.device) #[1, 1, 1, D_rbf]
        rbf = torch.exp(-(((dist_expand - _mu) / self.sigma) ** 2))
        return rbf

    def rbf_charge(self, charge):
        mu = torch.linspace(self._charge_conf.charge_rbf_min, self._charge_conf.charge_rbf_max, self._charge_conf.num_rbf_size)
        mu = mu.reshape([1, 1, -1])
        charge_expand = torch.unsqueeze(charge, -1) #[B, N, 1]
        _mu = mu.to(charge.device) #[1, 1, D_rbf]
        sigma = (self._charge_conf.charge_rbf_max - self._charge_conf.charge_rbf_min) / self._charge_conf.num_rbf_size
        rbf = torch.exp(-(((charge_expand - _mu) / sigma) ** 2))
        return rbf
    
    def forward(
        self,
        ligand_atom, # [B, N]
        ligand_pos, # [B, N, 3]
        edge_mask, # [B, N, N]
        atom_residue, # [B, N]
        ligand_charge = None # [B, N]
    ):

        num_batch, num_atom = ligand_atom.shape
        if num_atom == 0:
            device = ligand_atom.device
            node_embed_size = self._model_conf.node_embed_size
            edge_embed_size = self._model_conf.edge_embed_size
            # return empty tensors with corresponding shapes
            return (
                torch.zeros((num_batch, 0, node_embed_size), device=device),
                torch.zeros((num_batch, 0, 0, edge_embed_size), device=device)
            )
        
        # Initializing Node embedding
        atom_embed = self.node_embedder(ligand_atom) # [B, N, D_node]
        if self._charge_conf is not None and self._charge_conf.embed_charge and ligand_charge is not None:
            # [B, N, D_charge_rbf]
            charge_embed = self.rbf_charge(ligand_charge) 
            atom_embed = torch.cat([atom_embed, charge_embed], dim=-1) # [B, N, D_node + D_charge_rbf]
            atom_embed = self.atom_feature_fusion(atom_embed) 

        residue_embed = self.residue_embedder(atom_residue)
        gated_residue_embed = self.feature_fusion(atom_embed, residue_embed)
        node_embed  = atom_embed + gated_residue_embed


        # Initializing Edge embedding
        # [B, N, N]
        radial, dist = self.coord2dist(
                            coord=ligand_pos, 
                            edge_mask=edge_mask,
                        )
        #[B, N, N]
        distance_mask = (radial <= self._distance_threshold**2)
        edge_mask = edge_mask * distance_mask
        # [B, N, N, num_rbf_size]
        edge_embed = self.rbf(dist) * edge_mask[..., None]
        # [B, N, N, D_node]
        src_node_embed = node_embed.unsqueeze(1).repeat(1, num_atom, 1, 1) 
        # [B, N, N, D_node]
        tar_node_embed = node_embed.unsqueeze(2).repeat(1, 1, num_atom, 1) 
        # [B, N, N, D_node + D_node + D_rbf]
        edge_embed = torch.cat([src_node_embed, tar_node_embed, edge_embed], dim=-1)
        # [B, N, N, D_edge]
        edge_embed = self.edge_embedder(edge_embed.to(torch.float)) 

        # Compute message
        src_node_agg = (edge_embed.sum(dim=1) / (edge_mask[..., None].sum(dim=1)+1e-6)) * ligand_atom.clamp(max=1.)[..., None]
        # edge_embed.sum(dim=1): [B, N, D_edge] -- for each target node j, sum
        # the edge embeddings edge_embed[:, i, j, :] over source nodes i.
        # edge_mask[..., None].sum(dim=1)+1e-10 normalizes by each node's degree.
        # If a target node is an actual atom (not padding) -- indicated by
        # ligand_atom.clamp(max=1.) == 1 -- then src_node_agg is the averaged
        # neighbor message for that node.
        src_node_agg = torch.cat([node_embed, src_node_agg], dim=-1)

        # update node embedding
        node_embed = node_embed + self.node_aggregator(src_node_agg)

        return node_embed, edge_embed
    

#Cross-IPA
class CrossIPA(nn.Module):
    def __init__(
        self,
        ipa_conf,
        inf: float = 1e5,
        eps: float = 1e-8,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_l:
                Ligand Node Represenation channel dimension
            c_hidden:
                Hidden channel dimension
            no_heads:
                Number of attention heads
            no_qk_points:
                Number of query/key points to generate
            no_v_points:
                Number of value points to generate
        """
        super(CrossIPA, self).__init__()
        self._ipa_conf = ipa_conf

        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z 
        self.c_l = ipa_conf.c_l
        self.c_hidden = ipa_conf.c_hidden
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.topk_percent = ipa_conf.topk_percent
        self.inf = inf
        self.eps = eps

        # These linear layers differ from their specifications in the
        # supplement. There, they lack bias and use Glorot initialization.
        # Here as in the official source, they have bias and use the default
        # Lecun initialization.
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_l, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_l, hpkv)

        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        self.head_weights = nn.Parameter(torch.zeros((self.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim =  (
            self.c_z // 4 + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()
    
    def sparcify_attention(self, a, percent):
        '''
        a : [B, H, N_res, N_lig]
        '''
        num_ligand = a.shape[-1]
        k = int(percent * num_ligand)
        if k < 1: 
            return a
        
        topk_values, topk_indices = torch.topk(a, k=k, dim=-1) 

        mask = torch.zeros_like(a).scatter_(-1, topk_indices, 1)  
        sparse_a = a * mask

        return sparse_a
    
    def forward(
        self,
        s_p: torch.Tensor,
        s_l: torch.Tensor,
        z: Optional[torch.Tensor],
        r_p: Rigid,
        r_l: Rigid, 
        mask_p: torch.Tensor,
        mask_l: torch.Tensor,
        _offload_inference: bool = False,
        _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            s_p:
                [*, N_res, C_s] single representation (bb node embeddings)
            s_l:
                [*, N_lig, C_l] single representation (ligand node embeddings)
            z:
                [*, N_res, N_lig, C_z] pair representation (edge embeddings) 
                C_Z: bblig_edge_embed_size
            r_p:
                [*, N_res] transformation object for protein
            r_l:
                [*, N_lig] transformation object for ligand
            mask_p:
                [*, N_res] mask for protein
                actual_input: flow_mask
            mask_l:
                [*, N_lig] mask for ligand
        Returns:
            [*, N_res, C_s] single representation update
        """
        if s_l.shape[-2] == 0:
            return torch.zeros_like(s_p)
        
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s_p)
        # [*, N_lig, 2 * H * C_hidden]
        kv = self.linear_kv(s_l)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_lig, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_lig, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)
        del kv
        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s_p) # generate vector features in local coordinates; each residue has p_q local points

        # This is kind of clunky, but it's how the original does it
        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1) # split last dim into 3 parts
        q_pts = torch.stack(q_pts, dim=-1) # stack to form [*, N_res, H * P_q, 3]
        q_pts = r_p[..., None].apply(q_pts) # project local points to global coordinates

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        ) # split into attention heads
        
        # [*, N_lig, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s_l)
        del s_p, s_l
        # [*, N_lig, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r_l[..., None].apply(kv_pts)

        # [*, N_lig, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_lig, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_res, N_lig, H]
        b = self.linear_b(z[0])
        
        if(_offload_inference):
            z[0] = z[0].cpu()

        # [*, H, N_res, N_lig]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_lig]
        )
        del q,k
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))
        
        # [*, N_res, N_lig, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5) # for broadcasting
        pt_att = pt_displacement ** 2 # square each element
        del pt_displacement, q_pts
        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1)) # split last dim into three and sum to get squared L2 norm
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        # apply softplus so weights are positive
        # reshape head_weights so each attention head's weight ([H]) can be
        # broadcast with pt_att. For example, if pt_att has shape
        # [B, N_res, N_lig, H, P_q, 3], the leading and trailing dims become 1
        # while the head dimension is inferred (-1).
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights
        # [B , N_res, N_lig, H, P_q] * [1, 1, 1, H, 1]

        # [*, N_res, N_lig, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5) # sum attention over all query points
        # [*, N_res, N_lig]
        square_mask = mask_p.unsqueeze(-1) * mask_l.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_lig]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))
        
        a = a + pt_att 
        a = a + square_mask.unsqueeze(-3) 
        a = self.softmax(a) # masked positions will have zero attention

        # Sparse attention 
        a = self.sparcify_attention(a, self.topk_percent)
        del pt_att, head_weights, square_mask
        ################
        # Compute output
        ################
        # [*, N_res, H, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3)
            # a: [*, H, N_res, N_lig]
            # v: [*, N_lig, H, C_hidden] -> [*, H, N_lig, C_hidden]
        ).transpose(-2, -3) # [*, H, N_res, C_hidden]
        del v
        # [*, N_res, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, H, 3, N_res, P_v] 
        o_pt = torch.sum(
            (
                a[..., None, :, :, None] #a:[*, H, N_res, N_lig]
                * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )
        del v_pts
        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r_p[..., None, None].invert_apply(o_pt) # project back to local coordinates

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)
        del o_pt_dists
        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if(_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z[0])
        del z
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)
        del pair_z, a
        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        o_feats = [o, *torch.unbind(o_pt, dim=-1), o_pt_norm_feats, o_pair]

        # [*, N_res, C_s]
        s = self.linear_out(
            torch.cat(
                o_feats, dim=-1
            )
        )
        
        return s

class DualGLU(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.gate_proj = nn.Linear(input_dim, input_dim)

    def forward(self, self_attn_output, 
                      cross_attn_output,
                      return_gate = False):
        # [B, N, C]
        combined = self_attn_output + cross_attn_output
        gate = torch.sigmoid(self.gate_proj(combined))
        gated_cross_attn_output = gate * cross_attn_output
        if return_gate:
            return gated_cross_attn_output, gate
        else:
            return gated_cross_attn_output
    