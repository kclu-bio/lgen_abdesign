import torch
from torch import nn
from models.utils import get_index_embedding, get_time_embedding

class NodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        embed_size = self._cfg.c_pos_emb + self._cfg.c_timestep_emb * 2 + 1
        if self._cfg.embed_chain:
            embed_size += self._cfg.c_pos_emb
        if self._cfg.embed_aatype:
            self.aatype_embedding = nn.Embedding(getattr(self._cfg, "aatype_embed_num_tokens", 21), self.c_s) 
            # 22: 20 amino acids + 1 for unk (non-hetatm) + 1 for pad
            embed_size += self.c_s + self._cfg.c_timestep_emb + self._cfg.aatype_pred_num_tokens
        if self._cfg.use_mlp:
            self.linear = nn.Sequential(
                nn.Linear(embed_size, self.c_s),
                nn.SiLU(),
                nn.Linear(self.c_s, self.c_s),
                nn.SiLU(),
                nn.Linear(self.c_s, self.c_s),
                nn.LayerNorm(self.c_s),
            )
        else:
            self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(
            self,
            *,
            so3_t,
            r3_t,
            cat_t,
            res_mask,
            diffuse_mask,
            chain_index,
            pos,
            aatypes,
            aatypes_sc,
        ):
        # s: [b]

        # [b, n_res, c_pos_emb]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [
            pos_emb,
            diffuse_mask[..., None],
            self.embed_t(so3_t, res_mask),
            self.embed_t(r3_t, res_mask)
        ]
        if self._cfg.embed_aatype:
            input_feats.append(self.aatype_embedding(aatypes))
            input_feats.append(self.embed_t(cat_t, res_mask))
            input_feats.append(aatypes_sc) #one hot
        if self._cfg.embed_chain:
            input_feats.append(
                get_index_embedding(
                    chain_index,
                    self.c_pos_emb,
                    max_len=100
                )
            )
        return self.linear(torch.cat(input_feats, dim=-1))

class AllAtomNodeFeatureNet(nn.Module):
    def __init__(self, module_cfg, shared_gnn):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        # Atom embedding dimension output by minignn.
        self.c_atom_emb = self._cfg.c_atom_emb
        self.c_pos_emb = self._cfg.c_pos_emb
        embed_size = self.c_pos_emb + self.c_atom_emb + 1 # 1 for diffuse mask
        self.aatype_embedding = nn.Embedding(21, self.c_s) # Always 21 because of 20 amino acids + 1 for unk
        embed_size += self.c_s
        if self._cfg.embed_chain:
            embed_size += self._cfg.c_pos_emb//2
        
        self.minignn = shared_gnn #share weight with ligand net

        if self._cfg.use_mlp:
            self.linear = nn.Sequential(
                nn.Linear(embed_size, self.c_s),
                nn.SiLU(),
                nn.Linear(self.c_s, self.c_s),
                nn.SiLU(),
                nn.Linear(self.c_s, self.c_s),
                nn.LayerNorm(self.c_s),
            )
        else:
            self.linear = nn.Linear(embed_size, self.c_s)


    def forward(
            self,
            res_mask,
            diffuse_mask,
            chain_index,
            pos, #residue index, not atom coordinate
            aatypes, #[B, N, 21]
            atom14_gt_pos, # [B, N, 14, 3]
            atom14_elements, # [B, N, 14]
            atom14_atom_exists, # [B, N, 14]
            atom14_res_edge_mask #[B, N, 14, 14]
        ):
        # s: [b]

        # [b, n_res, c_pos_emb]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        # [b, n_res, c_pos_emb+1]
        input_feats = [
            pos_emb,
            diffuse_mask[..., None],
        ]

        #[B, N] -> [B, N, 14]
        atom_residue = aatypes[...,None].repeat(1, 1 ,14)
        #[B, N, 14, N_lig_node]
        res_atom_embed = self.minignn(ligand_atom = atom14_elements,
                                      ligand_pos = atom14_gt_pos,
                                      edge_mask = atom14_res_edge_mask,
                                      atom_residue = atom_residue)
        #[B, N, N_lig_node] / [B, N, 1]
        res_embed = torch.sum(res_atom_embed, dim=-2)/torch.sum(atom14_atom_exists, dim=-1)[...,None]
        
        # [b, n_res, c_pos_emb + 1 + N_lig_node]
        input_feats.append(res_embed)
        # [b, n_res, c_pos_emb + 1 + N_lig_node + c_s]
        input_feats.append(self.aatype_embedding(aatypes))
    
        if self._cfg.embed_chain:
            # [b, n_res, c_pos_emb + 1 + N_lig_node + c_s + c_pos_emb]
            input_feats.append(
                get_index_embedding(
                    chain_index,
                    self.c_pos_emb//2,
                    max_len=100
                )
            )
        
        # [b, n_res, c_s]
        return self.linear(torch.cat(input_feats, dim=-1))