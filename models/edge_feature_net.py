import torch
from torch import nn

from models.utils import get_index_embedding, calc_distogram

class EdgeFeatureNet(nn.Module):

    def __init__(self, module_cfg, embed_sc = True):
        #   c_s, c_p, relpos_k, template_type):
        super(EdgeFeatureNet, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s #node embed size
        self.c_p = self._cfg.c_p #edge embed size
        self.feat_dim = self._cfg.feat_dim

        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
        self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)
        self.new_version = getattr(self._cfg, 'new_version', False)
        total_edge_feats = self.feat_dim * 3 + self._cfg.num_bins * 1
        if embed_sc:
            total_edge_feats+=self._cfg.num_bins
        if self._cfg.embed_chain:
            total_edge_feats += 1
        if self._cfg.embed_diffuse_mask:
            total_edge_feats += 2
        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.SiLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.SiLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )

    def embed_relpos(self, r, chain_idx):
        # AlphaFold 2 Algorithm 4 & 5
        # Based on OpenFold utils/tensor_utils.py
        # r: [b, n_res]
        # chain_idx：[b, n_res]
        # [b, n_res, n_res]
        d = r[:, :, None] - r[:, None, :]
        # [b, n_res, n_res] -> True indicates residues from different chains.
        diff_chain_mask = chain_idx[:, :, None] != chain_idx[:, None, :]
        # The relative-position difference between chains is 2056.
        d = torch.where(diff_chain_mask, torch.full_like(d, 2056), d)
        pos_emb = get_index_embedding(d, self._cfg.feat_dim, max_len=2056)
        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])

    def forward(self, s, t, sc_t, res_idx, p_mask, diffuse_mask, chain_idx):
        """
        s: Init Node embed [b, n_res, c_s]
        t: Translation
        sc_t: Self-condition translation
        res_idx: Residue index [b, n_res]
        p_mask: Edge mask #[b, n_res, n_res]
        """
        # Input: [b, n_res, c_s]
        num_batch, num_res, _ = s.shape

        # [b, n_res, c_p]
        p_i = self.linear_s_p(s)
        # [b, n_res, n_res, c_p*2]
        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)

        # [b, n_res, n_res, c_p]
        r = res_idx

        relpos_feats = self.embed_relpos(r, chain_idx)

        dist_feats = calc_distogram(
            t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
        
        all_edge_feats = [cross_node_feats, relpos_feats, dist_feats]
        if sc_t is not None:
            sc_feats = calc_distogram(
                sc_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
            all_edge_feats.append(sc_feats)

        if self._cfg.embed_chain:
            rel_chain = (chain_idx[:, :, None] == chain_idx[:, None, :]).float()
            all_edge_feats.append(rel_chain[..., None])
        if self._cfg.embed_diffuse_mask:
            diff_feat = self._cross_concat(diffuse_mask[..., None], num_batch, num_res)
            all_edge_feats.append(diff_feat)
        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1))
        edge_feats *= p_mask.unsqueeze(-1)
        return edge_feats

class CrossEdgeFeatureNet(nn.Module):
    # Embed edges between protein and ligand
    def __init__(self, module_cfg, embed_sc=True):
        #   c_s, c_p, relpos_k, template_type):
        super(CrossEdgeFeatureNet, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s #protein node embed size
        self.c_l = self._cfg.c_l #ligand node embed size
        self.c_p = self._cfg.c_p #final edge embed size
        self.pro_feat_dim = self._cfg.pro_feat_dim
        self.lig_feat_dim = self._cfg.lig_feat_dim

        self.linear_s_p = nn.Linear(self.c_s, self.pro_feat_dim)
        self.linear_s_l = nn.Linear(self.c_l, self.lig_feat_dim)

        self.dist_min = self._cfg.ligand_rbf_d_min
        self.dist_max = self._cfg.ligand_rbf_d_max
        self.num_rbf_size = self._cfg.num_rbf_size

        total_edge_feats = self.pro_feat_dim + self.lig_feat_dim +  self.num_rbf_size
        if embed_sc:
            total_edge_feats += self.num_rbf_size

        mu = torch.linspace(self.dist_min, self.dist_max, self.num_rbf_size)
        self.mu = mu.reshape([1, 1, 1, -1])
        self.sigma = (self.dist_max - self.dist_min) / self.num_rbf_size

        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.SiLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.SiLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )

    def coord2dist(self, coord_p, coord_l, edge_mask):
        # edge_mask: [n_res, n_atom]
        n_batch, n_atom = coord_l.size(0), coord_l.size(1)
        _, n_res = coord_p.size(0), coord_p.size(1)
        # [B, num_res, num_atom]
        radial = torch.sum((coord_p.unsqueeze(2) - coord_l.unsqueeze(1)) ** 2, dim=-1) 
        dist = torch.sqrt(
                radial + 1e-10
            ) * edge_mask

        radial = radial * edge_mask
        return radial, dist
    
    def rbf(self, dist):
        # Dist: [B, N_res, N_lig]
        dist_expand = torch.unsqueeze(dist, -1) #[B, N_res, N_lig, 1]
        _mu = self.mu.to(dist.device) #[1, 1, 1, D_rbf]
        rbf = torch.exp(-(((dist_expand - _mu) / self.sigma) ** 2))
        return rbf

    def _cross_concat(self, feats_p, feats_l, num_batch, num_res, num_lig):
        """
        Cross-concatenate protein and ligand node features for edge construction.

        This function expands the protein node features along the ligand dimension
        and the ligand node features along the protein dimension, then concatenates
        them along the last axis. The result is a feature tensor representing all
        possible protein-ligand node pairs, suitable for edge feature construction
        in a protein-ligand interaction graph.

        Args:
            feats_p (torch.Tensor): Protein node features, shape [num_batch, num_res, feat_dim_p].
            feats_l (torch.Tensor): Ligand node features, shape [num_batch, num_lig, feat_dim_l].
            num_batch (int): Batch size.
            num_res (int): Number of protein residues.
            num_lig (int): Number of ligand atoms.

        Returns:
            torch.Tensor: Cross-concatenated features, shape [num_batch, num_res, num_lig, feat_dim_p + feat_dim_l].
        """
        return torch.cat([
            torch.tile(feats_p[:, :, None, :], (1, 1, num_lig, 1)),
            torch.tile(feats_l[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_lig, -1])

    def forward(self, s_p, s_l, t_p, sc_t_p, t_l, edge_mask):
        """
        s_p: Init Protein Node embed [b, n_res, c_s]
        s_l: Init Ligand Node embed [b, n_lig, c_l]
        t_p: Protein Translation [b, n_res, 3]
        t_l: Ligand Translation [b, n_lig, 3]
        sc_t_p: Self-condition protein translation [b, n_res, 3]
        edge_mask: Edge mask #[b, n_res, n_lig]
        """
        
        num_batch, num_res, _ = s_p.shape
        _, num_lig, _ = s_l.shape

        # [b, n_res, pro_feat_size]
        pro_node_feat = self.linear_s_p(s_p)
        # [b, n_res, lig_feat_size]
        lig_node_feat = self.linear_s_l(s_l)

        # [b ,n_res, n_lig, pro_feat_size + lig_feat_size]
        cross_node_feats = self._cross_concat(pro_node_feat, lig_node_feat, num_batch, num_res, num_lig)

        radial, dist = self.coord2dist(
                            coord_p = t_p,
                            coord_l = t_l, 
                            edge_mask=edge_mask,
                        )
        #dist : [B, N_res, N_lig]

        dist_feats = self.rbf(dist) * edge_mask[..., None]
        all_edge_feats = [cross_node_feats, dist_feats]
        if sc_t_p is not None:
            radial_sc, dist_sc = self.coord2dist(
                                coord_p = sc_t_p,
                                coord_l = t_l, 
                                edge_mask=edge_mask,
                            )
            sc_feats = self.rbf(dist_sc) * edge_mask[..., None]
            all_edge_feats.append(sc_feats)
            
        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1))
        edge_feats *= edge_mask.unsqueeze(-1)
        return edge_feats