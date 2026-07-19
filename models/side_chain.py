from models import ipa_pytorch
from openfold.utils.feats import (
    frames_and_literature_positions_to_atom14_pos,
    torsion_angles_to_frames,
)
from openfold.np.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
import torch
import torch.nn as nn
from openfold.utils.rigid_utils import Rotation, Rigid


# Sidechain module inputs: current rigids, predicted residue types, and node single representations.
def initialize_constants(module: nn.Module, float_dtype, device):
    """Register a constant buffer on any nn.Module."""

    if not hasattr(module, "default_frames"):
        module.register_buffer(
            "default_frames",
            torch.tensor(
                restype_rigid_group_default_frame,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            ),
            persistent=False,
        )

    if not hasattr(module, "group_idx"):
        module.register_buffer(
            "group_idx",
            torch.tensor(
                restype_atom14_to_rigid_group,
                device=device,
                requires_grad=False,
            ),
            persistent=False,
        )

    if not hasattr(module, "atom_mask"):
        module.register_buffer(
            "atom_mask",
            torch.tensor(
                restype_atom14_mask,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            ),
            persistent=False,
        )

    if not hasattr(module, "lit_positions"):
        module.register_buffer(
            "lit_positions",
            torch.tensor(
                restype_atom14_rigid_group_positions,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            ),
            persistent=False,
        )

# Predict atom14 coordinates (excluding four backbone atoms) in one pass.
class SidechainModule(nn.Module):
    def __init__(self, cfg, aatype_embedding_layer):
        super(SidechainModule, self).__init__()
        self.c_s = cfg.c_s #node_embed_dim
        self.c_resnet = cfg.c_resnet # resnet hidden dim
        self.no_resnet_blocks = cfg.no_resnet_blocks
        self.no_angles = cfg.no_angles
        self.epsilon = cfg.epsilon
        self.mode = cfg.mode
        self.aatype_embedding = aatype_embedding_layer
        self.angle_resnet = ipa_pytorch.AngleResnet(
            self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            self.no_angles,
            self.epsilon,
        )

    def forward(self, node_embed, pred_aa, curr_rigids, mask):
        """
        Args:
            node_embed:
                [*, N_res, C_s] single representation
            aatype:
                [*, N_res] amino acid indices
            mask:
                [*, N_res] residue mask
            curr_rigids:
                backbone rigid frame (unit: angstrom)
        """
        
        # curr_rigids already updates only the pocket, so diffuse_mask is unnecessary here.
        backb_to_global = Rigid(
            Rotation(
                rot_mats=curr_rigids.get_rots().get_rot_mats(), 
                quats=None
            ),
            curr_rigids.get_trans(),
        )

        s_initial = self.aatype_embedding(pred_aa)

        # [B, N, 4, 2]
        unnormalized_angles, angles = self.angle_resnet(node_embed, s_initial)

        num_batch, num_res, _, _ = angles.shape

        # [B, N, 2, 2]
        torsion_angles_zero = torch.zeros(num_batch, 
                                            num_res,
                                            3, 
                                            2, device=angles.device)
        torsion_angles = torch.cat([torsion_angles_zero, angles], dim=2) 
        initialize_constants(self, angles.dtype, angles.device)

            # [B, N, 8]
        all_frames_to_global = torsion_angles_to_frames(
            backb_to_global,
            torsion_angles,
            pred_aa,
            self.default_frames
        )

        # [B, N, 14, 3]
        pred_xyz = frames_and_literature_positions_to_atom14_pos(
            all_frames_to_global,
            pred_aa,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions, 
        )

        return {
            "sidechain_frames": all_frames_to_global.to_tensor_4x4(), # Eight sidechain frames [*, N, 8].
            "angles": torsion_angles, #[*, N, 7, 2]
            "positions": pred_xyz, #[*, N, 14, 3]
        }
