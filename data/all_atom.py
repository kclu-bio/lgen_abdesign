"""Utilities for calculating all atom representations."""
import torch
from openfold.data import data_transforms
from openfold.np import residue_constants
from openfold.utils import rigid_utils as ru
from data import utils as du
from openfold.utils import tensor_utils

Rigid = ru.Rigid
Rotation = ru.Rotation

# Residue Constants from OpenFold/AlphaFold2.


IDEALIZED_POS = torch.tensor(residue_constants.restype_atom14_rigid_group_positions)
DEFAULT_FRAMES = torch.tensor(residue_constants.restype_rigid_group_default_frame) 
#restype_rigid_group_default_frame = np.zeros([21, 8, 4, 4], dtype=np.float32)
ATOM_MASK = torch.tensor(residue_constants.restype_atom14_mask)
GROUP_IDX = torch.tensor(residue_constants.restype_atom14_to_rigid_group)

# Convert CA coordinates and rotation matrices to atom37.
def to_atom37(trans, rots):
    num_batch, num_res, _ = trans.shape
    final_atom37 = compute_backbone(
        du.create_rigid(rots, trans),
        torch.zeros(num_batch, num_res, 2, device=trans.device)
    )[0]
    return final_atom37


def torsion_angles_to_frames(
    r: Rigid,  # type: ignore [valid-type]
    alpha: torch.Tensor,
    aatype: torch.Tensor,
):
    """Conversion method of torsion angles to frames provided the backbone.

    Args:
        r: Backbone rigid groups.
        alpha: Torsion angles.
        aatype: residue types.

    Returns:
        All 8 frames corresponding to each torsion frame.

    """
    # [*, N, 8, 4, 4]
    with torch.no_grad():
        default_4x4 = DEFAULT_FRAMES.to(aatype.device)[aatype, ...]  # type: ignore [attr-defined]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)  # type: ignore [attr-defined] # Eight Rigid objects per residue.

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2)

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots = Rigid(Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)  # type: ignore [index]

    return all_frames_to_global


def prot_to_torsion_angles(aatype, atom37, atom37_mask):
    """Calculate torsion angle features from protein features."""
    prot_feats = {
        "aatype": aatype,
        "all_atom_positions": atom37,
        "all_atom_mask": atom37_mask,
    }
    torsion_angles_feats = data_transforms.atom37_to_torsion_angles()(prot_feats)
    torsion_angles = torsion_angles_feats["torsion_angles_sin_cos"]
    torsion_mask = torsion_angles_feats["torsion_angles_mask"]
    return torsion_angles, torsion_mask


def frames_to_atom14_pos(
    r: Rigid,  # type: ignore [valid-type]
    aatype: torch.Tensor,
):
    """Convert frames to their idealized all atom representation.

    Args:
        r: All rigid groups. [..., N, 8, 3]
        aatype: Residue types. [..., N]

    Returns:

    """
    with torch.no_grad():
        group_mask = GROUP_IDX.to(aatype.device)[aatype, ...]
        group_mask = torch.nn.functional.one_hot(
            group_mask,
            num_classes=DEFAULT_FRAMES.shape[-3],
        )
        frame_atom_mask = ATOM_MASK.to(aatype.device)[aatype, ...].unsqueeze(-1)  # type: ignore [attr-defined]
        frame_null_pos = IDEALIZED_POS.to(aatype.device)[aatype, ...]  # type: ignore [attr-defined]

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask  # type: ignore [index]

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # [*, N, 14, 3]
    pred_positions = t_atoms_to_global.apply(frame_null_pos)
    pred_positions = pred_positions * frame_atom_mask

    return pred_positions


def compute_backbone(bb_rigids, psi_torsions, aatype = None):
    # psi_torisons: [B, N, 2]
    # torsion_angles: [B, N, 7, 2]
    # aatype: [B, N]
    torsion_angles = torch.tile(
        psi_torsions[..., None, :],  #[B, N, 1, 2]
        tuple([1 for _ in range(len(bb_rigids.shape))]) # (1, 1)
        + (7, 1) #(1, 1, 7, 1)
    ) # Replicate the all-zero torsion_angles seven times.

    if aatype is None:
        # Generate alanine uniformly.
        aatype = torch.zeros(bb_rigids.shape, device=bb_rigids.device).long()
    all_frames = torsion_angles_to_frames(
        bb_rigids,
        torsion_angles,
        aatype,
    )
    atom14_pos = frames_to_atom14_pos(all_frames, aatype)
    atom14_pos[...,5:,:] = 0
    atom37_bb_pos = torch.zeros(bb_rigids.shape + (37, 3), device=bb_rigids.device)
    # atom14 bb order = ['N', 'CA', 'C', 'O', 'CB']
    # atom37 bb order = ['N', 'CA', 'C', 'CB', 'O']
    atom37_bb_pos[..., :3, :] = atom14_pos[..., :3, :]
    atom37_bb_pos[..., 3, :] = atom14_pos[..., 4, :]
    atom37_bb_pos[..., 4, :] = atom14_pos[..., 3, :]
    atom37_mask = torch.any(atom37_bb_pos, axis=-1)
    return atom37_bb_pos, atom37_mask, aatype, atom14_pos


def calculate_neighbor_angles(R_ac, R_ab):
    """Calculate angles between atoms c <- a -> b.

    Parameters
    ----------
        R_ac: Tensor, shape = (N,3)
            Vector from atom a to c.
        R_ab: Tensor, shape = (N,3)
            Vector from atom a to b.

    Returns
    -------
        angle_cab: Tensor, shape = (N,)
            Angle between atoms c <- a -> b.
    """
    # cos(alpha) = (u * v) / (|u|*|v|)
    x = torch.sum(R_ac * R_ab, dim=1)  # shape = (N,)
    # sin(alpha) = |u x v| / (|u|*|v|)
    y = torch.cross(R_ac, R_ab).norm(dim=-1)  # shape = (N,)
    # avoid that for y == (0,0,0) the gradient wrt. y becomes NaN
    y = torch.max(y, torch.tensor(1e-9))
    angle = torch.atan2(y, x)
    return angle


def vector_projection(R_ab, P_n):
    """
    Project the vector R_ab onto a plane with normal vector P_n.

    Parameters
    ----------
        R_ab: Tensor, shape = (N,3)
            Vector from atom a to b.
        P_n: Tensor, shape = (N,3)
            Normal vector of a plane onto which to project R_ab.

    Returns
    -------
        R_ab_proj: Tensor, shape = (N,3)
            Projected vector (orthogonal to P_n).
    """
    a_x_b = torch.sum(R_ab * P_n, dim=-1)
    b_x_b = torch.sum(P_n * P_n, dim=-1)
    return R_ab - (a_x_b / b_x_b)[:, None] * P_n


def transrot_traj_to_atom37(transrot_traj, res_mask):
    atom37_traj = []
    for trans, rots in transrot_traj:
        atom37_traj.append(atom37_from_trans_rot(trans, rots, res_mask))
    return atom37_traj


def atom37_from_trans_rot(trans, rots, res_mask = None, aatype = None):
    if res_mask is None:
        res_mask = torch.ones([*trans.shape[:-1]], device=trans.device)
    rigids = du.create_rigid(rots, trans)
    atom37 = compute_backbone(
        bb_rigids = rigids,
        psi_torsions = torch.zeros(
            trans.shape[0],
            trans.shape[1],
            2,
            device=trans.device
        ),
        aatype = aatype 

    )[0]
    # Return atom37_bb_pos, atom37_mask, aatype, and atom14_pos.
    batch_atom37 = []
    num_batch = res_mask.shape[0]
    for i in range(num_batch):
        batch_atom37.append(
            du.adjust_oxygen_pos(atom37[i], res_mask[i])
        )
    return torch.stack(batch_atom37)


def get_rc_tensor(rc_np, aatype):
    return torch.tensor(rc_np, device=aatype.device)[aatype]

def atom14_to_atom37(
    atom14_data: torch.Tensor,  # (*, N, 14, ...)
    aatype: torch.Tensor # (*, N)
) -> tuple:    # (*, N, 37, ...)
    """Convert atom14 to atom37 representation."""
    idx_atom37_to_atom14 = get_rc_tensor(residue_constants.RESTYPE_ATOM37_TO_ATOM14, aatype).long()
    no_batch_dims = len(aatype.shape) - 1
    atom37_data = tensor_utils.batched_gather(
        atom14_data, 
        idx_atom37_to_atom14, 
        dim=no_batch_dims + 1, 
        no_batch_dims=no_batch_dims + 1
    )
    atom37_mask = get_rc_tensor(residue_constants.RESTYPE_ATOM37_MASK, aatype) 
    if len(atom14_data.shape) == no_batch_dims + 2:
        atom37_data *= atom37_mask
    elif len(atom14_data.shape) == no_batch_dims + 3:
        atom37_data *= atom37_mask[..., None].to(dtype=atom37_data.dtype)
    else:
        raise ValueError("Incorrectly shaped data")
    return atom37_data, atom37_mask