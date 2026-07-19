from torch import nn
import torch
from openfold.utils.rigid_utils import Rigid
from typing import Optional
from openfold.np import residue_constants
import torch.nn.functional as F

def torsion_angle_loss(
    a,  # [*, N, n_angle, 2]
    a_gt,  # [*, N, n_angle, 2]
    a_alt_gt,  # [*, N, n_angle, 2],
    res_mask = None, # [*, N]
    a_mask = None #[*, N, n_angle]
):  
    n_angle = a.shape[-2]
    assert a_gt.shape[-2] == n_angle, f"a_gt n_angle mismatch: {a_gt.shape[-2]} vs {n_angle}"
    assert a_alt_gt.shape[-2] == n_angle, f"a_alt_gt n_angle mismatch: {a_alt_gt.shape[-2]} vs {n_angle}"
    assert a_mask.shape[-1] == n_angle, f"a_alt_gt n_angle mismatch: {a_alt_gt.shape[-1]} vs {n_angle}"

    # [*, N, n_anlge]
    norm = torch.norm(a, dim=-1)

    # [*, N, n_angle]
    a = a / norm.unsqueeze(-1)
    norm = torch.where(norm < 1e-6, torch.tensor(1e-6, device=norm.device), norm)

    # [*, N, n_angle]
    diff_norm_gt = torch.norm(a - a_gt, dim=-1)
    diff_norm_alt_gt = torch.norm(a - a_alt_gt, dim=-1)
    min_diff = torch.minimum(diff_norm_gt ** 2, diff_norm_alt_gt ** 2)

    # apply mask
    if res_mask is None:
        res_shape = a.shape[:-2]  
        res_mask = torch.ones(res_shape, device=a.device, dtype=a.dtype)
    
    if a_mask is None:
        a_shape = a.shape[:-1] 
        a_mask = torch.ones(a_shape, device=a.device, dtype=a.dtype)
    
    #  [*, N, 1]
    res_mask_expanded = res_mask.unsqueeze(-1)
    # [*, N, n_angle]
    combined_mask = res_mask_expanded * a_mask
    mask_sum = torch.sum(combined_mask, dim=(-1, -2))
    # [*]
    eps = 1e-6
    
    # [*] compute masked mean
    min_diff_masked = min_diff * combined_mask
    norm_diff_masked = torch.abs(norm - 1) * combined_mask
    l_torsion = torch.sum(min_diff_masked, dim=(-1, -2)) / (mask_sum + eps)
    l_angle_norm = torch.sum(norm_diff_masked, dim=(-1, -2)) / (mask_sum + eps)

    an_weight = 0.02
    return l_torsion + an_weight * l_angle_norm

def compute_fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    l1_clamp_distance: Optional[float] = None,
    eps=1e-8,
    ignore_nan=True,
) -> torch.Tensor:
    """
        Computes FAPE loss.

        Args:
            pred_frames:
                [*, N_frames] Rigid object of predicted frames
            target_frames:
                [*, N_frames] Rigid object of ground truth frames
            frames_mask:
                [*, N_frames] binary mask for the frames
            pred_positions:
                [*, N_pts, 3] predicted atom positions
            target_positions:
                [*, N_pts, 3] ground truth positions
            positions_mask:
                [*, N_pts] positions mask
            length_scale:
                Length scale by which the loss is divided
            l1_clamp_distance:
                Cutoff above which distance errors are disregarded
            eps:
                Small value used to regularize denominators
        Returns:
            [*] loss tensor
    """
    # [*, N_frames, N_pts, 3]
    # Project global coordinates into local frames
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )

    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    # If generated residues differ from the original residue types, the loss
    # may be inflated because `positions_mask` is determined by original types
    normed_error = normed_error * positions_mask[..., None, :] 
    if ignore_nan:
        normed_error = torch.nan_to_num(normed_error)

    # FP16-friendly averaging. Roughly equivalent to:
    #
    # norm_factor = (
    #     torch.sum(frames_mask, dim=-1) *
    #     torch.sum(positions_mask, dim=-1)
    # )
    # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
    #
    # ("roughly" because eps is necessarily duplicated in the latter)
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = (
        normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
    )
    normed_error = torch.sum(normed_error, dim=-1)
    normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))
    return normed_error

def sidechain_loss(
    sidechain_frames: torch.Tensor, # [B, N, n_rigid_group, 4, 4]
    sidechain_atom_pos: torch.Tensor,  # [B, N, 14, 3]
    rigidgroups_gt_frames: torch.Tensor, # [B, N, n_rigid_group, 4, 4]
    rigidgroups_alt_gt_frames: torch.Tensor, #[B, N, n_rigid_group, 4, 4]
    rigidgroups_gt_exists: torch.Tensor, #[B, N, n_rigid_group]
    renamed_atom14_gt_positions: torch.Tensor, # [B, N, 14, 3]
    renamed_atom14_gt_exists: torch.Tensor, # [B, N, 14]
    alt_naming_is_better: torch.Tensor, # [B, N]
    res_mask: torch.tensor =None, #[B, N]
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    # [B, N, n_rigid_group, 4, 4]
    renamed_gt_frames = (
        1.0 - alt_naming_is_better[..., None, None, None]
    ) * rigidgroups_gt_frames + alt_naming_is_better[
        ..., None, None, None
    ] * rigidgroups_alt_gt_frames

    # Steamroll the inputs
    batch_dims = sidechain_frames.shape[:-4]
    # [B, N, n_rigid_group, 4, 4] ->  [B, N*n_rigid_groups, 4, 4]
    sidechain_frames = sidechain_frames.view(*batch_dims, -1, 4, 4)
    sidechain_frames = Rigid.from_tensor_4x4(sidechain_frames)
    # [B, N*n_rigid_group, 4, 4]
    renamed_gt_frames = renamed_gt_frames.view(*batch_dims, -1, 4, 4)
    renamed_gt_frames = Rigid.from_tensor_4x4(renamed_gt_frames)

    if res_mask is not None:
        #[B, N, n_rigid_group]*[B, N, 1]
        rigidgroups_gt_exists = rigidgroups_gt_exists * res_mask[...,None]
        renamed_atom14_gt_exists = renamed_atom14_gt_exists * res_mask[...,None]

    # [B, N*n_rigid_group]
    rigidgroups_gt_exists = rigidgroups_gt_exists.reshape(*batch_dims, -1)
    sidechain_atom_pos = sidechain_atom_pos.view(*batch_dims, -1, 3)
    renamed_atom14_gt_positions = renamed_atom14_gt_positions.view(
        *batch_dims, -1, 3
    )
    renamed_atom14_gt_exists = renamed_atom14_gt_exists.view(*batch_dims, -1)

    fape = compute_fape(
        sidechain_frames, # predicted rigid group
        renamed_gt_frames, # ground truth rigid group
        rigidgroups_gt_exists, # rigid group mask
        sidechain_atom_pos, #pred global pos
        renamed_atom14_gt_positions, # ground truth global pos
        renamed_atom14_gt_exists, #sidechain mask
        l1_clamp_distance=clamp_distance,
        length_scale=length_scale,
        eps=eps,
    )

    return fape

def between_residue_bond_loss(
    pred_atom_positions: torch.Tensor,  # (*, N, 37/14, 3)
    pred_atom_mask: torch.Tensor,  # (*, N, 37/14)
    residue_index: torch.Tensor,  # (*, N)
    aatype: torch.Tensor,  # (*, N)
    res_mask: torch.Tensor, #(*, N)
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0,
    eps=1e-6,
) -> dict[str, torch.Tensor]:
    """Flat-bottom loss to penalize structural violations between residues.

    This is a loss penalizing any violation of the geometry around the peptide
    bond between consecutive amino acids. This loss corresponds to
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 44, 45.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      aatype: Amino acid type of given residue
      tolerance_factor_soft: soft tolerance factor measured in standard deviations
        of pdb distributions
      tolerance_factor_hard: hard tolerance factor measured in standard deviations
        of pdb distributions

    Returns:
      Dict containing:
        * 'c_n_loss_mean': Loss for peptide bond length violations
        * 'ca_c_n_loss_mean': Loss for violations of bond angle around C spanned
            by CA, C, N
        * 'c_n_ca_loss_mean': Loss for violations of bond angle around N spanned
            by C, N, CA
        * 'per_residue_loss_sum': sum of all losses for each residue
        * 'per_residue_violation_mask': mask denoting all residues with violation
            present.
    """
    # 1. Get the positions of the relevant backbone atoms.
    # [B, N-1, 1, 3]
    this_ca_pos = pred_atom_positions[..., :-1, 1, :] # current residue CA
    this_ca_mask = pred_atom_mask[..., :-1, 1] # whether current residue has CA
    this_c_pos = pred_atom_positions[..., :-1, 2, :]  # current residue C
    this_c_mask = pred_atom_mask[..., :-1, 2]
    next_n_pos = pred_atom_positions[..., 1:, 0, :] # next residue N
    next_n_mask = pred_atom_mask[..., 1:, 0]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :] # next residue CA
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0 # check residue continuity

    # 2. Compute loss for the C--N bond.
    # [B, N-1, 1]
    c_n_bond_length = torch.sqrt(
        eps + torch.sum((this_c_pos - next_n_pos) ** 2, dim=-1)
    )

    # The C-N bond to proline has slightly different length because of the ring.
    # [B, N-1]
    next_is_proline = aatype[..., 1:] == residue_constants.resname_to_idx["PRO"]
    gt_length = (
                    ~next_is_proline
                ) * residue_constants.between_res_bond_length_c_n[
                    0
                ] + next_is_proline * residue_constants.between_res_bond_length_c_n[
                    1
                ]
    gt_stddev = (
                    ~next_is_proline
                ) * residue_constants.between_res_bond_length_stddev_c_n[
                    0
                ] + next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[
                    1
                ]
    c_n_bond_length_error = torch.sqrt(eps + (c_n_bond_length - gt_length) ** 2)
    # [B, N-1]
    c_n_loss_per_residue = torch.nn.functional.relu(
        c_n_bond_length_error - tolerance_factor_soft * gt_stddev
    )

    # Apply pocket mask
    mask = this_c_mask * next_n_mask * has_no_gap_mask * res_mask[...,:-1]
    # This loss is averaged # [B, ]
    c_n_loss = torch.sum(mask * c_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    # [B, N]
    c_n_violation_mask = mask * (
        c_n_bond_length_error > (tolerance_factor_hard * gt_stddev)
    )

    # 3. Compute loss for the angles: CA-C-N angle, C-N-CA angle
    ca_c_bond_length = torch.sqrt(
        eps + torch.sum((this_ca_pos - this_c_pos) ** 2, dim=-1)
    )
    n_ca_bond_length = torch.sqrt(
        eps + torch.sum((next_n_pos - next_ca_pos) ** 2, dim=-1)
    )

    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[..., None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[..., None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[..., None]

    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
    gt_stddev = residue_constants.between_res_bond_length_stddev_c_n[0]
    ca_c_n_cos_angle_error = torch.sqrt(
        eps + (ca_c_n_cos_angle - gt_angle) ** 2
    )
    # [B, N]
    ca_c_n_loss_per_residue = torch.nn.functional.relu(
        ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask * res_mask[...,:-1]
    # [B]
    ca_c_n_loss = torch.sum(mask * ca_c_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    # [B, N]
    ca_c_n_violation_mask = mask * (
        ca_c_n_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
    gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = torch.sqrt(
        eps + torch.square(c_n_ca_cos_angle - gt_angle)
    )
    #[B, N-1]
    c_n_ca_loss_per_residue = torch.nn.functional.relu(
        c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask * res_mask[...,:-1]
    c_n_ca_loss = torch.sum(mask * c_n_ca_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c_n_ca_violation_mask = mask * (
        c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    # Compute a per residue loss (equally distribute the loss to both
    # neighbouring residues).
    per_residue_loss_sum = (
        c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
    )
    per_residue_loss_sum = 0.5 * (
        torch.nn.functional.pad(per_residue_loss_sum, (0, 1))
        + torch.nn.functional.pad(per_residue_loss_sum, (1, 0))
    )

    # Compute hard violations.
    violation_mask = torch.max(
        torch.stack(
            [c_n_violation_mask, ca_c_n_violation_mask, c_n_ca_violation_mask],
            dim=-2,
        ),
        dim=-2,
    )[0]
    violation_mask = torch.maximum(
        torch.nn.functional.pad(violation_mask, (0, 1)),
        torch.nn.functional.pad(violation_mask, (1, 0)),
    )

    return {
        "c_n_loss_mean": c_n_loss, #[B, ]
        "ca_c_n_loss_mean": ca_c_n_loss, #[B, ]
        "c_n_ca_loss_mean": c_n_ca_loss, #[B, ]
        "per_residue_loss_sum": per_residue_loss_sum, #[B, N-1]
        "per_residue_violation_mask": violation_mask, #[B, N-1]
    }

def between_residue_clash_loss(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_atom_radius: torch.Tensor,
    residue_index: torch.Tensor,
    res_mask: torch.Tensor,
    asym_id: Optional[torch.Tensor] = None,
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5,
    eps=1e-10,
) -> dict[str, torch.Tensor]:
    """Loss to penalize steric clashes between residues.

    This is a loss penalizing any steric clashes due to non bonded atoms in
    different peptides coming too close. This loss corresponds to the part with
    different residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
      atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
      atom14_atom_radius: Van der Waals radius for each atom.
      residue_index: Residue index for given amino acid.
      overlap_tolerance_soft: Soft tolerance factor.
      overlap_tolerance_hard: Hard tolerance factor.

    Returns:
      Dict containing:
        * 'mean_loss': average clash loss
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (N, 14)
    """
    fp_type = atom14_pred_positions.dtype
    # Create the distance matrix.
    # (*, N, N, 14, 14)
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Create the mask for valid distances.
    # shape (*, N, N, 14, 14)
    dists_mask = (
        atom14_atom_exists[..., :, None, :, None]
        * atom14_atom_exists[..., None, :, None, :]
    ).type(fp_type)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask = dists_mask * (
        residue_index[..., :, None, None, None] #[B, N, 1, 1, 1]
        < residue_index[..., None, :, None, None] #[B, 1, N, 1, 1]
    ) #[B, N, N, 1, 1] 

    # Backbone C-N bond between subsequent residues is no clash.
    # [14, ]
    c_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(2), num_classes=14
    ) # creates a one-hot vector of length 14 with index 2 set to 1
    # [1, 14]
    c_one_hot = c_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *c_one_hot.shape
    )
    c_one_hot = c_one_hot.type(fp_type)
    n_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(0), num_classes=14
    )
    # [1, 14]
    n_one_hot = n_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *n_one_hot.shape
    )
    n_one_hot = n_one_hot.type(fp_type)
    # [B, N, N] where row i and column i+1 are 1
    neighbour_mask = (residue_index[..., :, None] + 1) == residue_index[..., None, :]

    if asym_id is not None:
        neighbour_mask = neighbour_mask & (asym_id[..., :, None] == asym_id[..., None, :])
    # [B, N, N, 1, 1]
    neighbour_mask = neighbour_mask[..., None, None]

    c_n_bonds = (
        neighbour_mask
        * c_one_hot[..., None, None, :, None] #[1, 1, 1, 14, 1]
        * n_one_hot[..., None, None, None, :] #[1, 1, 1, 1, 14]
    ) # c_one_hot and n_one_hot form a 14x14 mask indicating C-N bonds; neighbour_mask selects adjacent residues
    dists_mask = dists_mask * (1.0 - c_n_bonds) # exclude adjacent residues

    # Disulfide bridge between two cysteines is no clash.
    cys = residue_constants.restype_name_to_atom14_names["CYS"]
    cys_sg_idx = cys.index("SG") # find index of 'SG' in atom14 list
    cys_sg_idx = residue_index.new_tensor(cys_sg_idx)
    cys_sg_idx = cys_sg_idx.reshape(
        *((1,) * len(residue_index.shape[:-1])), 1
    ).squeeze(-1) # reshape/squeeze to match batch dims
    cys_sg_one_hot = torch.nn.functional.one_hot(cys_sg_idx, num_classes=14)
    disulfide_bonds = (
        cys_sg_one_hot[..., None, None, :, None]
        * cys_sg_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (
        atom14_atom_radius[..., :, None, :, None]
        + atom14_atom_radius[..., None, :, None, :]
    ) # lower bound for clash distances

    # Compute the error.
    # For pocket_mask: any residue pair where either residue is in the pocket contributes to loss
    mask_i = res_mask.unsqueeze(2)          # [B, N, 1]
    mask_j = res_mask.unsqueeze(1)          # [B, 1, N]
    pair_res_mask = (mask_i | mask_j)[...,None,None]           # [B, N, N, 1 ,1]
    # shape (B, N, N, 14, 14)
    dists_mask = dists_mask * pair_res_mask
    dists_to_low_error = dists_mask * torch.nn.functional.relu(
        dists_lower_bound - overlap_tolerance_soft - dists
    )

    # Compute the mean loss.
    # shape (B,)
    mean_loss = torch.sum(dists_to_low_error, dim=(-4, -3, -2, -1)) / (1e-6 + torch.sum(dists_mask, dim=(-4, -3, -2, -1)))

    # Compute the per atom loss sum.
    # shape (B, N, 14)
    per_atom_loss_sum = torch.sum(dists_to_low_error, dim=(-4, -2)) + torch.sum(
        dists_to_low_error, dim=(-3, -1)
    )

    # Compute the hard clash mask.
    # shape (B, N, N, 14, 14)
    clash_mask = dists_mask * (
        dists < (dists_lower_bound - overlap_tolerance_hard)
    )

    per_atom_num_clash = torch.sum(clash_mask, dim=(-4, -2)) + torch.sum(clash_mask, dim=(-3, -1))

    # Compute the per atom clash.
    # shape (B, N, 14)
    per_atom_clash_mask = torch.maximum(
        torch.amax(clash_mask, dim=(-4, -2)),
        torch.amax(clash_mask, dim=(-3, -1)),
    )

    return {
        "mean_loss": mean_loss,  # shape (B)
        "per_atom_loss_sum": per_atom_loss_sum,  # shape (B, N, 14)
        "per_atom_clash_mask": per_atom_clash_mask,  # shape (B, N, 14)
        "per_atom_num_clash": per_atom_num_clash  # shape (B, N, 14)
    }


def within_residue_violations(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_dists_lower_bound: torch.Tensor,
    atom14_dists_upper_bound: torch.Tensor,
    res_mask:torch.Tensor,
    tighten_bounds_for_loss=0.0,
    eps=1e-10,
) -> dict[str, torch.Tensor]:
    """Loss to penalize steric clashes within residues.

    This is a loss penalizing any steric violations or clashes of non-bonded atoms
    in a given peptide. This loss corresponds to the part with
    the same residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
        atom14_pred_positions ([*, N, 14, 3]):
            Predicted positions of atoms in global prediction frame.
        atom14_atom_exists ([*, N, 14]):
            Mask denoting whether atom at positions exists for given
            amino acid type
        atom14_dists_lower_bound ([*, N, 14]):
            Lower bound on allowed distances.
        atom14_dists_upper_bound ([*, N, 14]):
            Upper bound on allowed distances
        tighten_bounds_for_loss ([*, N]):
            Extra factor to tighten loss

    Returns:
      Dict containing:
        * 'per_atom_loss_sum' ([*, N, 14]):
              sum of all clash losses per atom, shape
        * 'per_atom_clash_mask' ([*, N, 14]):
              mask whether atom clashes with any other atom shape
    """
    # Compute the mask for each residue.
    # [1, 14, 14]
    # Within each residue, mask out self-pairs (atom with itself)
    dists_masks = 1.0 - torch.eye(14, device=atom14_atom_exists.device)[None]
    # atom_14_exists: [B, N, 14]
    # [1, 1, 14, 14]
    dists_masks = dists_masks.reshape(
        *((1,) * len(atom14_atom_exists.shape[:-2])), *dists_masks.shape
    )

    # [B, N, 14, 14]
    dists_masks = (
        atom14_atom_exists[..., :, :, None] #[B, N, 14, 1]
        * atom14_atom_exists[..., :, None, :] #[B, N, 1, 14]
        * dists_masks #[1, 1, 14, 14] 
        * res_mask[...,None,None] #[B, N, 1, 1]
    )

    # Distance matrix [B, N, 14, 14]
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, :, None, :]
                - atom14_pred_positions[..., :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Compute the loss.
    dists_to_low_error = torch.nn.functional.relu(
        atom14_dists_lower_bound + tighten_bounds_for_loss - dists
    )
    dists_to_high_error = torch.nn.functional.relu(
        dists - (atom14_dists_upper_bound - tighten_bounds_for_loss)
    )
    # [B, N, 14, 14]
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    # Compute the per atom loss sum.
    # [B, N, 14]
    per_atom_loss_sum = torch.sum(loss, dim=-2) + torch.sum(loss, dim=-1)

    # Compute the violations mask.
    # [B, N, 14, 14]
    violations = dists_masks * (
        (dists < atom14_dists_lower_bound) | (dists > atom14_dists_upper_bound)
    )
    # [B, N, 14] 
    # Here we count the number of clashes
    per_atom_num_clash = torch.sum(violations, dim=-2) + torch.sum(violations, dim=-1)

    # Compute the per atom violations.
    per_atom_violations = torch.maximum(
        torch.max(violations, dim=-2)[0], torch.max(violations, axis=-1)[0]
    )

    return {
        "per_atom_loss_sum": per_atom_loss_sum,
        "per_atom_violations": per_atom_violations,
        "per_atom_num_clash": per_atom_num_clash
    }


def find_structural_violations(
    batch: dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    atom14_pred_mask:torch.Tensor,
    pred_aatype:torch.Tensor,
    diffuse_mask: torch.Tensor,
    sidechain_mask: torch.Tensor,
    pocket_mask: torch.Tensor,
    violation_tolerance_factor: float,
    clash_overlap_tolerance: float,
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Computes several checks for structural violations."""

    gen_mask = diffuse_mask | sidechain_mask
    pocket_sidechain_mask = pocket_mask & sidechain_mask
    # Compute between residue backbone violations of bonds and angles.
    connection_violations = between_residue_bond_loss(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=atom14_pred_mask,
        residue_index=batch["res_idx"],
        aatype=pred_aatype,
        res_mask = gen_mask,
        tolerance_factor_soft=violation_tolerance_factor,
        tolerance_factor_hard=violation_tolerance_factor,
    )
    ''' {
        "c_n_loss_mean": c_n_loss,
        "ca_c_n_loss_mean": ca_c_n_loss,
        "c_n_ca_loss_mean": c_n_ca_loss,
        "per_residue_loss_sum": per_residue_loss_sum,
        "per_residue_violation_mask": violation_mask,
    }
    '''

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # List of length 37
    atomtype_radius = [
        residue_constants.van_der_waals_radius[name[0]]
        for name in residue_constants.atom_types
    ] # Atom-type radii for atom37.
    
    # Create a tensor from the atomtype_radius list using the same device
    # and dtype as `atom14_pred_positions` (equivalent to
    # `torch.tensor(atomtype_radius, dtype=atom14_pred_positions.dtype, device=atom14_pred_positions.device)`).
    # [37]
    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)
    # Broadcast atom radii to atom14 per residue; fill zeros where atom doesn't exist.
    # [B, N, 14]
    atom14_atom_radius = (
        atom14_pred_mask # used for clash computation; must use generated atom14 atoms
        * atomtype_radius[batch["residx_atom14_to_atom37"]] # [B, N, 14]
    )

    # Compute the between residue clash loss.
    between_residue_clashes = between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_pred_mask,
        atom14_atom_radius=atom14_atom_radius,
        residue_index=batch["res_idx"],
        res_mask = gen_mask,
        overlap_tolerance_soft=clash_overlap_tolerance,
        overlap_tolerance_hard=clash_overlap_tolerance,
    )

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    # ['lower_bound', 'upper_bound', 'stddev']
    # [21, 14, 14]
    restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
        overlap_tolerance=clash_overlap_tolerance,
        bond_length_tolerance_factor=violation_tolerance_factor,
    )

    atom14_dists_lower_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["lower_bound"]
    )[pred_aatype]
    atom14_dists_upper_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["upper_bound"]
    )[pred_aatype]
    residue_violations = within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_pred_mask,
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        res_mask = sidechain_mask,
        tighten_bounds_for_loss=0.0,
    )

    pocket_peptide = pocket_peptide_loss(atom14_pred_positions, #[B, N, 14, 3]
                        atom14_pred_mask, # [B, N, 14]
                        batch["ligand_pos"], # [B, N_l, 3] ligand positions (batch["ligand_pos"])
                        batch["ligand_mask"], # [B, N_l] ligand mask (batch["ligand_mask"])
                        batch["ligand_elements"], # [B, N_l] ligand element types
                        atom14_atom_radius,
                        pocket_sidechain_mask, # [B, N]
                        overlap_tolerance_soft=clash_overlap_tolerance,
                        overlap_tolerance_hard=clash_overlap_tolerance,
                        eps = 1e-10
                        )

    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = torch.max(
        torch.stack(
            [
                connection_violations["per_residue_violation_mask"],
                torch.max(
                    between_residue_clashes["per_atom_clash_mask"], dim=-1
                )[0],
                torch.max(residue_violations["per_atom_violations"], dim=-1)[0],
            ],
            dim=-1,
        ),
        dim=-1,
    )[0]

    return {
        "between_residues": {
            "bonds_c_n_loss_mean": connection_violations["c_n_loss_mean"],  # (B)
            "angles_ca_c_n_loss_mean": connection_violations[
                "ca_c_n_loss_mean"
            ],  # (B)
            "angles_c_n_ca_loss_mean": connection_violations[
                "c_n_ca_loss_mean"
            ],  # (B)
            "connections_per_residue_loss_sum": connection_violations[
                "per_residue_loss_sum"
            ],  # (B, N)
            "connections_per_residue_violation_mask": connection_violations[
                "per_residue_violation_mask"
            ],  # (B, N)
            "clashes_mean_loss": between_residue_clashes["mean_loss"],  # (B, )
            "clashes_per_atom_loss_sum": between_residue_clashes[
                "per_atom_loss_sum"
            ],  # (B, N, 14)
            "clashes_per_atom_clash_mask": between_residue_clashes[
                "per_atom_clash_mask"
            ],  # (B, N, 14)
            "clashes_per_atom_num_clash": between_residue_clashes[
                "per_atom_num_clash"
            ],  # (B, N, 14)
        },
        "within_residues": {
            "per_atom_loss_sum": residue_violations[
                "per_atom_loss_sum"
            ],  # (B, N, 14)
            "per_atom_violations": residue_violations[
                "per_atom_violations"
            ],  # (B, N, 14),
            "per_atom_num_clash": residue_violations[
                "per_atom_num_clash"
            ],  # (B, N, 14)
        },
        "total_per_residue_violations_mask": per_residue_violations_mask,  # (N)
        "pocket_peptide": {
            "mean_loss": pocket_peptide["mean_loss"],  # shape (B)
            "per_atom_loss_sum": pocket_peptide["per_atom_loss_sum"],  # shape (B, N, 14)
            "per_atom_clash_mask": pocket_peptide["per_atom_clash_mask"],  # shape (B, N, 14)
            "per_atom_num_clash": pocket_peptide["per_atom_num_clash"]  # shape (B, N, 14)
        }
    }


def pocket_peptide_loss(atom14_pred_positions, #[B, N, 14, 3]
                        atom14_pred_mask, # [B, N, 14]
                        ligand_atom_pos, # [B, N_l, 3] batch["ligand_pos"]
                        ligand_atom_mask, # [B, N_l] batch["ligand_mask"]
                        ligand_elements, # [B, N_l]
                        atom14_atom_radius,
                        res_mask, # [B, N]
                        overlap_tolerance_soft,
                        overlap_tolerance_hard,
                        eps = 1e-10
                        ):
    B, N, _, _ = atom14_pred_positions.shape
    N_l = ligand_atom_pos.shape[1]
    device = atom14_pred_positions.device
    if N_l == 0:
        return {
            "mean_loss": torch.zeros(B, device=device, dtype=atom14_pred_positions.dtype),
            "per_atom_loss_sum": torch.zeros((B, N, 14), device=device, dtype=atom14_pred_positions.dtype),
            "per_atom_clash_mask": torch.zeros((B, N, 14), device=device, dtype=atom14_pred_positions.dtype),
            "per_atom_num_clash": torch.zeros((B, N, 14), device=device, dtype=atom14_pred_positions.dtype)
        }
    
    # [B, N_l]
    vdW = residue_constants.van_der_waals_lookup.to(ligand_elements.device)
    ligand_atom_radius = vdW[ligand_elements]
    # [B, N, 14, 1, 3] - [B, 1, 1, N_l, 3] = [B, N, 14, N_l, 3]
    diff = atom14_pred_positions.unsqueeze(-2) - ligand_atom_pos.unsqueeze(-3).unsqueeze(-3)
    # [B, N, 14, N_l]
    dists = torch.sqrt(eps+ torch.sum( diff** 2,dim=-1))
    # [B, N, 14, 1] * [B, 1, 1, N_l] * [B, N, 1, 1]= [B, N, 14, N_l]
    dists_mask = atom14_pred_mask[..., None] * ligand_atom_mask.unsqueeze(-2).unsqueeze(-2) * res_mask[...,None, None]
    # [B, N, 14, N_l]*([B, N, 14, 1]+[B, 1, 1, N_l]) = [B, N, 14, N_l]
    dists_lower_bound = dists_mask * (atom14_atom_radius.unsqueeze(-1) + ligand_atom_radius.unsqueeze(-2).unsqueeze(-2))

    dists_to_low_error = dists_mask * torch.nn.functional.relu(
        dists_lower_bound - overlap_tolerance_soft - dists
    )
    # [B, N, 14, N_l]
    dists_to_low_error = dists_mask * torch.nn.functional.relu(
        dists_lower_bound - overlap_tolerance_soft - dists
    )
    
    # [B, ]
    mean_loss = torch.sum(dists_to_low_error, dim=(-3, -2, -1)) / (1e-6 + torch.sum(dists_mask, dim=(-3, -2, -1)))


    per_atom_loss_sum = torch.sum(dists_to_low_error, dim=-1)
    # Compute the hard clash mask.
    # shape (B, N, 14, N_l)
    clash_mask = dists_mask * (
        dists < (dists_lower_bound - overlap_tolerance_hard)
    )
    # (B, N, 14)
    per_atom_num_clash = torch.sum(clash_mask, dim=-1)

    # Compute the per atom clash.
    # shape (B, N, 14)
    per_atom_clash_mask = torch.amax(clash_mask, dim=-1)

    return {
        "mean_loss": mean_loss,  # shape (B)
        "per_atom_loss_sum": per_atom_loss_sum,  # shape (B, N, 14)
        "per_atom_clash_mask": per_atom_clash_mask,  # shape (B, N, 14)
        "per_atom_num_clash": per_atom_num_clash  # shape (B, N, 14)
    }

def violation_loss(
    violations: dict[str, torch.Tensor],
    atom14_atom_exists: torch.Tensor, # [B, N, 14]
    res_mask: torch.Tensor,
    average_clashes: bool = False,
    eps=1e-6,
    **kwargs,
) -> torch.Tensor:
    
    #[B, N, 14]*[B, N, 1]
    atom14_atom_exists = atom14_atom_exists * res_mask[...,None]
    num_atoms = torch.sum(atom14_atom_exists, dim=(-1,-2))

    between_residue_mean_clash = violations["between_residues"]["clashes_mean_loss"]  # [B]
    within_residue_atom_clash = violations["within_residues"]["per_atom_loss_sum"] # [B, N, 14]

    if average_clashes:
        num_clash = (violations["between_residues"]["clashes_per_atom_num_clash"] +
                     violations["within_residues"]["per_atom_num_clash"])
        per_atom_clash = per_atom_clash / (num_clash + eps)

    within_residue_loss = torch.sum(within_residue_atom_clash, dim=(-1,-2)) / (eps + num_atoms) #[B, ]
    between_residue_loss = (
        violations["between_residues"]["bonds_c_n_loss_mean"] # [B,]
        + violations["between_residues"]["angles_ca_c_n_loss_mean"] #[B, ]
        + violations["between_residues"]["angles_c_n_ca_loss_mean"] # [B, ]
        + between_residue_mean_clash #[B,]
    )
    pocket_peptide_loss = violations["pocket_peptide"]["mean_loss"]

    # Average over the batch dimension
    return between_residue_loss, within_residue_loss, pocket_peptide_loss

def pairwise_distance_loss(gt_bb_atoms, pred_bb_atoms, num_batch, num_res, loss_mask):
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res*3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1) 
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res*3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1)

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res*3])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res*3])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists)**2 * pair_dist_mask,
            dim=(1, 2)) 
        dist_mat_loss /= (torch.sum(pair_dist_mask, dim=(1, 2)) + 1) 
        return dist_mat_loss

def bb_atom_loss(gt_bb_atoms, pred_bb_atoms, loss_mask, r3_norm_scale, training_cfg):
    gt_bb_atoms = gt_bb_atoms * training_cfg.bb_atom_scale / r3_norm_scale[..., None]
    pred_bb_atoms = pred_bb_atoms * training_cfg.bb_atom_scale / r3_norm_scale[..., None]
    loss_denom = torch.sum(loss_mask, dim=-1).clamp(min=1.) * 3
    bb_atom_loss = torch.sum(
        (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
        dim=(-1, -2, -3)
    ) / loss_denom
    return bb_atom_loss

def trans_vf_loss(gt_trans_1, pred_trans_1, loss_mask, r3_norm_scale, training_cfg):
    trans_error = (gt_trans_1 - pred_trans_1) / r3_norm_scale * training_cfg.trans_scale
    loss_denom = torch.sum(loss_mask, dim=-1).clamp(min=1.) * 3
    trans_vf_loss = training_cfg.translation_loss_weight * torch.sum(
        trans_error ** 2 * loss_mask[..., None],
        dim=(-1, -2)
    ) / loss_denom
    trans_vf_loss = torch.clamp(trans_vf_loss, max=5)
    return trans_vf_loss

def rots_vf_loss(gt_rot_vf, pred_rots_vf, loss_mask, so3_norm_scale, training_cfg):
    rots_vf_error = (gt_rot_vf - pred_rots_vf) / so3_norm_scale
    loss_denom = torch.sum(loss_mask, dim=-1).clamp(min=1.) * 3
    rots_vf_loss = training_cfg.rotation_loss_weights * torch.sum(
        rots_vf_error ** 2 * loss_mask[..., None],
        dim=(-1, -2)
    ) / loss_denom
    rots_vf_loss = torch.clamp(rots_vf_loss, max=5)
    return rots_vf_loss

def aa_ce_loss(pred_logits, aatypes_1, cat_norm_scale, loss_mask, num_batch, training_cfg, aatype_pred_num_tokens=21):

    aa_loss = F.cross_entropy(
        pred_logits.view(-1, aatype_pred_num_tokens), #[B, N_res, 21] -> [B*N_res, 21]
        aatypes_1.view(-1).long(), #[B, N_res] -> [B * N_res]
        reduction="none"
    ).reshape(num_batch, -1) / cat_norm_scale #[B, N_Res]
    aa_loss = torch.sum(aa_loss * loss_mask, dim = -1) / loss_mask.sum(dim = -1).clamp(min=1e-6)
    aa_loss = torch.clamp(aa_loss * training_cfg.aatypes_loss_weight, max=5)
    return aa_loss
