import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

def project_points(
    points_3d: torch.Tensor,
    K_cam: torch.Tensor,
    R_prior: torch.Tensor,
    t_prior: torch.Tensor,
    near_plane: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D points from world space into 2D camera coordinates.

    Args:
        points_3d (torch.Tensor): 3D coordinates, shape [B, N, 3], type torch.float32.
        K_cam (torch.Tensor): Camera intrinsic matrix, shape [B, 3, 3], type torch.float32.
        R_prior (torch.Tensor): Camera rotation prior matrix, shape [B, 3, 3], type torch.float32.
        t_prior (torch.Tensor): Camera translation prior, shape [B, 3] or [B, 3, 1], type torch.float32.
        near_plane (float): Near clipping plane (default: 0.1).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - proj_coords (torch.Tensor): Projected 2D coordinates in pixel space, shape [B, N, 2], type torch.float32.
            - proj_valid_mask (torch.Tensor): Boolean mask of valid projections (depth > near_plane), shape [B, N], type torch.bool.
    """
    B, N, _ = points_3d.shape
    
    # Force t_prior to shape [B, 3, 1]
    if t_prior.ndim == 2:
        t_prior = t_prior.unsqueeze(-1)
    elif t_prior.ndim == 3 and t_prior.shape[-1] != 1:
        t_prior = t_prior.view(B, 3, 1)
        
    # Transform points to camera coordinates:
    # points_cam = R_prior * points_3d^T + t_prior
    # R_prior: [B, 3, 3]
    # points_3d.transpose(1, 2): [B, 3, N]
    # t_prior: [B, 3, 1]
    # points_cam: [B, 3, N]
    points_cam = torch.bmm(R_prior, points_3d.transpose(1, 2)) + t_prior
    
    # Depth values: points_cam[:, 2, :] of shape [B, N]
    depth = points_cam[:, 2, :]
    
    # Boolean mask: depth > near_plane
    proj_valid_mask = depth > near_plane
    
    # Avoid division by zero/safe depth: replace values <= near_plane with near_plane
    depth_safe = torch.where(depth > near_plane, depth, torch.full_like(depth, near_plane))
    
    # Homogeneous projection: K_cam * points_cam
    # K_cam: [B, 3, 3]
    # points_cam: [B, 3, N]
    # projected: [B, 3, N]
    projected = torch.bmm(K_cam, points_cam)
    
    # Compute u, v coordinates
    u = projected[:, 0, :] / depth_safe
    v = projected[:, 1, :] / depth_safe
    
    # Output projection coordinates of shape [B, N, 2]
    proj_coords = torch.stack([u, v], dim=-1)
    
    return proj_coords, proj_valid_mask


class GeoStructuralCrossAttention(nn.Module):
    """
    Geo-Structural Cross-Attention (GSCA) module.
    Refines 2D visual descriptors by performing cross-attention with 3D keypoint descriptors,
    constrained by a geo-structural mask and stabilized to prevent NaNs in Softmax.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.q_proj = nn.Linear(channels, channels, bias=True)
        self.k_proj = nn.Linear(channels, channels, bias=True)
        self.v_proj = nn.Linear(channels, channels, bias=True)
        self.scale = math.sqrt(channels)

    def forward(
        self,
        feat_2d: torch.Tensor,
        feat_3d: torch.Tensor,
        coords_2d: torch.Tensor,
        proj_coords: torch.Tensor,
        proj_valid_mask: torch.Tensor,
        normals_2d: Optional[torch.Tensor] = None,
        normals_3d: Optional[torch.Tensor] = None,
        delta: float = 30.0,
        tau: float = 0.5
    ) -> torch.Tensor:
        """
        Forward pass for the GSCA module.

        Args:
            feat_2d (torch.Tensor): 2D dense/sampled descriptors, shape [B, HW, C], type torch.float32.
            feat_3d (torch.Tensor): 3D geometric descriptors, shape [B, N, C], type torch.float32.
            coords_2d (torch.Tensor): 2D coordinates (pixel space), shape [B, HW, 2], type torch.float32.
            proj_coords (torch.Tensor): Projected 3D coordinates (pixel space), shape [B, N, 2], type torch.float32.
            proj_valid_mask (torch.Tensor): Validity mask of 3D projections, shape [B, N], type torch.bool.
            normals_2d (torch.Tensor, optional): 2D unit normals, shape [B, HW, 3], type torch.float32.
            normals_3d (torch.Tensor, optional): 3D unit normals, shape [B, N, 3], type torch.float32.
            delta (float): 2D spatial distance threshold (default: 30.0).
            tau (float): Normal vector cosine similarity threshold (default: 0.5).

        Returns:
            torch.Tensor: Refined 2D descriptors, shape [B, HW, C], type torch.float32.
        """
        # Linear projections
        Q = self.q_proj(feat_2d)      # [B, HW, C]
        K = self.k_proj(feat_3d)      # [B, N, C]
        V = self.v_proj(feat_3d)      # [B, N, C]

        # Raw attention logits: Q * K^T / scale
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [B, HW, N]

        # 2D Spatial distance mask
        # dist[b, i, j] = L2 distance between coords_2d[b, i] and proj_coords[b, j]
        dist = torch.cdist(coords_2d, proj_coords, p=2.0)  # [B, HW, N]
        dist_mask = dist > delta                           # [B, HW, N]

        # Combine all boolean masks (invalid when dist > delta OR projection is invalid)
        # proj_valid_mask has shape [B, N], unsqueeze(1) -> [B, 1, N]
        invalid_mask = dist_mask | (~proj_valid_mask.unsqueeze(1))  # [B, HW, N]

        # Coplanaity / Normal alignment mask if normals are provided
        if normals_2d is not None and normals_3d is not None:
            n2d_norm = F.normalize(normals_2d, p=2.0, dim=-1)
            n3d_norm = F.normalize(normals_3d, p=2.0, dim=-1)
            cos_normal = torch.bmm(n2d_norm, n3d_norm.transpose(1, 2))  # [B, HW, N]
            normal_mask = cos_normal < tau                              # [B, HW, N]
            invalid_mask = invalid_mask | normal_mask

        # Create geo-structural mask with large negative value penalty
        m_geo = torch.zeros_like(attn_logits)
        m_geo[invalid_mask] = -1e9

        # Mitigate NaNs in Softmax: detect rows (queries) that are completely invalid
        all_invalid = invalid_mask.all(dim=-1)  # [B, HW]
        
        # Replace completely invalid rows in m_geo with 0.0 to prevent sum-to-zero / -inf in Softmax
        m_geo = torch.where(all_invalid.unsqueeze(-1), torch.zeros_like(m_geo), m_geo)

        # Softmax over keypoints dimension (dim=-1)
        attn_weights = torch.softmax(attn_logits + m_geo, dim=-1)  # [B, HW, N]

        # Output representation aggregation
        out = torch.bmm(attn_weights, V)  # [B, HW, C]

        # Zero out descriptors of completely invalid queries to ensure they are clean (all 0.0)
        out = out * (~all_invalid).unsqueeze(-1).float()

        return out


def compute_mnn_matches(
    feat_2d: torch.Tensor,
    feat_3d: torch.Tensor,
    sim_threshold: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Mutual Nearest Neighbors (MNN) matches between 2D and 3D descriptors.

    Args:
        feat_2d (torch.Tensor): Refined 2D descriptors, shape [HW, C], type torch.float32.
        feat_3d (torch.Tensor): 3D geometric descriptors, shape [N, C], type torch.float32.
        sim_threshold (float): Minimum cosine similarity threshold (default: 0.2).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - matches (torch.Tensor): Matching indices, shape [M, 2], type torch.int64.
                                      matches[:, 0] contains 2D indices, matches[:, 1] contains 3D indices.
            - match_scores (torch.Tensor): Cosine similarity scores for matches, shape [M], type torch.float32.
    """
    # L2 normalize features to calculate cosine similarity via matrix multiplication
    feat_2d_norm = F.normalize(feat_2d, p=2, dim=-1)
    feat_3d_norm = F.normalize(feat_3d, p=2, dim=-1)

    # Similarity matrix: [HW, N]
    sim_matrix = torch.mm(feat_2d_norm, feat_3d_norm.transpose(0, 1))

    # Mutual Nearest Neighbor check:
    # argmax from 2D to 3D
    nn_2d_to_3d = torch.argmax(sim_matrix, dim=1)  # [HW]
    # argmax from 3D to 2D
    nn_3d_to_2d = torch.argmax(sim_matrix, dim=0)  # [N]

    # Reciprocity condition:
    # A pixel i matches point j if j is the nearest neighbor of i, AND i is the nearest neighbor of j.
    hw = feat_2d.shape[0]
    indices_2d = torch.arange(hw, device=feat_2d.device)
    mnn_mask = nn_3d_to_2d[nn_2d_to_3d] == indices_2d  # [HW]

    # Similarity threshold condition:
    # Cosine similarity must be at least sim_threshold
    best_sim = sim_matrix[indices_2d, nn_2d_to_3d]  # [HW]
    final_mask = mnn_mask & (best_sim >= sim_threshold)

    # Filtered indices
    matches_2d = indices_2d[final_mask]
    matches_3d = nn_2d_to_3d[final_mask]

    # Stack to get [M, 2]
    matches = torch.stack([matches_2d, matches_3d], dim=1)
    match_scores = best_sim[final_mask]

    return matches, match_scores
