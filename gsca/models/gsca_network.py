import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from torch_geometric.utils import to_dense_batch

from .visual_branch import Visual2DBranch
from .geometric import GeometricFeatureExtractor
from .gsca_matcher import project_points, GeoStructuralCrossAttention


class GSCANetwork(nn.Module):
    """
    Unified Geo-Structural Cross-Attention (GSCA) Network.
    Coordinates the 2D Visual Branch, 3D Geometric Branch, and the Cross-Attention matcher.
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vits14",
        bottleneck_dim: int = 64,
        intermediate_layers: List[int] = [3, 6, 9, 12],
        out_channels: int = 256,
        k: int = 20,
        pretrained: bool = True,
    ):
        super().__init__()
        # 1. Visual 2D Branch (PEFT adaptors + FPN)
        self.visual_branch = Visual2DBranch(
            backbone_name=backbone_name,
            bottleneck_dim=bottleneck_dim,
            intermediate_layers=intermediate_layers,
            out_dim=out_channels,
            pretrained=pretrained,
        )

        # 2. Geometric 3D Branch (DGCNN with dynamic EdgeConv layers)
        self.geometric_branch = GeometricFeatureExtractor(
            k=k,
            out_channels=out_channels,
        )

        # 3. Geo-Structural Cross-Attention Module
        self.cross_attention = GeoStructuralCrossAttention(
            channels=out_channels,
        )

    def forward(
        self,
        images: torch.Tensor,
        coords_2d: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        K_cam: torch.Tensor,
        R_prior: torch.Tensor,
        t_prior: torch.Tensor,
        normals_2d: torch.Tensor,
        normals_3d: torch.Tensor,
        delta: float = 30.0,
        tau: float = 0.5,
        near_plane: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the unified GSCA network.

        Args:
            images: [B, 3, H, W] Visual RGB images.
            coords_2d: [B, N_2D, 2] 2D coordinates in range [-1.0, 1.0].
            pos: [N_total, 3] 3D point cloud coordinate positions.
            batch: [N_total] Batch index for each point in pos.
            K_cam: [B, 3, 3] Camera intrinsic matrices.
            R_prior: [B, 3, 3] Rotation matrices of the camera pose prior.
            t_prior: [B, 3, 1] or [B, 3] Translation vectors of the camera prior.
            normals_2d: [B, N_2D, 3] Normals associated with 2D coordinates.
            normals_3d: [N_total, 3] Normals associated with 3D points.
            delta: Distance threshold for cross-attention masking.
            tau: Normal compatibility cosine threshold.
            near_plane: Minimum camera depth.

        Returns:
            feat_2d_refined: [B, N_2D, C] Cross-attended visual descriptors.
            feat_3d_dense: [B, N_3D_max, C] Batched geometric descriptors.
        """
        # --- 1. Extract 2D visual features ---
        # dense_descriptors: [B, C, H, W], feat_2d: [B, N_2D, C]
        _, feat_2d = self.visual_branch(images, coords_2d)

        # --- 2. Extract 3D geometric features ---
        # feat_3d_sparse: [N_total, C]
        feat_3d_sparse = self.geometric_branch(pos, batch)

        # --- 3. Batch/dense representation of sparse 3D structures ---
        # feat_3d_dense: [B, max_nodes, C], mask_3d: [B, max_nodes]
        feat_3d_dense, mask_3d = to_dense_batch(feat_3d_sparse, batch)
        pos_dense, _ = to_dense_batch(pos, batch)
        normals_3d_dense, _ = to_dense_batch(normals_3d, batch)

        # --- 4. Projection of 3D points into 2D camera coordinates ---
        # proj_coords: [B, max_nodes, 2], proj_valid_mask: [B, max_nodes]
        proj_coords, proj_valid_mask = project_points(
            pos_dense, K_cam, R_prior, t_prior, near_plane=near_plane
        )

        # Combine projection validity with node existence mask
        proj_valid_mask_combined = proj_valid_mask & mask_3d

        # --- 5. Geo-Structural Cross-Attention ---
        feat_2d_refined = self.cross_attention(
            feat_2d=feat_2d,
            feat_3d=feat_3d_dense,
            coords_2d=coords_2d,
            proj_coords=proj_coords,
            proj_valid_mask=proj_valid_mask_combined,
            normals_2d=normals_2d,
            normals_3d=normals_3d_dense,
            delta=delta,
            tau=tau,
        )

        return feat_2d_refined, feat_3d_dense
