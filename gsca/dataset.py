import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Optional

from gsca.validation import degrade_point_cloud, apply_visual_degradations, synthesize_pose_prior
from gsca.models.gsca_matcher import project_points


class GSCADataset(Dataset):
    """
    Custom Dataset for the GSCA project.
    Loads RGB images, normal maps, 2D coordinates, 3D point clouds,
    camera intrinsics, and ground truth poses.
    
    Optionally applies point cloud and visual degradations for Sim-to-Real training.
    """

    def __init__(
        self,
        sample_paths: List[str],
        degrade_points: bool = False,
        degrade_visual: bool = False,
        noise_std: float = 0.03,
        downsample_ratio: float = 0.5,
        roughness_factor: float = 0.2,
        sun_azimuth_range: Tuple[float, float] = (0.0, 360.0),
        sun_elevation_range: Tuple[float, float] = (15.0, 75.0),
        pixel_match_threshold: float = 3.0,
        near_plane: float = 0.1,
    ):
        """
        Args:
            sample_paths: List of file paths to PyTorch saved dictionaries (.pt).
            degrade_points: Whether to apply point cloud downsampling and noise.
            degrade_visual: Whether to apply visual solar illumination and roughness.
            noise_std: Std dev of noise for point cloud degradation.
            downsample_ratio: Ratio of points to keep in point cloud degradation.
            roughness_factor: Roughness multiplier for visual degradation.
            sun_azimuth_range: Range for random sun azimuth (degrees).
            sun_elevation_range: Range for random sun elevation (degrees).
            pixel_match_threshold: Pixel distance tolerance to compute matches (gt_mask).
            near_plane: Near plane clipping distance for point projection.
        """
        self.sample_paths = sample_paths
        self.degrade_points = degrade_points
        self.degrade_visual = degrade_visual
        self.noise_std = noise_std
        self.downsample_ratio = downsample_ratio
        self.roughness_factor = roughness_factor
        self.sun_azimuth_range = sun_azimuth_range
        self.sun_elevation_range = sun_elevation_range
        self.pixel_match_threshold = pixel_match_threshold
        self.near_plane = near_plane

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.sample_paths[idx]
        
        # Load sample dictionary
        # Expected keys: 'image', 'normal_map', 'pos', 'normals_3d', 'K_cam', 'pose_gt', 'coords_pixel_2d', 'normals_2d'
        data = torch.load(path, map_location="cpu")
        
        image = data['image'].float()             # [3, H, W] in [0, 1]
        normal_map = data['normal_map'].float()   # [3, H, W] in [-1, 1]
        pos = data['pos'].float()                 # [N_3D, 3]
        normals_3d = data['normals_3d'].float()   # [N_3D, 3]
        K_cam = data['K_cam'].float()             # [3, 3]
        pose_gt = data['pose_gt'].float()         # [4, 4]
        coords_pixel_2d = data['coords_pixel_2d'].float() # [N_2D, 2]
        normals_2d = data['normals_2d'].float()   # [N_2D, 3]
        
        H, W = image.shape[1], image.shape[2]

        # 1. Apply Point Cloud Degradation
        if self.degrade_points:
            # degrade_point_cloud expects [B, N, 3]
            pos_batch = pos.unsqueeze(0)
            pos_degraded = degrade_point_cloud(
                pos_batch, 
                noise_std=self.noise_std, 
                downsample_ratio=self.downsample_ratio
            ).squeeze(0)
            
            # Since point cloud is sub-sampled, sub-sample 3D normals accordingly
            B, N, _ = pos_batch.shape
            N_degraded = pos_degraded.shape[0]
            # Match random indices used in downsampling
            torch.manual_seed(idx)  # seed matching for consistent downsampling indices
            indices = torch.randperm(N)[:N_degraded]
            normals_3d = normals_3d[indices]
            pos = pos_degraded

        # 2. Apply Visual Degradation
        if self.degrade_visual:
            # Generate random sun azimuth and elevation
            azimuth = torch.FloatTensor(1).uniform_(*self.sun_azimuth_range)
            elevation = torch.FloatTensor(1).uniform_(*self.sun_elevation_range)
            
            # apply_visual_degradations expects [B, 3, H, W] and sun angles as [B]
            image_batch = image.unsqueeze(0)
            normal_map_batch = normal_map.unsqueeze(0)
            
            # Note: We do NOT pass albedo_map as we do not receive it (fallback uses image)
            image_degraded = apply_visual_degradations(
                image=image_batch,
                normal_map=normal_map_batch,
                albedo_map=None,
                sun_azimuth=azimuth,
                sun_elevation=elevation,
                roughness_factor=self.roughness_factor
            ).squeeze(0)
            image = image_degraded

        # 3. Project 3D points and compute Ground Truth Match Mask (gt_mask)
        # project_points expects: points_3d: [B, N, 3], K_cam: [B, 3, 3], R_prior/t_prior: [B, 3, 3]/[B, 3, 1]
        R_gt = pose_gt[:3, :3].unsqueeze(0)
        t_gt = pose_gt[:3, 3].unsqueeze(0).unsqueeze(-1)
        
        proj_coords, proj_valid_mask = project_points(
            points_3d=pos.unsqueeze(0),
            K_cam=K_cam.unsqueeze(0),
            R_prior=R_gt,
            t_prior=t_gt,
            near_plane=self.near_plane
        )
        proj_coords = proj_coords.squeeze(0)          # [N_3D, 2]
        proj_valid_mask = proj_valid_mask.squeeze(0)  # [N_3D]
        
        # Calculate pairwise distances: [N_2D, N_3D]
        dists = torch.cdist(coords_pixel_2d.unsqueeze(0), proj_coords.unsqueeze(0), p=2.0).squeeze(0)
        gt_mask = dists < self.pixel_match_threshold
        
        # Mask out coordinates projecting behind camera
        gt_mask = gt_mask & proj_valid_mask.unsqueeze(0)

        # 4. Normalize 2D keypoint coordinates to [-1.0, 1.0] for grid_sample
        u_norm = (coords_pixel_2d[:, 0] / (W - 1)) * 2.0 - 1.0
        v_norm = (coords_pixel_2d[:, 1] / (H - 1)) * 2.0 - 1.0
        coords_2d = torch.stack([u_norm, v_norm], dim=-1)

        return {
            'image': image,
            'coords_2d': coords_2d,
            'pos': pos,
            'normals_2d': normals_2d,
            'normals_3d': normals_3d,
            'K_cam': K_cam,
            'pose_gt': pose_gt,
            'gt_mask': gt_mask
        }


def gsca_collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function to assemble batched dictionaries for training.
    Specifically handles PyTorch Geometric's sparse concatenation format for 3D points
    and pads the gt_mask to support varying numbers of 3D points in the batch.
    """
    B = len(samples)
    
    # 1. Stack standardized tensors
    images = torch.stack([s['image'] for s in samples], dim=0)
    coords_2d = torch.stack([s['coords_2d'] for s in samples], dim=0)
    normals_2d = torch.stack([s['normals_2d'] for s in samples], dim=0)
    K_cam = torch.stack([s['K_cam'] for s in samples], dim=0)
    pose_gt = torch.stack([s['pose_gt'] for s in samples], dim=0)

    # 2. Sparse PyG representation of point clouds
    pos = torch.cat([s['pos'] for s in samples], dim=0)
    normals_3d = torch.cat([s['normals_3d'] for s in samples], dim=0)
    
    batch_indices = torch.cat([
        torch.full((s['pos'].shape[0],), idx, dtype=torch.long)
        for idx, s in enumerate(samples)
    ], dim=0)

    # 3. Pose prior synthesis (for the batch)
    pose_prior = synthesize_pose_prior(pose_gt, max_trans=1.0, max_rot_deg=5.0)
    R_prior = pose_prior[:, :3, :3]
    t_prior = pose_prior[:, :3, 3:4]

    # 4. Pad ground truth masks to the maximum number of 3D nodes in the batch
    max_nodes = max(s['pos'].shape[0] for s in samples)
    N_2D = samples[0]['coords_2d'].shape[0]
    
    gt_mask_padded = torch.zeros(B, N_2D, max_nodes, dtype=torch.bool)
    for idx, s in enumerate(samples):
        num_nodes = s['pos'].shape[0]
        gt_mask_padded[idx, :, :num_nodes] = s['gt_mask']

    return {
        'image': images,
        'coords_2d': coords_2d,
        'pos': pos,
        'batch': batch_indices,
        'K_cam': K_cam,
        'R_prior': R_prior,
        't_prior': t_prior,
        'normals_2d': normals_2d,
        'normals_3d': normals_3d,
        'pose_gt': pose_gt,
        'gt_mask': gt_mask_padded
    }


def get_gsca_dataloader(
    dataset: GSCADataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    """
    Returns a configured PyTorch DataLoader utilizing the custom GSCA collate function.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=gsca_collate_fn,
        pin_memory=pin_memory
    )
