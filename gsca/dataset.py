import math
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import torchvision.transforms.functional as TF

from gsca.validation import apply_visual_degradations, degrade_point_cloud, synthesize_pose_prior
from gsca.models.gsca_matcher import project_points


class GSCADataset(Dataset):
    """
    Custom Dataset for the GSCA project.
    Reads RGB images from data/rgb/ and JSON metadata from data/metadata/.
    Also loads the 3D model vertices from data/Original.fbx to get the point cloud.
    
    Optionally applies visual and point cloud degradations for Sim-to-Real training.
    """

    def __init__(
        self,
        sample_paths: List[str] = None,
        data_dir: Optional[str] = None,
        degrade_points: bool = False,
        degrade_visual: bool = False,
        noise_std: float = 0.03,
        downsample_ratio: float = 0.5,
        roughness_factor: float = 0.2,
        sun_azimuth_range: Tuple[float, float] = (0.0, 360.0),
        sun_elevation_range: Tuple[float, float] = (15.0, 75.0),
        pixel_match_threshold: float = 3.0,
        near_plane: float = 0.1,
        fbx_path: str = "data/Original.fbx",
        num_points: int = 2048,
    ):
        self.degrade_points = degrade_points
        self.degrade_visual = degrade_visual
        self.noise_std = noise_std
        self.downsample_ratio = downsample_ratio
        self.roughness_factor = roughness_factor
        self.sun_azimuth_range = sun_azimuth_range
        self.sun_elevation_range = sun_elevation_range
        self.pixel_match_threshold = pixel_match_threshold
        self.near_plane = near_plane
        self.num_points = num_points

        if sample_paths is not None and len(sample_paths) > 0:
            self.sample_paths = sample_paths
        elif data_dir is not None:
            rgb_dir = os.path.join(data_dir, 'rgb')
            self.sample_paths = [os.path.join(rgb_dir, f) for f in sorted(os.listdir(rgb_dir)) if f.endswith('.png')]
        else:
            raise ValueError("Must provide either sample_paths or data_dir.")

        # Load FBX model if we have at least one .png sample
        # Resolve FBX path relative to dataset.py if it's not absolute
        if any(p.endswith('.png') for p in self.sample_paths):
            actual_fbx_path = fbx_path
            if not os.path.isabs(actual_fbx_path):
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                actual_fbx_path = os.path.join(project_root, actual_fbx_path)
            
            if os.path.exists(actual_fbx_path):
                from fbxloader import FBXLoader
                import numpy as np
                loader = FBXLoader(actual_fbx_path)
                mesh = loader.export_trimesh()
                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                
                # Simple random downsampling to self.num_points
                if len(vertices) > self.num_points:
                    np.random.seed(42)
                    indices = np.random.choice(len(vertices), self.num_points, replace=False)
                    vertices = vertices[indices]
                # Scale model to 1% (0.01) and invert the X axis to match OpenGL/OpenCV right-handed coordinates
                scaled_vertices = torch.from_numpy(vertices) * 0.01
                scaled_vertices[:, 0] *= -1.0
                
                # Unity's Bounds.center is (min + max) / 2.0
                bbox_center = (scaled_vertices.min(dim=0)[0] + scaled_vertices.max(dim=0)[0]) / 2.0
                scaled_vertices -= bbox_center
                
                # Apply computed pre-alignment matrix to match Unity world space JSON cameras
                R_align = torch.tensor([
                    [-0.39660221338272095, -0.8784772157669067, -0.2664293050765991], 
                    [0.4707551598548889, -0.44379132986068726, 0.7625213861465454], 
                    [-0.7880966663360596, 0.1769946664571762, 0.5895562171936035]
                ], dtype=torch.float32)
                t_align = torch.tensor([-0.18210142850875854, 1.1812915802001953, 0.9816922545433044], dtype=torch.float32)
                
                scaled_vertices = torch.matmul(scaled_vertices, R_align) + t_align
                self.pos = scaled_vertices
            else:
                print(f"Warning: FBX model not found at {actual_fbx_path}. Falling back to random points.")
                self.pos = torch.randn(self.num_points, 3)
        else:
            self.pos = None

    def __len__(self) -> int:
        return len(self.sample_paths)

    def _quat_to_unity_euler(self, x: float, y: float, z: float, w: float):
        """Extracts Pitch, Yaw, Roll from Unity quaternion (ZXY order)."""
        sqw, sqx, sqy, sqz = w*w, x*x, y*y, z*z
        unit = sqx + sqy + sqz + sqw
        test = x*w - y*z
        
        if test > 0.4995 * unit:
            pitch = math.pi / 2
            yaw = 2 * math.atan2(y, x)
            roll = 0
        elif test < -0.4995 * unit:
            pitch = -math.pi / 2
            yaw = -2 * math.atan2(y, x)
            roll = 0
        else:
            pitch = math.asin(2.0 * (w*x - y*z))
            yaw = math.atan2(2.0*w*y + 2.0*z*x, 1 - 2.0*(x*x + y*y))
            roll = math.atan2(2.0*w*z + 2.0*x*y, 1 - 2.0*(z*z + x*x))
        return pitch, yaw, roll

    def _get_opengl_matrices(self, rx, ry, rz, rw, px, py, pz):
        """Applies exact markdown transformations for Unity -> OpenGL."""
        pitch, yaw, roll = self._quat_to_unity_euler(rx, ry, rz, rw)
        
        cam_pitch = pitch
        cam_yaw = -yaw
        cam_roll = roll
        
        # Rx
        Rx = torch.tensor([
            [1, 0, 0],
            [0, math.cos(cam_pitch), -math.sin(cam_pitch)],
            [0, math.sin(cam_pitch), math.cos(cam_pitch)]
        ], dtype=torch.float32)
        # Ry
        Ry = torch.tensor([
            [math.cos(cam_yaw), 0, math.sin(cam_yaw)],
            [0, 1, 0],
            [-math.sin(cam_yaw), 0, math.cos(cam_yaw)]
        ], dtype=torch.float32)
        # Rz
        Rz = torch.tensor([
            [math.cos(cam_roll), -math.sin(cam_roll), 0],
            [math.sin(cam_roll), math.cos(cam_roll), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # Composición ZYX: Rz * Ry * Rx
        R_c2w_gl = torch.matmul(Rz, torch.matmul(Ry, Rx))
        t_c2w_gl = torch.tensor([-px * 0.01, py * 0.01, pz * 0.01], dtype=torch.float32).unsqueeze(1) # position inversion and scale by 0.01
        
        R_w2c_gl = R_c2w_gl.t()
        t_w2c_gl = -torch.matmul(R_w2c_gl, t_c2w_gl)
        
        # Convertir OpenGL (Y-up, Z-back) a OpenCV (Y-down, Z-forward)
        S_gl2cv = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32)
        R_w2c_cv = torch.matmul(S_gl2cv, R_w2c_gl)
        t_w2c_cv = torch.matmul(S_gl2cv, t_w2c_gl)
        
        return R_w2c_cv, t_w2c_cv.squeeze(1)

    def _quat_to_matrix(self, x: float, y: float, z: float, w: float) -> torch.Tensor:
        """Converts a quaternion into a 3x3 rotation matrix."""
        x2, y2, z2 = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        return torch.tensor([
            [1 - 2*y2 - 2*z2, 2*xy - 2*wz,     2*xz + 2*wy],
            [2*xy + 2*wz,     1 - 2*x2 - 2*z2, 2*yz - 2*wx],
            [2*xz - 2*wy,     2*yz + 2*wx,     1 - 2*x2 - 2*y2]
        ], dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.sample_paths[idx]
        
        # Legacy .pt format compatibility (for pytest and old datasets)
        if img_path.endswith('.pt'):
            data = torch.load(img_path, weights_only=False)
            image = data['image'].float()
            pos = data['pos'].float()
            normals_3d = data.get('normals_3d')
            if normals_3d is not None:
                normals_3d = normals_3d.float()
            normals_2d = data.get('normals_2d')
            if normals_2d is not None:
                normals_2d = normals_2d.float()
            
            # Apply Point Cloud Degradation
            if self.degrade_points:
                torch.manual_seed(idx)
                pos_batch = pos.unsqueeze(0)
                pos_degraded = degrade_point_cloud(
                    pos_batch, 
                    noise_std=self.noise_std, 
                    downsample_ratio=self.downsample_ratio
                ).squeeze(0)
                
                N_degraded = pos_degraded.shape[0]
                indices = torch.randperm(pos.shape[0])[:N_degraded]
                if normals_3d is not None:
                    normals_3d = normals_3d[indices]
                pos = pos_degraded
                
            K_cam = data['K_cam'].float()
            pose_gt = data['pose_gt'].float()
            coords_pixel_2d = data['coords_pixel_2d'].float()
            
            pose_prior = synthesize_pose_prior(pose_gt.unsqueeze(0), max_trans=1.0, max_rot_deg=5.0).squeeze(0)
            R_prior = pose_prior[:3, :3]
            t_prior = pose_prior[:3, 3:4]
            sun_direction = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
            
            H, W = image.shape[1], image.shape[2]
            u_norm = (coords_pixel_2d[:, 0] / (W - 1)) * 2.0 - 1.0
            v_norm = (coords_pixel_2d[:, 1] / (H - 1)) * 2.0 - 1.0
            coords_2d = torch.stack([u_norm, v_norm], dim=-1)
            
            # Project 3D points
            R_gt = pose_gt[:3, :3].unsqueeze(0)
            t_gt = pose_gt[:3, 3].unsqueeze(0).unsqueeze(-1)
            
            proj_coords, proj_valid_mask = project_points(
                points_3d=pos.unsqueeze(0),
                K_cam=K_cam.unsqueeze(0),
                R_prior=R_gt,
                t_prior=t_gt,
                near_plane=self.near_plane
            )
            proj_coords = proj_coords.squeeze(0)
            proj_valid_mask = proj_valid_mask.squeeze(0)
            
            dists = torch.cdist(coords_pixel_2d.unsqueeze(0), proj_coords.unsqueeze(0), p=2.0).squeeze(0)
            gt_mask = dists < self.pixel_match_threshold
            gt_mask = gt_mask & proj_valid_mask.unsqueeze(0)
            
            return {
                'image': image,
                'coords_2d': coords_2d,
                'pos': pos,
                'normals_3d': normals_3d,
                'normals_2d': normals_2d,
                'K_cam': K_cam,
                'pose_gt': pose_gt,
                'R_prior': R_prior,
                't_prior': t_prior,
                'sun_direction': sun_direction,
                'gt_mask': gt_mask
            }
            
        filename = os.path.basename(img_path)
        meta_name = filename.replace('.png', '.json')
        base_dir = os.path.dirname(os.path.dirname(img_path))
        meta_path = os.path.join(base_dir, 'metadata', meta_name)
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
        img = Image.open(img_path).convert('RGB')
        image = TF.to_tensor(img)
        
        if self.degrade_visual:
            azimuth = torch.FloatTensor(1).uniform_(*self.sun_azimuth_range)
            elevation = torch.FloatTensor(1).uniform_(*self.sun_elevation_range)
            
            image_batch = image.unsqueeze(0)
            image_degraded = apply_visual_degradations(
                image=image_batch,
                albedo_map=None,
                sun_azimuth=azimuth,
                sun_elevation=elevation,
                roughness_factor=self.roughness_factor
            ).squeeze(0)
            image = image_degraded
            
        # Conversion matrices for Unity to OpenCV Camera Coordinate
        rx, ry, rz, rw = meta['rx_gt'], meta['ry_gt'], meta['rz_gt'], meta['rw_gt']
        px, py, pz = meta['px_gt'], meta['py_gt'], meta['pz_gt']
        R_w2c_gt, t_w2c_gt = self._get_opengl_matrices(rx, ry, rz, rw, px, py, pz)
        
        pose_gt = torch.eye(4, dtype=torch.float32)
        pose_gt[:3, :3] = R_w2c_gt
        pose_gt[:3, 3] = t_w2c_gt
        
        # Construct pose_prior (Unity to OpenCV Camera Coordinate Conversion)
        rx_p, ry_p, rz_p, rw_p = meta['rx_prior'], meta['ry_prior'], meta['rz_prior'], meta['rw_prior']
        px_p, py_p, pz_p = meta['px_prior'], meta['py_prior'], meta['pz_prior']
        R_w2c_prior, t_w2c_prior = self._get_opengl_matrices(rx_p, ry_p, rz_p, rw_p, px_p, py_p, pz_p)
        
        pose_prior = torch.eye(4, dtype=torch.float32)
        pose_prior[:3, :3] = R_w2c_prior
        pose_prior[:3, 3] = t_w2c_prior
        
        # Construct K_cam
        focal_length = meta['focal_length']
        cx = meta['cx']
        cy = meta['cy']
        K_cam = torch.tensor([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        # Invert World X for sun direction as well
        sun_direction = torch.tensor([
            -meta['sun_direction']['x'],
            meta['sun_direction']['y'],
            meta['sun_direction']['z']
        ], dtype=torch.float32)
        
        # Load object 3D point cloud
        pos = self.pos if self.pos is not None else torch.randn(self.num_points, 3)
        
        # 1. Apply Point Cloud Degradation
        if self.degrade_points:
            pos_batch = pos.unsqueeze(0)
            pos_degraded = degrade_point_cloud(
                pos_batch, 
                noise_std=self.noise_std, 
                downsample_ratio=self.downsample_ratio
            ).squeeze(0)
            pos = pos_degraded
            
        # 1.5. Apply Frustum Culling based on prior pose (saves massive VRAM)
        # Project using the prior pose to see which points might land on screen
        proj_coords_prior, proj_valid_mask_prior = project_points(
            points_3d=pos.unsqueeze(0),
            K_cam=K_cam.unsqueeze(0),
            R_prior=R_w2c_prior.unsqueeze(0),
            t_prior=t_w2c_prior.unsqueeze(0),
            near_plane=self.near_plane
        )
        proj_coords_prior = proj_coords_prior.squeeze(0)
        proj_valid_mask_prior = proj_valid_mask_prior.squeeze(0)
        
        # Keep points that are valid (in front of camera) and within a generous margin of the image
        H, W = image.shape[1], image.shape[2]
        margin_x = W * 0.5  # generous 50% margin for pose error
        margin_y = H * 0.5
        in_frustum_mask = (
            proj_valid_mask_prior &
            (proj_coords_prior[:, 0] > -margin_x) & (proj_coords_prior[:, 0] < W + margin_x) &
            (proj_coords_prior[:, 1] > -margin_y) & (proj_coords_prior[:, 1] < H + margin_y)
        )
        
        # Filter points (if mask is totally empty or too small, keep at least 20 points to avoid BatchNorm crashes in DGCNN)
        min_points_required = 20
        if in_frustum_mask.sum() >= min_points_required:
            pos = pos[in_frustum_mask]
        else:
            # Fallback: keep the first 20 points if not enough points fall in frustum
            # This prevents the [1, C] error in BatchNorm1d
            fallback_count = min(min_points_required, pos.shape[0])
            pos = pos[:fallback_count]
            
        # 2. Project 3D points to image plane to get 2D keypoints and compute gt_mask
        R_gt_batch = R_w2c_gt.unsqueeze(0)
        t_gt_batch = t_w2c_gt.unsqueeze(0)
        
        proj_coords, proj_valid_mask = project_points(
            points_3d=pos.unsqueeze(0),
            K_cam=K_cam.unsqueeze(0),
            R_prior=R_gt_batch,
            t_prior=t_gt_batch,
            near_plane=self.near_plane
        )
        proj_coords = proj_coords.squeeze(0)          # [N_3D, 2]
        proj_valid_mask = proj_valid_mask.squeeze(0)  # [N_3D]
        
        # Calculate pairwise distances (L2) and ground truth match mask
        dists = torch.cdist(proj_coords.unsqueeze(0), proj_coords.unsqueeze(0), p=2.0).squeeze(0)
        gt_mask = dists < self.pixel_match_threshold
        gt_mask = gt_mask & proj_valid_mask.unsqueeze(0)
        
        # Normalize coordinates to [-1.0, 1.0] for bilinear sampling
        H, W = image.shape[1], image.shape[2]
        u_norm = (proj_coords[:, 0] / (W - 1)) * 2.0 - 1.0
        v_norm = (proj_coords[:, 1] / (H - 1)) * 2.0 - 1.0
        coords_2d = torch.stack([u_norm, v_norm], dim=-1)
        
        return {
            'image': image,
            'coords_2d': coords_2d,
            'pos': pos,
            'K_cam': K_cam,
            'pose_gt': pose_gt,
            'R_prior': R_w2c_prior,
            't_prior': t_w2c_prior,
            'sun_direction': sun_direction,
            'gt_mask': gt_mask
        }


def gsca_collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function to assemble batched dictionaries for training.
    Specifically handles PyTorch Geometric's sparse concatenation format for 3D points
    and pads the gt_mask to support varying numbers of 3D points in the batch.
    """
    B = len(samples)
    
    images = torch.stack([s['image'] for s in samples], dim=0)
    max_nodes = max(s['pos'].shape[0] for s in samples)
    
    # Pad coords_2d to [B, max_nodes, 2]
    coords_2d_padded = torch.zeros(B, max_nodes, 2, dtype=torch.float32)
    for idx, s in enumerate(samples):
        num_nodes = s['coords_2d'].shape[0]
        coords_2d_padded[idx, :num_nodes] = s['coords_2d']
        
    K_cam = torch.stack([s['K_cam'] for s in samples], dim=0)
    pose_gt = torch.stack([s['pose_gt'] for s in samples], dim=0)
    R_prior = torch.stack([s['R_prior'] for s in samples], dim=0)
    t_prior = torch.stack([s['t_prior'] for s in samples], dim=0)
    sun_direction = torch.stack([s['sun_direction'] for s in samples], dim=0)
    
    pos = torch.cat([s['pos'] for s in samples], dim=0)
    batch_indices = torch.cat([
        torch.full((s['pos'].shape[0],), idx, dtype=torch.long)
        for idx, s in enumerate(samples)
    ], dim=0)
    
    # Pad gt_mask to [B, max_nodes, max_nodes] since N_2D == N_3D in this setup
    gt_mask_padded = torch.zeros(B, max_nodes, max_nodes, dtype=torch.bool)
    for idx, s in enumerate(samples):
        num_nodes = s['pos'].shape[0]
        gt_mask_padded[idx, :num_nodes, :num_nodes] = s['gt_mask']
        
    res = {
        'image': images,
        'coords_2d': coords_2d_padded,
        'pos': pos,
        'batch': batch_indices,
        'K_cam': K_cam,
        'pose_gt': pose_gt,
        'R_prior': R_prior,
        't_prior': t_prior,
        'sun_direction': sun_direction,
        'gt_mask': gt_mask_padded
    }
    
    if 'normals_2d' in samples[0] and samples[0]['normals_2d'] is not None:
        res['normals_2d'] = torch.stack([s['normals_2d'] for s in samples], dim=0)
    if 'normals_3d' in samples[0] and samples[0]['normals_3d'] is not None:
        res['normals_3d'] = torch.cat([s['normals_3d'] for s in samples], dim=0)
        
    return res


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
