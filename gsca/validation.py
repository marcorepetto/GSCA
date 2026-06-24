import math
from typing import Dict, Union, Optional, Tuple
import torch

def degrade_point_cloud(
    point_cloud: torch.Tensor, 
    noise_std: float, 
    downsample_ratio: float
) -> torch.Tensor:
    """
    Applies synthetic degradations to a 3D point cloud by performing stochastic
    uniform downsampling and adding isotropic Gaussian noise.

    Args:
        point_cloud (torch.Tensor): Tensor of shape [B, N, 3] and dtype torch.float32.
        noise_std (float): Standard deviation of Gaussian noise (must be >= 0.0).
        downsample_ratio (float): Fraction of points to keep, in the range (0.0, 1.0].

    Returns:
        torch.Tensor: Degraded point cloud of shape [B, N_degraded, 3] and dtype torch.float32,
                      where N_degraded = int(N * downsample_ratio).
    """
    if not isinstance(point_cloud, torch.Tensor):
        raise TypeError("point_cloud must be a torch.Tensor")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    if not (0.0 < downsample_ratio <= 1.0):
        raise ValueError("downsample_ratio must be in the range (0.0, 1.0]")
    
    if point_cloud.dim() != 3 or point_cloud.shape[2] != 3:
        raise ValueError("point_cloud must have shape [B, N, 3]")
    
    B, N, C = point_cloud.shape

    N_degraded = int(N * downsample_ratio)
    
    # Stochastic uniform downsampling
    indices = torch.randperm(N, device=point_cloud.device)[:N_degraded]
    points_degraded = point_cloud[:, indices, :]
    
    # Isotropic Gaussian noise injection
    if noise_std > 0.0:
        noise = torch.randn_like(points_degraded) * noise_std
        points_degraded = points_degraded + noise
        
    return points_degraded


def synthesize_pose_prior(
    pose_gt: torch.Tensor, 
    max_trans: float = 1.0, 
    max_rot_deg: float = 5.0
) -> torch.Tensor:
    """
    Generates a perturbed camera pose prior from a ground-truth pose using
    uniform spatial displacements and random Euler rotations on all three axes.

    Args:
        pose_gt (torch.Tensor): Homogeneous ground truth pose of shape [B, 4, 4].
        max_trans (float): Maximum translation perturbation per axis (default: 1.0).
        max_rot_deg (float): Maximum Euler angle perturbation in degrees (default: 5.0).

    Returns:
        torch.Tensor: Perturbed camera pose prior of shape [B, 4, 4].
    """
    if not isinstance(pose_gt, torch.Tensor):
        raise TypeError("pose_gt must be a torch.Tensor")
    if pose_gt.dim() != 3 or pose_gt.shape[1] != 4 or pose_gt.shape[2] != 4:
        raise ValueError("pose_gt must have shape [B, 4, 4]")
    if max_trans < 0.0 or max_rot_deg < 0.0:
        raise ValueError("max_trans and max_rot_deg must be non-negative")

    B = pose_gt.shape[0]
    max_rot_rad = max_rot_deg * (math.pi / 180.0)
    
    # Generate random Euler angles uniformly in [-max_rot_rad, max_rot_rad]
    rx = (torch.rand(B, device=pose_gt.device, dtype=pose_gt.dtype) * 2.0 - 1.0) * max_rot_rad
    ry = (torch.rand(B, device=pose_gt.device, dtype=pose_gt.dtype) * 2.0 - 1.0) * max_rot_rad
    rz = (torch.rand(B, device=pose_gt.device, dtype=pose_gt.dtype) * 2.0 - 1.0) * max_rot_rad
    
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    
    # Build Rx
    Rx = torch.zeros(B, 3, 3, device=pose_gt.device, dtype=pose_gt.dtype)
    Rx[:, 0, 0] = 1.0
    Rx[:, 1, 1] = cx
    Rx[:, 1, 2] = -sx
    Rx[:, 2, 1] = sx
    Rx[:, 2, 2] = cx
    
    # Build Ry
    Ry = torch.zeros(B, 3, 3, device=pose_gt.device, dtype=pose_gt.dtype)
    Ry[:, 0, 0] = cy
    Ry[:, 0, 2] = sy
    Ry[:, 1, 1] = 1.0
    Ry[:, 2, 0] = -sy
    Ry[:, 2, 2] = cy
    
    # Build Rz
    Rz = torch.zeros(B, 3, 3, device=pose_gt.device, dtype=pose_gt.dtype)
    Rz[:, 0, 0] = cz
    Rz[:, 0, 1] = -sz
    Rz[:, 1, 0] = sz
    Rz[:, 1, 1] = cz
    Rz[:, 2, 2] = 1.0
    
    # Compose dR = Rz * Ry * Rx
    dR = torch.bmm(torch.bmm(Rz, Ry), Rx)
    
    # Generate random translation perturbation uniformly in [-max_trans, max_trans]
    dt = (torch.rand(B, 3, 1, device=pose_gt.device, dtype=pose_gt.dtype) * 2.0 - 1.0) * max_trans
    
    # Compose homogeneous transformation perturbation dT
    dT = torch.zeros(B, 4, 4, device=pose_gt.device, dtype=pose_gt.dtype)
    dT[:, :3, :3] = dR
    dT[:, :3, 3:4] = dt
    dT[:, 3, 3] = 1.0
    
    # Apply perturbation: T_prior = T_gt * dT
    pose_prior = torch.bmm(pose_gt, dT)
    return pose_prior


def _generate_fbm_noise_2d(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    octaves: int = 4,
    lacunarity: float = 2.0,
    gain: float = 0.5
) -> torch.Tensor:
    """
    Helper function to generate 2D Fractional Brownian Motion (fBm) noise
    using bilinear interpolation of random grids at different frequencies.
    """
    noise = torch.zeros(batch_size, channels, height, width, device=device, dtype=dtype)
    amplitude = 1.0
    frequency = 1.0
    total_amplitude = 0.0
    
    base_h, base_w = 4, 4
    
    for i in range(octaves):
        h = max(2, int(base_h * frequency))
        w = max(2, int(base_w * frequency))
        
        # Generate random noise grid
        octave_noise = torch.randn(batch_size, channels, h, w, device=device, dtype=dtype)
        
        # Interpolate to target resolution
        upsampled = torch.nn.functional.interpolate(
            octave_noise, 
            size=(height, width), 
            mode='bilinear', 
            align_corners=False
        )
        noise = noise + upsampled * amplitude
        total_amplitude += amplitude
        
        amplitude *= gain
        frequency *= lacunarity
        
    return noise / total_amplitude


def apply_visual_degradations(
    image: torch.Tensor, 
    normal_map: Optional[torch.Tensor] = None, 
    albedo_map: Optional[torch.Tensor] = None, 
    sun_azimuth: torch.Tensor = None, 
    sun_elevation: torch.Tensor = None, 
    roughness_factor: float = 0.0,
    return_perturbed_normal: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Applies physically consistent visual degradations including solar illumination,
    fractal-perturbed normal maps (roughness) if normal_map is provided, weathering on albedo,
    and stochastic white balance.

    Args:
        image (torch.Tensor): Image tensor of shape [B, 3, H, W] and values in [0.0, 1.0].
        normal_map (torch.Tensor, optional): Normal map of shape [B, 3, H, W] in [-1.0, 1.0].
        albedo_map (torch.Tensor, optional): Albedo map of shape [B, 3, H, W] in [0.0, 1.0]. Defaults to image.
        sun_azimuth (torch.Tensor): Solar azimuth angle in degrees of shape [B].
        sun_elevation (torch.Tensor): Solar elevation angle in degrees of shape [B].
        roughness_factor (float): Roughness factor for fractal perturbation (must be >= 0.0).
        return_perturbed_normal (bool): If True, returns the tuple (degraded_image, normal_map_perturbed).

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Degraded image of shape [B, 3, H, W],
        optionally with the perturbed normal map of shape [B, 3, H, W].
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")
    if normal_map is not None and not isinstance(normal_map, torch.Tensor):
        raise TypeError("normal_map must be a torch.Tensor")
    if albedo_map is not None and not isinstance(albedo_map, torch.Tensor):
        raise TypeError("albedo_map must be a torch.Tensor")
    if sun_azimuth is None:
        sun_azimuth = torch.zeros(image.shape[0], device=image.device)
    if sun_elevation is None:
        sun_elevation = torch.zeros(image.shape[0], device=image.device)
    if not isinstance(sun_azimuth, torch.Tensor):
        raise TypeError("sun_azimuth must be a torch.Tensor")
    if not isinstance(sun_elevation, torch.Tensor):
        raise TypeError("sun_elevation must be a torch.Tensor")
    if not isinstance(roughness_factor, (int, float)):
        raise TypeError("roughness_factor must be a float or int")
    if roughness_factor < 0.0:
        raise ValueError("roughness_factor must be non-negative")

    # Shape validation
    if image.dim() != 4 or image.shape[1] != 3:
        raise ValueError("image must have shape [B, 3, H, W]")
    B, _, H, W = image.shape
    
    if normal_map is not None and normal_map.shape != (B, 3, H, W):
        raise ValueError(f"normal_map must have shape [B, 3, H, W], got {list(normal_map.shape)}")
        
    if albedo_map is None:
        albedo_map = image
    else:
        if albedo_map.shape != (B, 3, H, W):
            raise ValueError(f"albedo_map must have shape [B, 3, H, W], got {list(albedo_map.shape)}")
            
    if sun_azimuth.shape != (B,):
        raise ValueError(f"sun_azimuth must have shape [B], got {list(sun_azimuth.shape)}")
    if sun_elevation.shape != (B,):
        raise ValueError(f"sun_elevation must have shape [B], got {list(sun_elevation.shape)}")

    # Convert sun angles to radians
    theta_rad = sun_azimuth * (math.pi / 180.0)
    phi_rad = sun_elevation * (math.pi / 180.0)
    
    # Calculate sun light direction vector L in camera space
    Lx = torch.cos(phi_rad) * torch.sin(theta_rad)
    Ly = torch.sin(phi_rad)
    Lz = torch.cos(phi_rad) * torch.cos(theta_rad)
    L = torch.stack([Lx, Ly, Lz], dim=1).view(B, 3, 1, 1)

    # Generate 2D fractal noise (fBm) for normals and albedo
    fractal_noise_albedo = _generate_fbm_noise_2d(
        batch_size=B, channels=3, height=H, width=W,
        device=image.device, dtype=image.dtype
    )

    # Weathered albedo alteration
    albedo_weathered = albedo_map * (1.0 - 0.2 * fractal_noise_albedo)

    normal_map_perturbed = None
    if normal_map is not None:
        fractal_noise_normal = _generate_fbm_noise_2d(
            batch_size=B, channels=3, height=H, width=W,
            device=image.device, dtype=image.dtype
        )
        # Perturb and normalize normal map
        normal_map_perturbed = normal_map + roughness_factor * fractal_noise_normal
        norm = torch.norm(normal_map_perturbed, p=2, dim=1, keepdim=True)
        normal_map_perturbed = normal_map_perturbed / (norm + 1e-8)

        # Lambertian diffuse lighting term
        lambertian = torch.sum(normal_map_perturbed * L, dim=1, keepdim=True)
        lambertian = torch.clamp(lambertian, min=0.0)
        image_degraded = albedo_weathered * lambertian
    else:
        # Compose output image without lighting
        image_degraded = albedo_weathered

    # Stochastic white balance factor [B, 3, 1, 1] in [0.95, 1.05]
    wb_factors = 0.95 + 0.10 * torch.rand(B, 3, 1, 1, device=image.device, dtype=image.dtype)
    image_degraded = image_degraded * wb_factors

    # Final clamping to valid RGB range [0.0, 1.0]
    output_image = torch.clamp(image_degraded, min=0.0, max=1.0)
    
    if return_perturbed_normal:
        return output_image, normal_map_perturbed
    return output_image


def _compute_auc_for_threshold(
    errors_trans: torch.Tensor, 
    errors_rot: torch.Tensor, 
    t_max: float, 
    r_max: float, 
    steps: int = 50
) -> float:
    """
    Computes Pose AUC for a given threshold category (t_max, r_max) using 
    trapezoidal integration over steps.
    """
    if errors_trans.numel() == 0:
        return 0.0
        
    # Treat samples satisfying the maximum thresholds of the category as 0 error.
    # This guarantees that if 100% of samples satisfy the criteria, the AUC is exactly 1.0.
    satisfied_max = (errors_trans <= t_max) & (errors_rot <= r_max)
    eff_trans = torch.where(satisfied_max, torch.zeros_like(errors_trans), errors_trans)
    eff_rot = torch.where(satisfied_max, torch.zeros_like(errors_rot), errors_rot)

    recalls = []
    for i in range(steps):
        ratio = i / (steps - 1)
        t_thresh = ratio * t_max
        r_thresh = ratio * r_max
        
        satisfied = (eff_trans <= t_thresh) & (eff_rot <= r_thresh)
        recall = satisfied.float().mean().item()
        recalls.append(recall)
        
    # Trapezoidal integration
    area = 0.0
    dx = 1.0 / (steps - 1)
    for i in range(steps - 1):
        area += (recalls[i] + recalls[i+1]) * 0.5 * dx
    return area


def evaluate_alignment(
    pose_est: torch.Tensor, 
    pose_gt: torch.Tensor, 
    correspondences: Dict[str, torch.Tensor], 
    intrinsics_K: torch.Tensor
) -> Dict[str, Union[torch.Tensor, Dict[str, float]]]:
    """
    Calculates alignment errors between estimated and ground-truth camera poses,
    along with average reprojection errors for inliers, inlier ratio, and pose AUC.

    Args:
        pose_est (torch.Tensor): Estimated homogeneous camera poses of shape [B, 4, 4].
        pose_gt (torch.Tensor): Ground-truth homogeneous camera poses of shape [B, 4, 4].
        correspondences (Dict[str, torch.Tensor]): Dict containing:
            - "pts_2d": pixel coordinates of shape [B, M, 2].
            - "pts_3d": 3D points of shape [B, M, 3].
            - "inlier_mask": boolean mask of shape [B, M].
        intrinsics_K (torch.Tensor): Camera intrinsic matrices of shape [B, 3, 3].

    Returns:
        Dict[str, Union[torch.Tensor, Dict[str, float]]]: Evaluation metrics containing:
            - "translation_error" (torch.Tensor [B]): error in meters.
            - "rotation_error" (torch.Tensor [B]): geodesic error in degrees.
            - "auc_pose" (Dict[str, float]): AUC values for "easy", "medium", "strict" thresholds.
            - "inlier_ratio" (torch.Tensor [B]): ratio of inliers.
            - "reprojection_error" (torch.Tensor [B]): mean reprojection error in pixels for inliers.
    """
    # Validation checks
    if not isinstance(pose_est, torch.Tensor) or not isinstance(pose_gt, torch.Tensor):
        raise TypeError("pose_est and pose_gt must be torch.Tensors")
    if not isinstance(intrinsics_K, torch.Tensor):
        raise TypeError("intrinsics_K must be a torch.Tensor")
    if not isinstance(correspondences, dict):
        raise TypeError("correspondences must be a dictionary")
    
    for k in ["pts_2d", "pts_3d", "inlier_mask"]:
        if k not in correspondences:
            raise KeyError(f"Missing key '{k}' in correspondences")
        if not isinstance(correspondences[k], torch.Tensor):
            raise TypeError(f"correspondences['{k}'] must be a torch.Tensor")
            
    B = pose_est.shape[0]
    if pose_est.shape != (B, 4, 4) or pose_gt.shape != (B, 4, 4):
        raise ValueError("pose_est and pose_gt must have shape [B, 4, 4]")
    if intrinsics_K.shape != (B, 3, 3):
        raise ValueError("intrinsics_K must have shape [B, 3, 3]")
        
    pts_2d = correspondences["pts_2d"]
    pts_3d = correspondences["pts_3d"]
    inlier_mask = correspondences["inlier_mask"]
    
    M = pts_2d.shape[1]
    if pts_2d.shape != (B, M, 2):
        raise ValueError("pts_2d must have shape [B, M, 2]")
    if pts_3d.shape != (B, M, 3):
        raise ValueError("pts_3d must have shape [B, M, 3]")
    if inlier_mask.shape != (B, M):
        raise ValueError("inlier_mask must have shape [B, M]")
    if inlier_mask.dtype != torch.bool:
        raise TypeError("inlier_mask must be of type torch.bool")

    # Translation error (L2 norm)
    t_est = pose_est[:, :3, 3]
    t_gt = pose_gt[:, :3, 3]
    translation_error = torch.norm(t_est - t_gt, p=2, dim=1)

    # Geodesic rotation error (degrees)
    R_est = pose_est[:, :3, :3]
    R_gt = pose_gt[:, :3, :3]
    R_diff = torch.bmm(R_est, R_gt.transpose(1, 2))
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    val_clamped = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    rotation_error = torch.acos(val_clamped) * (180.0 / math.pi)

    # Inlier ratio
    inlier_ratio = inlier_mask.sum(dim=1).float() / M

    # Reprojection error for inlier points
    t_est_col = t_est.unsqueeze(-1)  # [B, 3, 1]
    # Transform points to camera coordinates: [B, 3, M]
    pts_cam = torch.bmm(R_est, pts_3d.transpose(1, 2)) + t_est_col
    # Project to image plane: [B, 3, M]
    pts_proj_hom = torch.bmm(intrinsics_K, pts_cam)
    # Divide by depth (z) to obtain pixel coordinates
    z = pts_proj_hom[:, 2:3, :]
    z_clamp = torch.where(z >= 0, torch.clamp(z, min=1e-8), torch.clamp(z, max=-1e-8))
    pts_proj_2d = pts_proj_hom[:, :2, :] / z_clamp
    pts_proj_2d = pts_proj_2d.transpose(1, 2)  # [B, M, 2]

    # Euclidean distance per point
    dist_errors = torch.norm(pts_proj_2d - pts_2d, p=2, dim=2)  # [B, M]

    # Average reprojection error only over mask (handling zero-inliers safely)
    reproj_errors = []
    for b in range(B):
        mask_b = inlier_mask[b]
        if not mask_b.any():
            reproj_errors.append(torch.tensor(0.0, device=pose_est.device, dtype=pose_est.dtype))
        else:
            reproj_errors.append(dist_errors[b, mask_b].mean())
    reprojection_error = torch.stack(reproj_errors)

    # Calculate Pose AUC for easy, medium, strict categories
    auc_easy = _compute_auc_for_threshold(translation_error, rotation_error, t_max=0.20, r_max=10.0)
    auc_medium = _compute_auc_for_threshold(translation_error, rotation_error, t_max=0.10, r_max=5.0)
    auc_strict = _compute_auc_for_threshold(translation_error, rotation_error, t_max=0.05, r_max=1.0)

    auc_pose = {
        "easy": auc_easy,
        "medium": auc_medium,
        "strict": auc_strict
    }

    return {
        "translation_error": translation_error,
        "rotation_error": rotation_error,
        "auc_pose": auc_pose,
        "inlier_ratio": inlier_ratio,
        "reprojection_error": reprojection_error
    }
