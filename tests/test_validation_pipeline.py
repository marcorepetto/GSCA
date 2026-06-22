import math
import pytest
import torch
from gsca.validation import (
    degrade_point_cloud,
    synthesize_pose_prior,
    apply_visual_degradations,
    evaluate_alignment
)

def test_pose_perturbation_limits():
    """
    T-VAL-05 (1): Verify that for 1000 iterations with identity ground-truth pose,
    the translation of the prior is strictly within [-max_trans, max_trans] per axis,
    and the residual rotation angle is bounded appropriately (<= 3 * max_rot_deg).
    """
    pose_gt = torch.eye(4, dtype=torch.float32).unsqueeze(0)  # [1, 4, 4]
    max_trans = 1.2
    max_rot_deg = 6.0
    
    for _ in range(1000):
        pose_prior = synthesize_pose_prior(pose_gt, max_trans=max_trans, max_rot_deg=max_rot_deg)
        
        # Check translation limits per axis: must be within [-max_trans, max_trans]
        t_prior = pose_prior[0, :3, 3]
        assert torch.all(t_prior >= -max_trans - 1e-6)
        assert torch.all(t_prior <= max_trans + 1e-6)
        
        # Check rotation limits: trace and geodesic angle
        R_prior = pose_prior[0, :3, :3]
        trace = R_prior[0, 0] + R_prior[1, 1] + R_prior[2, 2]
        val_clamped = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
        rot_error_deg = torch.acos(val_clamped) * (180.0 / math.pi)
        
        # The sum of 3 Euler angles r_x, r_y, r_z defines the upper bound for the geodesic rotation:
        # geodesic_error <= |r_x| + |r_y| + |r_z| <= 3 * max_rot_deg
        assert rot_error_deg.item() <= 3.0 * max_rot_deg + 1e-4
        assert rot_error_deg.item() >= 0.0


def test_point_cloud_degradation_statistics():
    """
    T-VAL-05 (2): Evaluate a point cloud of shape [1, 10000, 3] initialized to zero
    degraded with downsample_ratio=0.5 and noise_std=0.03. Check that the output shape
    is exactly [1, 5000, 3], and the empirical standard deviation and mean in each
    dimension lie within the target intervals.
    """
    torch.manual_seed(42)
    point_cloud = torch.zeros(1, 10000, 3, dtype=torch.float32)
    noise_std = 0.03
    downsample_ratio = 0.5
    
    degraded = degrade_point_cloud(point_cloud, noise_std=noise_std, downsample_ratio=downsample_ratio)
    
    # Assert exact shape
    assert degraded.shape == (1, 5000, 3)
    
    # Calculate empirical standard deviation and mean per dimension
    std_emp = torch.std(degraded, dim=1)  # [1, 3]
    mean_emp = torch.mean(degraded, dim=1)  # [1, 3]
    
    for c in range(3):
        # target std_dev: [0.028, 0.032]
        assert 0.028 <= std_emp[0, c].item() <= 0.032
        # target mean: [-0.001, 0.001]
        assert -0.001 <= mean_emp[0, c].item() <= 0.001


def test_normal_map_conservation():
    """
    T-VAL-05 (3): Validate that after applying roughness with apply_visual_degradations
    on a uniform [0, 0, 1] normal map, the Euclidean norm of vectors at each pixel is exactly 1.0 +/- 1e-6.
    """
    B, H, W = 2, 32, 32
    image = torch.ones(B, 3, H, W, dtype=torch.float32)
    
    # Uniform normal map [0, 0, 1]
    normal_map = torch.zeros(B, 3, H, W, dtype=torch.float32)
    normal_map[:, 2, :, :] = 1.0
    
    albedo_map = torch.ones(B, 3, H, W, dtype=torch.float32)
    sun_azimuth = torch.zeros(B, dtype=torch.float32)
    sun_elevation = torch.ones(B, dtype=torch.float32) * 45.0
    roughness_factor = 0.8
    
    _, normal_map_perturbed = apply_visual_degradations(
        image=image,
        normal_map=normal_map,
        albedo_map=albedo_map,
        sun_azimuth=sun_azimuth,
        sun_elevation=sun_elevation,
        roughness_factor=roughness_factor,
        return_perturbed_normal=True
    )
    
    # Compute L2 norm of the normal vectors at each pixel
    pixel_norms = torch.norm(normal_map_perturbed, p=2, dim=1)  # [B, H, W]
    
    # Assert norm is 1.0 +/- 1e-6
    assert torch.allclose(pixel_norms, torch.ones_like(pixel_norms), atol=1e-6)


def test_pose_error_metrics_correctness():
    """
    T-VAL-05 (4): Verify pose error metrics for an analytic case where estimated pose
    represents a 90-degree rotation about the Z axis and translation of [1.5, 0.0, 0.0] meters
    with respect to identity ground-truth pose.
    """
    B = 1
    pose_gt = torch.eye(4, dtype=torch.float32).unsqueeze(0)  # [1, 4, 4]
    
    # pose_est: 90 deg rotation on Z axis, translation [1.5, 0.0, 0.0]
    pose_est = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    pose_est[0, 0, 0] = 0.0
    pose_est[0, 0, 1] = -1.0
    pose_est[0, 1, 0] = 1.0
    pose_est[0, 1, 1] = 0.0
    pose_est[0, 0, 3] = 1.5
    
    # Camera intrinsics (Identity matrix for simplicity)
    intrinsics_K = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    
    # 5 dummy points
    pts_2d = torch.zeros(B, 5, 2, dtype=torch.float32)
    pts_3d = torch.zeros(B, 5, 3, dtype=torch.float32)
    inlier_mask = torch.ones(B, 5, dtype=torch.bool)
    correspondences = {
        "pts_2d": pts_2d,
        "pts_3d": pts_3d,
        "inlier_mask": inlier_mask
    }
    
    metrics = evaluate_alignment(pose_est, pose_gt, correspondences, intrinsics_K)
    
    trans_err = metrics["translation_error"][0].item()
    rot_err = metrics["rotation_error"][0].item()
    
    # Verify translation error is exactly 1.5 +/- 1e-5 meters
    assert math.isclose(trans_err, 1.5, abs_tol=1e-5)
    # Verify rotation error is exactly 90.0 +/- 1e-5 degrees
    assert math.isclose(rot_err, 90.0, abs_tol=1e-5)


def test_auc_calculation_control():
    """
    T-VAL-05 (5): Verify positive and negative control cases for the Pose AUC calculations:
    - Positive control: translation errors < 5 cm and rotation errors < 1 deg (should yield AUC = 1.0)
    - Negative control: translation errors = 1.0 m and rotation errors = 15 deg (should yield AUC = 0.0)
    """
    B = 10
    pose_gt = torch.eye(4, dtype=torch.float32).repeat(B, 1, 1)
    
    # Dummy inputs for correspondences and intrinsics
    intrinsics_K = torch.eye(3, dtype=torch.float32).repeat(B, 1, 1)
    correspondences = {
        "pts_2d": torch.zeros(B, 5, 2, dtype=torch.float32),
        "pts_3d": torch.zeros(B, 5, 3, dtype=torch.float32),
        "inlier_mask": torch.ones(B, 5, dtype=torch.bool)
    }

    # --- Case 1: Positive Control ---
    pose_est_pos = torch.eye(4, dtype=torch.float32).repeat(B, 1, 1)
    pose_est_pos[:, 0, 3] = 0.02  # 2 cm translation error
    # Rotate by 0.5 degrees about Z axis
    angle_rad = 0.5 * (math.pi / 180.0)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    pose_est_pos[:, 0, 0] = cos_a
    pose_est_pos[:, 0, 1] = -sin_a
    pose_est_pos[:, 1, 0] = sin_a
    pose_est_pos[:, 1, 1] = cos_a
    
    metrics_pos = evaluate_alignment(pose_est_pos, pose_gt, correspondences, intrinsics_K)
    auc_pos = metrics_pos["auc_pose"]
    
    assert torch.all(metrics_pos["translation_error"] < 0.05)
    assert torch.all(metrics_pos["rotation_error"] < 1.0)
    
    assert math.isclose(auc_pos["strict"], 1.0, abs_tol=1e-5)
    assert math.isclose(auc_pos["medium"], 1.0, abs_tol=1e-5)
    assert math.isclose(auc_pos["easy"], 1.0, abs_tol=1e-5)

    # --- Case 2: Negative Control ---
    pose_est_neg = torch.eye(4, dtype=torch.float32).repeat(B, 1, 1)
    pose_est_neg[:, 0, 3] = 1.0  # 1.0 meter translation error
    # Rotate by 15.0 degrees about Z axis
    angle_rad_neg = 15.0 * (math.pi / 180.0)
    cos_a_neg, sin_a_neg = math.cos(angle_rad_neg), math.sin(angle_rad_neg)
    pose_est_neg[:, 0, 0] = cos_a_neg
    pose_est_neg[:, 0, 1] = -sin_a_neg
    pose_est_neg[:, 1, 0] = sin_a_neg
    pose_est_neg[:, 1, 1] = cos_a_neg
    
    metrics_neg = evaluate_alignment(pose_est_neg, pose_gt, correspondences, intrinsics_K)
    auc_neg = metrics_neg["auc_pose"]
    
    assert torch.all(metrics_neg["translation_error"] > 0.20)
    assert torch.all(metrics_neg["rotation_error"] > 10.0)
    
    assert math.isclose(auc_neg["strict"], 0.0, abs_tol=1e-5)
    assert math.isclose(auc_neg["medium"], 0.0, abs_tol=1e-5)
    assert math.isclose(auc_neg["easy"], 0.0, abs_tol=1e-5)


def test_exception_handling():
    """
    Verify parameter and input type check exceptions for robustness.
    """
    # degrade_point_cloud checks
    with pytest.raises(TypeError):
        degrade_point_cloud("not_a_tensor", 0.03, 0.5)
    with pytest.raises(ValueError):
        degrade_point_cloud(torch.zeros(1, 10, 3), -0.1, 0.5)
    with pytest.raises(ValueError):
        degrade_point_cloud(torch.zeros(1, 10, 3), 0.03, 1.5)
        
    # synthesize_pose_prior checks
    with pytest.raises(TypeError):
        synthesize_pose_prior("not_a_tensor")
    with pytest.raises(ValueError):
        synthesize_pose_prior(torch.zeros(4, 4), max_trans=1.0)
    with pytest.raises(ValueError):
        synthesize_pose_prior(torch.zeros(1, 4, 4), max_trans=-0.5)

    # apply_visual_degradations checks
    img = torch.zeros(1, 3, 16, 16)
    n_map = torch.zeros(1, 3, 16, 16)
    alb = torch.zeros(1, 3, 16, 16)
    az = torch.zeros(1)
    el = torch.zeros(1)
    with pytest.raises(ValueError):
        apply_visual_degradations(img, n_map, alb, az, el, roughness_factor=-1.0)
    with pytest.raises(ValueError):
        # spatial size mismatch
        apply_visual_degradations(img, torch.zeros(1, 3, 8, 8), alb, az, el, roughness_factor=0.5)
        
    # evaluate_alignment checks
    with pytest.raises(TypeError):
        evaluate_alignment("not_a_tensor", torch.eye(4).unsqueeze(0), {}, torch.eye(3).unsqueeze(0))
