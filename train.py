import os
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from typing import Tuple, Dict, Any, List
from torch_geometric.utils import to_dense_batch

from gsca.dataset import GSCADataset, get_gsca_dataloader
from gsca.models.gsca_network import GSCANetwork
from gsca.models.losses import CircleLossWithSelfPacedWeighting
from gsca.training import configure_optimizers, train_step
from gsca.models.gsca_matcher import compute_mnn_matches
from gsca.validation import evaluate_alignment


def estimate_pose_pnp(
    coords_pixel_2d: torch.Tensor,
    pos_dense: torch.Tensor,
    matches: torch.Tensor,
    K_cam: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solves Perspective-n-Points using RANSAC and EPnP.
    
    Args:
        coords_pixel_2d: [N_2D, 2] pixel coordinates.
        pos_dense: [max_nodes, 3] 3D points.
        matches: [M, 2] matching indices.
        K_cam: [3, 3] intrinsics.

    Returns:
        pose_est: [4, 4] estimated camera pose.
        inlier_mask: [M] boolean mask.
    """
    device = coords_pixel_2d.device
    M = matches.shape[0]

    if M < 4:
        return torch.eye(4, device=device), torch.zeros(M, dtype=torch.bool, device=device)

    # Extract matched coordinates
    pts_2d = coords_pixel_2d[matches[:, 0]].cpu().numpy().astype(np.float32)
    pts_3d = pos_dense[matches[:, 1]].cpu().numpy().astype(np.float32)
    K = K_cam.cpu().numpy().astype(np.float32)

    # Solve PnP RANSAC
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=pts_3d,
        imagePoints=pts_2d,
        cameraMatrix=K,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not success or inliers is None:
        return torch.eye(4, device=device), torch.zeros(M, dtype=torch.bool, device=device)

    # Convert rotation vector to matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # homogeneous transform matrix
    T_est = np.eye(4, dtype=np.float32)
    T_est[:3, :3] = R
    T_est[:3, 3:4] = tvec

    pose_est = torch.from_numpy(T_est).to(device=device, dtype=torch.float32)

    # Construct inlier boolean mask
    inlier_indices = inliers.flatten()
    inlier_mask_np = np.zeros(M, dtype=bool)
    inlier_mask_np[inlier_indices] = True
    inlier_mask = torch.from_numpy(inlier_mask_np).to(device=device, dtype=torch.bool)

    return pose_est, inlier_mask


def generate_dummy_data(tmp_dir: str, num_samples: int = 4) -> List[str]:
    """Generates dummy .pt sample files to support test dry-runs."""
    os.makedirs(tmp_dir, exist_ok=True)
    sample_paths = []
    H, W = 224, 224
    N_2D = 30
    N_3D = 40
    
    for i in range(num_samples):
        sample_data = {
            'image': torch.rand(3, H, W),
            'pos': torch.randn(N_3D, 3),
            'K_cam': torch.eye(3),
            'pose_gt': torch.eye(4),
            'coords_pixel_2d': torch.rand(N_2D, 2) * (H - 1),
        }
        path = os.path.join(tmp_dir, f"dummy_sample_{i}.pt")
        torch.save(sample_data, path)
        sample_paths.append(path)
    return sample_paths


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # 1. Resolve OmegaConf to Python native dict for logging
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # 2. Check if we are running in dry-run mode (testing/ci)
    dry_run = cfg_dict.get('dry_run', False)
    
    # 3. Initialize Weights & Biases
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        config=cfg_dict,
        mode="disabled" if dry_run else None  # Disable wandb uploads in dry-run mode
    )

    # 4. Determine device
    device = torch.device(cfg.training.device if torch.cuda.is_available() and cfg.training.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    # 5. Populate train and val paths
    train_paths = list(cfg.data.train_paths)
    val_paths = list(cfg.data.val_paths)
    
    if dry_run:
        # Create temporary dummy files for validation run
        print("Dry-run mode enabled: generating synthetic dummy datasets...")
        dummy_dir = os.path.join(os.getcwd(), "dummy_data_cache")
        train_paths = generate_dummy_data(os.path.join(dummy_dir, "train"), num_samples=4)
        val_paths = generate_dummy_data(os.path.join(dummy_dir, "val"), num_samples=2)
    else:
        # Populate from directory if paths lists are empty
        if not train_paths and os.path.exists(cfg.data.train_dir):
            train_paths = [os.path.join(cfg.data.train_dir, f) for f in os.listdir(cfg.data.train_dir) if f.endswith('.pt')]
        if not val_paths and os.path.exists(cfg.data.val_dir):
            val_paths = [os.path.join(cfg.data.val_dir, f) for f in os.listdir(cfg.data.val_dir) if f.endswith('.pt')]

        # Fallback to loading from data/rgb and splitting if still empty
        if not train_paths and not val_paths:
            rgb_dir = "data/rgb"
            if os.path.exists(rgb_dir):
                all_samples = [os.path.join(rgb_dir, f) for f in sorted(os.listdir(rgb_dir)) if f.endswith('.png')]
                if len(all_samples) > 0:
                    import random
                    random.seed(42)
                    random.shuffle(all_samples)
                    
                    split_idx = int(len(all_samples) * 0.9)
                    train_paths = all_samples[:split_idx]
                    val_paths = all_samples[split_idx:]
                    print(f"Split {len(all_samples)} samples from {rgb_dir} randomly (90/10) into {len(train_paths)} train and {len(val_paths)} val samples.")

    if not train_paths:
        raise ValueError("Training sample list is empty. Set data.train_paths or configure a valid data.train_dir.")
    if not val_paths:
        raise ValueError("Validation sample list is empty. Set data.val_paths or configure a valid data.val_dir.")

    # 6. Instantiate datasets and dataloaders
    train_dataset = GSCADataset(
        sample_paths=train_paths,
        degrade_points=cfg.data.degrade_points,
        degrade_visual=cfg.data.degrade_visual,
        noise_std=cfg.data.noise_std,
        downsample_ratio=cfg.data.downsample_ratio,
        roughness_factor=cfg.data.roughness_factor,
        pixel_match_threshold=cfg.data.pixel_match_threshold,
        near_plane=cfg.data.near_plane
    )
    
    val_dataset = GSCADataset(
        sample_paths=val_paths,
        degrade_points=cfg.data.degrade_points,
        degrade_visual=cfg.data.degrade_visual,
        noise_std=cfg.data.noise_std,
        downsample_ratio=cfg.data.downsample_ratio,
        roughness_factor=cfg.data.roughness_factor,
        pixel_match_threshold=cfg.data.pixel_match_threshold,
        near_plane=cfg.data.near_plane
    )

    train_loader = get_gsca_dataloader(
        train_dataset,
        batch_size=cfg.data.batch_size if not dry_run else 2,
        shuffle=True,
        num_workers=cfg.data.num_workers
    )
    
    val_loader = get_gsca_dataloader(
        val_dataset,
        batch_size=cfg.data.batch_size if not dry_run else 2,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )

    # 7. Instantiate unified model
    model = GSCANetwork(
        backbone_name=cfg.model.backbone_name,
        bottleneck_dim=cfg.model.bottleneck_dim,
        intermediate_layers=list(cfg.model.intermediate_layers),
        out_channels=cfg.model.out_channels,
        k=cfg.model.k,
        pretrained=cfg.model.pretrained and not dry_run  # skip weights download on dry-run
    ).to(device)

    # 8. Setup optimizer, scheduler, and loss function
    epochs = cfg.training.epochs if not dry_run else 1
    steps_per_epoch = len(train_loader)
    
    optimizer, scheduler = configure_optimizers(
        model=model,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch
    )
    
    loss_fn = CircleLossWithSelfPacedWeighting(
        gamma=cfg.loss.gamma,
        margin=cfg.loss.margin
    )

    # 9. Main training loop
    checkpoint_dir = cfg.training.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')

    import time
    start_time = time.time()
    total_steps = epochs * steps_per_epoch

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        # Training loop
        optimizer.zero_grad() # Clean start for the epoch
        accumulation_steps = 4 # Simulate 4x larger batch size
        
        for step, batch in enumerate(train_loader, 1):
            stats = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                device=device,
                step=step,
                accumulation_steps=accumulation_steps
            )
            epoch_loss += stats['loss']
            
            global_step = (epoch - 1) * steps_per_epoch + step
            if global_step % cfg.wandb.log_freq == 0:
                elapsed = time.time() - start_time
                avg_step_time = elapsed / global_step
                steps_left = total_steps - global_step
                eta_sec = steps_left * avg_step_time

                hours = int(eta_sec // 3600)
                minutes = int((eta_sec % 3600) // 60)
                seconds = int(eta_sec % 60)
                if hours > 0:
                    eta_str = f"{hours}h {minutes}m {seconds}s"
                elif minutes > 0:
                    eta_str = f"{minutes}m {seconds}s"
                else:
                    eta_str = f"{seconds}s"

                wandb.log({
                    "train/loss": stats['loss'],
                    "train/lr": stats['lr'],
                    "epoch": epoch,
                    "eta_seconds": eta_sec
                })
                print(f"[Epoch {epoch}/{epochs} | Step {step}/{steps_per_epoch}] Loss: {stats['loss']:.4f} | LR: {stats['lr']:.6f} | ETA: {eta_str}")

        avg_train_loss = epoch_loss / steps_per_epoch
        wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch})
        print(f"--- Epoch {epoch} Training Summary --- Avg Loss: {avg_train_loss:.4f}")

        # 10. Validation loop
        if epoch % cfg.training.val_freq == 0:
            model.eval()
            val_loss = 0.0
            val_steps = len(val_loader)
            
            all_pose_est = []
            all_pose_gt = []
            all_pts_2d = []
            all_pts_3d = []
            all_inlier_masks = []
            all_K_cam = []
            
            with torch.no_grad():
                for batch in val_loader:
                    # Run model forward pass on device
                    # GSCANetwork expectations are resolved by train_step compatibility check
                    images = batch['image'].to(device)
                    coords_2d = batch['coords_2d'].to(device)
                    pos = batch['pos'].to(device)
                    batch_idx = batch['batch'].to(device)
                    K_cam = batch['K_cam'].to(device)
                    R_prior = batch['R_prior'].to(device)
                    t_prior = batch['t_prior'].to(device)
                    
                    feat_2d_refined, feat_3d_dense = model(
                        images=images,
                        coords_2d=coords_2d,
                        pos=pos,
                        batch=batch_idx,
                        K_cam=K_cam,
                        R_prior=R_prior,
                        t_prior=t_prior,
                        delta=cfg.training.delta,
                        near_plane=cfg.data.near_plane,
                    )
                    
                    # Compute validation loss
                    feat_2d_refined_norm = F.normalize(feat_2d_refined, p=2, dim=-1)
                    feat_3d_dense_norm = F.normalize(feat_3d_dense, p=2, dim=-1)
                    loss = loss_fn(feat_2d_refined_norm, feat_3d_dense_norm, batch['gt_mask'].to(device))
                    val_loss += loss.item()

                    # Pose validation
                    B_val, _, H_img, W_img = images.shape
                    # Reconstruct pixel coordinates
                    u_pixel = (coords_2d[:, :, 0] + 1.0) * (W_img - 1) / 2.0
                    v_pixel = (coords_2d[:, :, 1] + 1.0) * (H_img - 1) / 2.0
                    coords_pixel_2d = torch.stack([u_pixel, v_pixel], dim=-1)

                    # We also need dense pos for matching
                    pos_dense, _ = to_dense_batch(pos, batch_idx)

                    # Solve PnP sample by sample in batch
                    for b in range(B_val):
                        # compute matches
                        matches, _ = compute_mnn_matches(
                            feat_2d_refined_norm[b], 
                            feat_3d_dense_norm[b], 
                            sim_threshold=cfg.loss.margin
                        )
                        pose_est, inlier_mask = estimate_pose_pnp(
                            coords_pixel_2d=coords_pixel_2d[b],
                            pos_dense=pos_dense[b],
                            matches=matches,
                            K_cam=K_cam[b]
                        )
                        
                        all_pose_est.append(pose_est)
                        all_pose_gt.append(batch['pose_gt'][b].to(device))
                        all_K_cam.append(K_cam[b])
                        
                        # Store matches for evaluate_alignment
                        pts_2d_matched = coords_pixel_2d[b][matches[:, 0]]
                        pts_3d_matched = pos_dense[b][matches[:, 1]]
                        
                        all_pts_2d.append(pts_2d_matched)
                        all_pts_3d.append(pts_3d_matched)
                        all_inlier_masks.append(inlier_mask)

            avg_val_loss = val_loss / val_steps
            print(f"Validation Loss: {avg_val_loss:.4f}")

            # Compute alignment metrics (requires padding matches to stack batches)
            if len(all_pose_est) > 0:
                B_eval = len(all_pose_est)
                max_matches = max(pts.shape[0] for pts in all_pts_2d)
                
                # Check if there are any matches in the batch
                if max_matches > 0:
                    pts_2d_pad = torch.zeros(B_eval, max_matches, 2, device=device)
                    pts_3d_pad = torch.zeros(B_eval, max_matches, 3, device=device)
                    inlier_mask_pad = torch.zeros(B_eval, max_matches, dtype=torch.bool, device=device)
                    
                    for b in range(B_eval):
                        n_matches = all_pts_2d[b].shape[0]
                        if n_matches > 0:
                            pts_2d_pad[b, :n_matches] = all_pts_2d[b]
                            pts_3d_pad[b, :n_matches] = all_pts_3d[b]
                            inlier_mask_pad[b, :n_matches] = all_inlier_masks[b]
                    
                    pose_est_tensor = torch.stack(all_pose_est, dim=0)
                    pose_gt_tensor = torch.stack(all_pose_gt, dim=0)
                    K_cam_tensor = torch.stack(all_K_cam, dim=0)
                    
                    eval_correspondences = {
                        "pts_2d": pts_2d_pad,
                        "pts_3d": pts_3d_pad,
                        "inlier_mask": inlier_mask_pad
                    }
                    
                    # Evaluate
                    eval_metrics = evaluate_alignment(
                        pose_est=pose_est_tensor,
                        pose_gt=pose_gt_tensor,
                        correspondences=eval_correspondences,
                        intrinsics_K=K_cam_tensor
                    )
                    
                    avg_rot_err = eval_metrics['rotation_error'].mean().item()
                    avg_trans_err = eval_metrics['translation_error'].mean().item()
                    avg_inliers = eval_metrics['inlier_ratio'].mean().item()
                    avg_reproj = eval_metrics['reprojection_error'].mean().item()
                    
                    auc_easy = eval_metrics['auc_pose']['easy']
                    auc_medium = eval_metrics['auc_pose']['medium']
                    auc_strict = eval_metrics['auc_pose']['strict']
                    
                    print(f"Pose Translation Error: {avg_trans_err:.4f}m | Rotation Error: {avg_rot_err:.2f} deg")
                    print(f"AUC Pose (Easy/Med/Strict): {auc_easy:.2f} / {auc_medium:.2f} / {auc_strict:.2f}")
                    
                    wandb.log({
                        "val/loss": avg_val_loss,
                        "val/rotation_error": avg_rot_err,
                        "val/translation_error": avg_trans_err,
                        "val/inlier_ratio": avg_inliers,
                        "val/reprojection_error": avg_reproj,
                        "val/auc_easy": auc_easy,
                        "val/auc_medium": auc_medium,
                        "val/auc_strict": auc_strict,
                        "epoch": epoch
                    })
                else:
                    print("Validation Warning: No matches found during validation. Skipping pose error evaluation.")
                    wandb.log({"val/loss": avg_val_loss, "epoch": epoch})
            
            # Save best checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(checkpoint_dir, "best_model.pth")
                torch.save(model.state_dict(), best_path)
                print(f"Saved new best model checkpoint to {best_path}")

        # Save regular checkpoints
        if epoch % cfg.training.save_freq == 0:
            epoch_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), epoch_path)
            print(f"Saved regular model checkpoint to {epoch_path}")

    # 11. Cleanup temporary directories in dry-run mode
    if dry_run:
        import shutil
        print("Cleaning up dry-run dummy data cache...")
        shutil.rmtree(dummy_dir, ignore_errors=True)

    wandb.finish()


if __name__ == "__main__":
    main()
