import pytest
import math
import torch
import os
from gsca.dataset import GSCADataset, get_gsca_dataloader
from gsca.models.gsca_network import GSCANetwork
from gsca.models.losses import CircleLossWithSelfPacedWeighting
from gsca.training import configure_optimizers, train_step


def test_gsca_dataset_and_dataloader_integration(tmp_path):
    # 1. Create dummy sample dictionary files
    sample_paths = []
    H, W = 224, 224
    N_2D = 30
    
    # We will generate 2 samples with different number of 3D points to test PyG collation and padding
    configs = [
        {"N_3D": 40, "filename": "sample_0.pt"},
        {"N_3D": 50, "filename": "sample_1.pt"}
    ]
    
    for config in configs:
        N_3D = config["N_3D"]
        # Generate random inputs matching actual sizes
        sample_data = {
            'image': torch.rand(3, H, W),             # image RGB in [0, 1]
            'normal_map': torch.randn(3, H, W),       # normal map
            'pos': torch.randn(N_3D, 3),              # 3D points
            'normals_3d': torch.randn(N_3D, 3),       # 3D normals
            'K_cam': torch.eye(3),                    # Intrinsics
            'pose_gt': torch.eye(4),                  # camera pose
            'coords_pixel_2d': torch.rand(N_2D, 2) * (H - 1),  # pixel coordinates
            'normals_2d': torch.randn(N_2D, 3)        # 2D normals
        }
        # Normalize normals
        sample_data['normal_map'] = torch.nn.functional.normalize(sample_data['normal_map'], p=2, dim=0)
        sample_data['normals_3d'] = torch.nn.functional.normalize(sample_data['normals_3d'], p=2, dim=-1)
        sample_data['normals_2d'] = torch.nn.functional.normalize(sample_data['normals_2d'], p=2, dim=-1)
        
        path = os.path.join(tmp_path, config["filename"])
        torch.save(sample_data, path)
        sample_paths.append(path)

    # 2. Instantiate GSCADataset with degradations enabled
    dataset = GSCADataset(
        sample_paths=sample_paths,
        degrade_points=True,
        degrade_visual=True,
        noise_std=0.03,
        downsample_ratio=0.8,      # downsample 3D points to 80%
        roughness_factor=0.1,
        pixel_match_threshold=5.0,
        near_plane=0.1
    )
    
    assert len(dataset) == 2

    # Test __getitem__
    sample = dataset[0]
    # Expected number of 3D points after downsampling: int(40 * 0.8) = 32
    assert sample['image'].shape == (3, H, W)
    assert sample['coords_2d'].shape == (N_2D, 2)
    assert sample['pos'].shape == (32, 3)
    assert sample['normals_3d'].shape == (32, 3)
    assert sample['gt_mask'].shape == (N_2D, 32)
    assert sample['gt_mask'].dtype == torch.bool

    # 3. Create get_gsca_dataloader
    dataloader = get_gsca_dataloader(dataset, batch_size=2, shuffle=False)
    
    # Get a batch
    batch = next(iter(dataloader))
    
    # Verify batched shapes
    # Max nodes expected in batch is max(int(40*0.8), int(50*0.8)) = max(32, 40) = 40
    assert batch['image'].shape == (2, 3, H, W)
    assert batch['coords_2d'].shape == (2, N_2D, 2)
    assert batch['normals_2d'].shape == (2, N_2D, 3)
    assert batch['K_cam'].shape == (2, 3, 3)
    assert batch['pose_gt'].shape == (2, 4, 4)
    assert batch['R_prior'].shape == (2, 3, 3)
    assert batch['t_prior'].shape == (2, 3, 1)
    
    # PyG concatenated points
    # Total nodes: 32 (sample 0) + 40 (sample 1) = 72
    assert batch['pos'].shape == (72, 3)
    assert batch['normals_3d'].shape == (72, 3)
    assert batch['batch'].shape == (72,)
    
    # Padded ground truth mask
    assert batch['gt_mask'].shape == (2, N_2D, 40)

    # 4. Perform an end-to-end forward/backward step with the actual train_step pipeline!
    model = GSCANetwork(
        backbone_name="dinov2_vits14",
        bottleneck_dim=64,
        intermediate_layers=[2, 5, 8, 11],
        out_channels=256,
        k=5,
        pretrained=False
    )
    
    optimizer, scheduler = configure_optimizers(model, epochs=1, steps_per_epoch=1)
    loss_fn = CircleLossWithSelfPacedWeighting(gamma=80.0, margin=0.25)
    
    stats = train_step(
        model=model,
        batch=batch,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=torch.device("cpu")
    )
    
    assert "loss" in stats
    assert "lr" in stats
    assert isinstance(stats["loss"], float)
    assert not math.isnan(stats["loss"])
