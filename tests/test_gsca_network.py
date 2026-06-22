import pytest
import torch
from gsca.models.gsca_network import GSCANetwork


def test_gsca_network_forward_and_backward():
    # 1. Initialize device (CPU for testing)
    device = torch.device("cpu")

    # 2. Instantiate GSCANetwork (using light DINOv2 backbone for testing)
    model = GSCANetwork(
        backbone_name="dinov2_vits14",
        bottleneck_dim=64,
        intermediate_layers=[2, 5, 8, 11],
        out_channels=256,
        k=5,
        pretrained=False,  # Avoid downloading weights during unit testing
    ).to(device)

    # 3. Freeze backbone parameters as done in configure_optimizers
    for name, param in model.named_parameters():
        if ("dinov2" in name or "backbone" in name) and "adapter" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # 4. Create dummy batch inputs (Batch size B = 2)
    B = 2
    N_2D = 30
    N_3D_1 = 20
    N_3D_2 = 25
    N_total = N_3D_1 + N_3D_2

    images = torch.randn(B, 3, 224, 224, device=device)
    coords_2d = torch.rand(B, N_2D, 2, device=device) * 2.0 - 1.0  # Range [-1, 1]

    # Sparse PyG point representation
    pos_1 = torch.randn(N_3D_1, 3, device=device)
    pos_2 = torch.randn(N_3D_2, 3, device=device)
    pos = torch.cat([pos_1, pos_2], dim=0)

    batch = torch.cat([
        torch.zeros(N_3D_1, dtype=torch.long, device=device),
        torch.ones(N_3D_2, dtype=torch.long, device=device)
    ])

    # Intrinsics and pose priors
    K_cam = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    K_cam[:, 0, 0] = 100.0  # focal length x
    K_cam[:, 1, 1] = 100.0  # focal length y
    K_cam[:, 0, 2] = 112.0  # principal point x
    K_cam[:, 1, 2] = 112.0  # principal point y

    R_prior = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    t_prior = torch.zeros(B, 3, 1, device=device)
    t_prior[:, 2, 0] = 5.0  # Move camera 5 meters back (Z axis)

    # Normals
    normals_2d = torch.randn(B, N_2D, 3, device=device)
    normals_2d = torch.nn.functional.normalize(normals_2d, p=2, dim=-1)

    normals_3d_1 = torch.randn(N_3D_1, 3, device=device)
    normals_3d_2 = torch.randn(N_3D_2, 3, device=device)
    normals_3d = torch.cat([normals_3d_1, normals_3d_2], dim=0)
    normals_3d = torch.nn.functional.normalize(normals_3d, p=2, dim=-1)

    # 5. Run forward pass
    feat_2d_refined, feat_3d_dense = model(
        images=images,
        coords_2d=coords_2d,
        pos=pos,
        batch=batch,
        K_cam=K_cam,
        R_prior=R_prior,
        t_prior=t_prior,
        normals_2d=normals_2d,
        normals_3d=normals_3d,
        delta=50.0,
        tau=0.1,
        near_plane=0.1,
    )

    # 6. Verify output shapes
    # Max nodes per batch sample is max(20, 25) = 25
    assert feat_2d_refined.shape == (B, N_2D, 256), f"Wrong 2D refined shape: {feat_2d_refined.shape}"
    assert feat_3d_dense.shape == (B, 25, 256), f"Wrong 3D dense shape: {feat_3d_dense.shape}"

    # 7. Run backward pass
    loss = feat_2d_refined.sum() + feat_3d_dense.sum()
    loss.backward()

    # 8. Assert gradient flow is correct and frozen parameters are untouched
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient but is trainable."
            assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients."
        else:
            assert param.grad is None, f"Parameter {name} has gradient but is frozen."
