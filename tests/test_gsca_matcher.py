import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from gsca.models.gsca_matcher import (
    project_points,
    GeoStructuralCrossAttention,
    compute_mnn_matches
)

# Determine the device to run tests on dynamically (CPU/CUDA)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_dimensions_consistency():
    """
    T-GSCA-06: Prueba de consistencia de dimensiones.
    Verifica que las dimensiones resultantes de GeoStructuralCrossAttention
    sean exactamente [B, HW, C].
    """
    B = 2
    HW = 100
    N = 150
    C = 64
    
    feat_2d = torch.randn(B, HW, C, device=DEVICE)
    feat_3d = torch.randn(B, N, C, device=DEVICE)
    coords_2d = torch.randn(B, HW, 2, device=DEVICE)
    proj_coords = torch.randn(B, N, 2, device=DEVICE)
    proj_valid_mask = torch.ones(B, N, dtype=torch.bool, device=DEVICE)
    normals_2d = torch.randn(B, HW, 3, device=DEVICE)
    normals_3d = torch.randn(B, N, 3, device=DEVICE)
    
    model = GeoStructuralCrossAttention(channels=C).to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        out = model(
            feat_2d=feat_2d,
            feat_3d=feat_3d,
            coords_2d=coords_2d,
            proj_coords=proj_coords,
            proj_valid_mask=proj_valid_mask,
            normals_2d=normals_2d,
            normals_3d=normals_3d
        )
        
    assert out.shape == (B, HW, C)
    assert out.dtype == torch.float32


def test_2d_distance_filter():
    """
    T-GSCA-06: Prueba del filtro de distancia 2D (delta).
    Pixel query en [0.0, 0.0].
    Punto A en [15.0, 0.0] (distancia 15.0 <= delta=30.0).
    Punto B en [45.0, 0.0] (distancia 45.0 > delta=30.0).
    Verifica que el punto B tiene peso de atención 0.0, y por lo tanto
    el output es idéntico a la proyección de A.
    """
    B = 1
    HW = 1
    N = 2
    C = 8
    delta = 30.0
    
    coords_2d = torch.tensor([[[0.0, 0.0]]], device=DEVICE)
    proj_coords = torch.tensor([[[15.0, 0.0], [45.0, 0.0]]], device=DEVICE)
    proj_valid_mask = torch.tensor([[True, True]], device=DEVICE)
    
    normals_2d = torch.tensor([[[0.0, 0.0, 1.0]]], device=DEVICE)
    normals_3d = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], device=DEVICE)
    
    feat_2d = torch.randn(B, HW, C, device=DEVICE)
    feat_3d = torch.randn(B, N, C, device=DEVICE)
    
    feat_3d[0, 0, :] = 1.0   # Point A
    feat_3d[0, 1, :] = 100.0 # Point B
    
    model = GeoStructuralCrossAttention(channels=C).to(DEVICE)
    model.eval()
    
    # Set v_proj to identity projection
    with torch.no_grad():
        model.v_proj.weight.copy_(torch.eye(C, device=DEVICE))
        model.v_proj.bias.zero_()
        
    with torch.no_grad():
        out = model(
            feat_2d=feat_2d,
            feat_3d=feat_3d,
            coords_2d=coords_2d,
            proj_coords=proj_coords,
            proj_valid_mask=proj_valid_mask,
            normals_2d=normals_2d,
            normals_3d=normals_3d,
            delta=delta,
            tau=0.0
        )
        
    expected_out = feat_3d[:, 0, :]
    assert torch.allclose(out[:, 0, :], expected_out, atol=1e-5)


def test_normal_filter():
    """
    T-GSCA-06: Prueba del filtro de normales (tau).
    Query con normal [0, 0, 1].
    Punto A con normal [0, 0, 1] (similitud 1.0 >= tau=0.5).
    Punto B con normal [0, 0, -1] (similitud -1.0 < tau=0.5).
    Verifica que el punto B no tiene influencia (peso de atención 0.0).
    """
    B = 1
    HW = 1
    N = 2
    C = 8
    tau = 0.5
    
    coords_2d = torch.tensor([[[0.0, 0.0]]], device=DEVICE)
    proj_coords = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], device=DEVICE)
    proj_valid_mask = torch.tensor([[True, True]], device=DEVICE)
    
    normals_2d = torch.tensor([[[0.0, 0.0, 1.0]]], device=DEVICE)
    normals_3d = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]], device=DEVICE)
    
    feat_2d = torch.randn(B, HW, C, device=DEVICE)
    feat_3d = torch.randn(B, N, C, device=DEVICE)
    
    feat_3d[0, 0, :] = 2.0
    feat_3d[0, 1, :] = -20.0
    
    model = GeoStructuralCrossAttention(channels=C).to(DEVICE)
    model.eval()
    
    # Set v_proj to identity projection
    with torch.no_grad():
        model.v_proj.weight.copy_(torch.eye(C, device=DEVICE))
        model.v_proj.bias.zero_()
        
    with torch.no_grad():
        out = model(
            feat_2d=feat_2d,
            feat_3d=feat_3d,
            coords_2d=coords_2d,
            proj_coords=proj_coords,
            proj_valid_mask=proj_valid_mask,
            normals_2d=normals_2d,
            normals_3d=normals_3d,
            delta=30.0,
            tau=tau
        )
        
    expected_out = feat_3d[:, 0, :]
    assert torch.allclose(out[:, 0, :], expected_out, atol=1e-5)


def test_camera_clipping():
    """
    T-GSCA-06: Prueba de recorte de la cámara (near_plane).
    Verifica que proj_valid_mask para un punto detrás de la cámara (Z = -2.0)
    sea False, y que no ejerza influencia en la atención cruzada.
    """
    points_3d = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]]], dtype=torch.float32, device=DEVICE)
    K_cam = torch.eye(3, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    R_prior = torch.eye(3, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    t_prior = torch.zeros(1, 3, 1, dtype=torch.float32, device=DEVICE)
    
    proj_coords, proj_valid_mask = project_points(
        points_3d=points_3d,
        K_cam=K_cam,
        R_prior=R_prior,
        t_prior=t_prior,
        near_plane=0.1
    )
    
    # Point A has depth = 2.0 > 0.1 (True)
    # Point B has depth = -2.0 <= 0.1 (False)
    assert proj_valid_mask[0, 0].item() is True
    assert proj_valid_mask[0, 1].item() is False
    
    # Verification of coordinate values (using near_plane clamping for depth_safe)
    assert torch.allclose(proj_coords, torch.zeros_like(proj_coords), atol=1e-5)
    
    # Test in GeoStructuralCrossAttention
    C = 8
    feat_2d = torch.randn(1, 1, C, device=DEVICE)
    feat_3d = torch.randn(1, 2, C, device=DEVICE)
    feat_3d[0, 0, :] = 5.0
    feat_3d[0, 1, :] = -50.0
    
    coords_2d = torch.tensor([[[0.0, 0.0]]], device=DEVICE)
    normals_2d = torch.tensor([[[0.0, 0.0, 1.0]]], device=DEVICE)
    normals_3d = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]], device=DEVICE)
    
    model = GeoStructuralCrossAttention(channels=C).to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        model.v_proj.weight.copy_(torch.eye(C, device=DEVICE))
        model.v_proj.bias.zero_()
        
        out = model(
            feat_2d=feat_2d,
            feat_3d=feat_3d,
            coords_2d=coords_2d,
            proj_coords=proj_coords,
            proj_valid_mask=proj_valid_mask,
            normals_2d=normals_2d,
            normals_3d=normals_3d,
            delta=30.0,
            tau=0.0
        )
        
    expected_out = feat_3d[:, 0, :]
    assert torch.allclose(out[:, 0, :], expected_out, atol=1e-5)


def test_stability_prevention_of_nans():
    """
    T-GSCA-07: Prueba de estabilidad y prevención de NaNs.
    Verifica que no se generen NaNs o Infs cuando un query visual está
    muy alejado de todos los puntos 3D proyectados (haciendo que el 100% de la
    fila en la máscara sea marcada como inválida), y que el descriptor de salida
    sea cero numérico (< 1e-7).
    """
    B = 2
    HW = 3
    N = 4
    C = 8
    
    feat_2d = torch.randn(B, HW, C, device=DEVICE)
    feat_3d = torch.randn(B, N, C, device=DEVICE)
    
    # Distance is ~1414.2 pixels, which exceeds delta = 30.0
    coords_2d = torch.zeros(B, HW, 2, device=DEVICE)
    proj_coords = torch.full((B, N, 2), 1000.0, device=DEVICE)
    
    proj_valid_mask = torch.ones(B, N, dtype=torch.bool, device=DEVICE)
    normals_2d = torch.tensor([[[0.0, 0.0, 1.0]]], device=DEVICE).expand(B, HW, 3)
    normals_3d = torch.tensor([[[0.0, 0.0, 1.0]]], device=DEVICE).expand(B, N, 3)
    
    model = GeoStructuralCrossAttention(channels=C).to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        out = model(
            feat_2d=feat_2d,
            feat_3d=feat_3d,
            coords_2d=coords_2d,
            proj_coords=proj_coords,
            proj_valid_mask=proj_valid_mask,
            normals_2d=normals_2d,
            normals_3d=normals_3d,
            delta=30.0,
            tau=0.5
        )
        
    assert not torch.isnan(out).any(), "Output contains NaN values"
    assert not torch.isinf(out).any(), "Output contains Inf values"
    assert torch.all(torch.abs(out) < 1e-7)


def test_mnn_with_threshold():
    """
    T-GSCA-07: Prueba de MNN con umbral.
    Verifica la reciprocidad de vecindario mutuo y el filtrado por sim_threshold.
    """
    feat_2d = torch.tensor([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ], device=DEVICE)
    
    feat_3d = torch.tensor([
        [0.9, 0.0, 0.0, 0.0, 0.0, 0.43588989],  # Mutual NN with 0 (sim 0.9 >= 0.2)
        [0.0, 0.8, 0.0, 0.0, 0.0, 0.6],         # Mutual NN with 1 (sim 0.8 >= 0.2)
        [0.0, 0.0, 0.15, 0.0, 0.0, 0.98868599], # Mutual NN with 2, but sim 0.15 < 0.2 (filtered)
        [0.0, 0.0, 0.0, 0.3, 0.0, 0.9539392],    # Non-mutual (NN is 4)
        [0.0, 0.0, 0.0, 0.7, 0.9, 0.0]          # Mutual NN with 4 (sim ~0.789 >= 0.2)
    ], device=DEVICE)
    
    matches, scores = compute_mnn_matches(feat_2d, feat_3d, sim_threshold=0.2)
    
    assert matches.shape[0] == 3
    
    # Expected matches: (0, 0), (1, 1), (4, 4)
    expected_matches = torch.tensor([[0, 0], [1, 1], [4, 4]], dtype=torch.int64, device=DEVICE)
    assert torch.equal(matches, expected_matches)
    
    # Expected scores: 0.9, 0.8, 0.9/sqrt(1.3)
    expected_scores = torch.tensor([0.9, 0.8, 0.9 / math.sqrt(1.3)], dtype=torch.float32, device=DEVICE)
    assert torch.allclose(scores, expected_scores)


def test_gradient_flow():
    """
    T-GSCA-07: Prueba de flujo de gradientes (Backpropagation).
    Verifica que los gradientes fluyan correctamente a feat_2d y feat_3d sin NaNs ni Infs.
    """
    B = 2
    HW = 10
    N = 15
    C = 16
    
    feat_2d = torch.randn(B, HW, C, device=DEVICE, requires_grad=True)
    feat_3d = torch.randn(B, N, C, device=DEVICE, requires_grad=True)
    
    coords_2d = torch.randn(B, HW, 2, device=DEVICE)
    proj_coords = torch.randn(B, N, 2, device=DEVICE)
    proj_valid_mask = torch.ones(B, N, dtype=torch.bool, device=DEVICE)
    normals_2d = torch.randn(B, HW, 3, device=DEVICE)
    normals_3d = torch.randn(B, N, 3, device=DEVICE)
    
    model = GeoStructuralCrossAttention(channels=C).to(DEVICE)
    model.train()
    
    out = model(
        feat_2d=feat_2d,
        feat_3d=feat_3d,
        coords_2d=coords_2d,
        proj_coords=proj_coords,
        proj_valid_mask=proj_valid_mask,
        normals_2d=normals_2d,
        normals_3d=normals_3d,
        delta=30.0,
        tau=0.5
    )
    
    loss = out.sum()
    loss.backward()
    
    assert feat_2d.grad is not None, "feat_2d gradient is None"
    assert feat_3d.grad is not None, "feat_3d gradient is None"
    
    assert not torch.isnan(feat_2d.grad).any(), "feat_2d gradient contains NaN values"
    assert not torch.isnan(feat_3d.grad).any(), "feat_3d gradient contains NaN values"
    
    assert not torch.isinf(feat_2d.grad).any(), "feat_2d gradient contains Inf values"
    assert not torch.isinf(feat_3d.grad).any(), "feat_3d gradient contains Inf values"
    
    assert torch.norm(feat_2d.grad) > 0.0, "feat_2d gradient is zero"
    assert torch.norm(feat_3d.grad) > 0.0, "feat_3d gradient is zero"
