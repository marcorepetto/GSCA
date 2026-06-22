import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from gsca.models.visual_branch import (
    VisualAdapter,
    AdaptedTransformerBlock,
    load_and_adapt_dinov2,
    VisualFPN,
    Visual2DBranch
)

def test_output_dimensions():
    """
    T-VISUAL-05: test_output_dimensions
    Probar con combinaciones: lote B in {1, 4}, H, W in {(224, 224), (518, 518)}, y N in {50, 150}.
    Comprobar las formas resultantes de dense_descriptors ([B, out_dim, H, W]) y
    sampled_descriptors ([B, N, out_dim]).
    """
    out_dim = 128
    model = Visual2DBranch(
        backbone_name='dinov2_vits14',
        bottleneck_dim=64,
        intermediate_layers=[2, 5, 8, 11],
        out_dim=out_dim,
        pretrained=True
    )
    model.eval()
    
    combinations = [
        (1, 224, 50),
        (4, 224, 150),
        (1, 518, 50),
        (4, 518, 150),
    ]
    
    for B, size, N in combinations:
        images = torch.randn(B, 3, size, size)
        coords = torch.rand(B, N, 2) * 2.0 - 1.0  # normalized to [-1, 1]
        
        with torch.no_grad():
            dense_desc, sampled_desc = model(images, coords)
            
        assert dense_desc.shape == (B, out_dim, size, size), f"Expected shape {(B, out_dim, size, size)}, got {dense_desc.shape}"
        assert sampled_desc.shape == (B, N, out_dim), f"Expected shape {(B, N, out_dim)}, got {sampled_desc.shape}"
        
        # Verify L2 normalization of sampled descriptors (norm is exactly 1.0)
        norms = torch.norm(sampled_desc, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Sampled descriptors are not L2 normalized"


def test_gradient_flow_and_frozen_weights():
    """
    T-VISUAL-05: test_gradient_flow_and_frozen_weights
    Todos los parámetros pertenecientes al backbone base de DINOv2 (que no sean adaptadores)
    deben tener .grad como None.
    Todos los parámetros de VisualAdapter y VisualFPN deben tener .grad distinto de None y norma mayor a cero.
    """
    model = Visual2DBranch(
        backbone_name='dinov2_vits14',
        bottleneck_dim=64,
        intermediate_layers=[2, 5, 8, 11],
        out_dim=128,
        pretrained=True
    )
    model.train()
    
    B, N = 2, 50
    images = torch.randn(B, 3, 224, 224)
    coords = torch.rand(B, N, 2) * 2.0 - 1.0
    
    # Perturb the adapter's up_proj weights slightly to ensure gradients flow to down_proj
    # (since at exact initialization up_proj.weight is zero, making down_proj gradient zero)
    for name, param in model.backbone.named_parameters():
        if 'adapter.up_proj' in name:
            param.data.fill_(0.01)
            
    # Forward and backward
    dense_desc, sampled_desc = model(images, coords)
    loss = sampled_desc.sum()
    loss.backward()
    
    # Verify gradient flow in backbone
    for name, param in model.backbone.named_parameters():
        if 'adapter' in name:
            assert param.grad is not None, f"Adapter parameter '{name}' did not receive gradients"
            assert torch.norm(param.grad) > 0.0, f"Adapter parameter '{name}' has zero gradient"
        else:
            assert param.grad is None, f"Frozen DINOv2 parameter '{name}' received gradients: {param.grad}"
            
    # Verify gradient flow in FPN
    for name, param in model.fpn.named_parameters():
        assert param.grad is not None, f"FPN parameter '{name}' did not receive gradients"
        assert torch.norm(param.grad) > 0.0, f"FPN parameter '{name}' has zero gradient"


def test_adapter_identity_on_init():
    """
    T-VISUAL-05: test_adapter_identity_on_init
    Comparar la salida de un bloque Transformer original y su bloque adaptado correspondiente.
    La diferencia absoluta máxima entre ambos tensores de salida debe ser menor que 10^-7.
    """
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
    original_block = backbone.blocks[2]
    original_block.eval()
    
    # Freeze block parameters
    for param in original_block.parameters():
        param.requires_grad = False
        
    embed_dim = original_block.mlp.fc1.in_features
    adapter = VisualAdapter(embed_dim=embed_dim, bottleneck_dim=64)
    adapted_block = AdaptedTransformerBlock(original_block, adapter)
    adapted_block.eval()
    
    # Dummy input sequence [B, L, embed_dim]
    x = torch.randn(2, 257, embed_dim)
    
    with torch.no_grad():
        out_orig = original_block(x)
        out_adapt = adapted_block(x)
        
    max_diff = torch.max(torch.abs(out_orig - out_adapt))
    assert max_diff < 1e-7, f"Adapter identity check failed. Max diff: {max_diff}"


def test_robust_sampling_coordinates():
    """
    T-VISUAL-05: test_robust_sampling_coordinates
    Pasar un lote de coordenadas límite como [-1.0, -1.0], [1.0, 1.0] y fuera de límites
    como [-1.1, 1.2] a coords_2d.
    Verificar que la salida no contenga valores NaN o inf.
    Verificar que los descriptores resultantes de las coordenadas fuera del rango [-1.0, 1.0]
    sean cero (debido a padding_mode='zeros' de grid_sample).
    """
    model = Visual2DBranch(
        backbone_name='dinov2_vits14',
        bottleneck_dim=64,
        intermediate_layers=[2, 5, 8, 11],
        out_dim=128,
        pretrained=True
    )
    model.eval()
    
    B = 2
    # Coordinates configuration:
    # 0: [-1.0, -1.0] (boundary)
    # 1: [1.0, 1.0] (boundary)
    # 2: [-1.1, 1.2] (out of bounds)
    # 3: [0.0, 0.0] (inside bounds)
    # 4: [1.5, -2.0] (out of bounds)
    coords = torch.tensor([
        [[-1.0, -1.0], [1.0, 1.0], [-1.1, 1.2], [0.0, 0.0], [1.5, -2.0]]
    ]).repeat(B, 1, 1)
    
    images = torch.randn(B, 3, 224, 224)
    
    with torch.no_grad():
        dense_desc, sampled_desc = model(images, coords)
        
    assert not torch.isnan(sampled_desc).any(), "Output contains NaN values"
    assert not torch.isinf(sampled_desc).any(), "Output contains inf values"
    
    # Elements at index 2 and 4 are out of bounds. Under padding_mode='zeros',
    # their descriptors should be exactly 0.0 (since they sample outside the grid)
    for b in range(B):
        # Index 2 [-1.1, 1.2]
        assert torch.all(sampled_desc[b, 2] == 0.0), f"Sampled descriptor for [-1.1, 1.2] was not zero: {sampled_desc[b, 2]}"
        # Index 4 [1.5, -2.0]
        assert torch.all(sampled_desc[b, 4] == 0.0), f"Sampled descriptor for [1.5, -2.0] was not zero: {sampled_desc[b, 4]}"


def test_batching_invariance():
    """
    T-VISUAL-05: test_batching_invariance
    Comparar el descriptor en la posición k del lote obtenido al procesar una imagen de forma aislada,
    versus el obtenido cuando es parte de un lote de tamaño B > 1.
    La diferencia absoluta máxima debe ser inferior a 10^-6.
    """
    model = Visual2DBranch(
        backbone_name='dinov2_vits14',
        bottleneck_dim=64,
        intermediate_layers=[2, 5, 8, 11],
        out_dim=128,
        pretrained=True
    )
    model.eval()
    
    # Images and coordinates
    img1 = torch.randn(1, 3, 224, 224)
    img2 = torch.randn(1, 3, 224, 224)
    batch_imgs = torch.cat([img1, img2], dim=0)
    
    coords1 = torch.rand(1, 10, 2) * 2.0 - 1.0
    coords2 = torch.rand(1, 10, 2) * 2.0 - 1.0
    batch_coords = torch.cat([coords1, coords2], dim=0)
    
    with torch.no_grad():
        # Process in isolation
        dense_isolated, sampled_isolated = model(img1, coords1)
        # Process in batch
        dense_batch, sampled_batch = model(batch_imgs, batch_coords)
        
    diff_dense = torch.max(torch.abs(dense_isolated - dense_batch[0:1]))
    diff_sampled = torch.max(torch.abs(sampled_isolated - sampled_batch[0:1]))
    
    assert diff_dense < 1e-6, f"Batching variance in dense descriptors: {diff_dense}"
    assert diff_sampled < 1e-6, f"Batching variance in sampled descriptors: {diff_sampled}"
