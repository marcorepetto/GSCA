import pytest
import torch
from gsca.models.geometric import GeometricFeatureExtractor

def test_input_validation():
    model = GeometricFeatureExtractor(k=5, out_channels=128)
    
    # 1. Non-tensor inputs
    with pytest.raises(ValueError, match="pos must be a torch.Tensor"):
        model("not_a_tensor", torch.zeros(10, dtype=torch.long))
    with pytest.raises(ValueError, match="batch must be a torch.Tensor"):
        model(torch.randn(10, 3), "not_a_tensor")
        
    # 2. Dimensions not 3D
    with pytest.raises(ValueError, match="ndim == 2"):
        model(torch.randn(10, 3, 1), torch.zeros(10, dtype=torch.long))
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(10, 2), torch.zeros(10, dtype=torch.long))
        
    # 3. Mismatched length
    with pytest.raises(ValueError, match="same number of points"):
        model(torch.randn(10, 3), torch.zeros(9, dtype=torch.long))

def test_permutation_invariance():
    # T-GEOM-05: Invariancia a la Permutación
    torch.manual_seed(42)
    N = 100
    k = 10
    out_channels = 256
    model = GeometricFeatureExtractor(k=k, out_channels=out_channels)
    model.eval()  # Put in eval mode to avoid BatchNorm updates causing differences
    
    # Generate point cloud and batch tensor
    pos_A = torch.randn(N, 3)
    batch_A = torch.zeros(N, dtype=torch.long)
    
    # Random permutation
    P = torch.randperm(N)
    pos_B = pos_A[P]
    batch_B = batch_A[P]
    
    with torch.no_grad():
        feat_A = model(pos_A, batch_A)
        feat_B = model(pos_B, batch_B)
        
    # Assert permutation invariance: feat_A[P] == feat_B
    assert torch.allclose(feat_A[P], feat_B, atol=1e-5)

def test_translation_invariance():
    # T-GEOM-05: Invariancia a la Traslación Global
    torch.manual_seed(42)
    N = 100
    k = 10
    out_channels = 256
    model = GeometricFeatureExtractor(k=k, out_channels=out_channels)
    model.eval()
    
    pos = torch.randn(N, 3)
    batch = torch.zeros(N, dtype=torch.long)
    
    # Global translation vector
    t = torch.randn(1, 3)
    pos_trans = pos + t
    
    with torch.no_grad():
        feat_orig = model(pos, batch)
        feat_trans = model(pos_trans, batch)
        
    assert torch.allclose(feat_orig, feat_trans, atol=1e-5)

def test_consistency_and_normalization():
    # T-GEOM-06: Consistencia y Normalización
    torch.manual_seed(42)
    N = 150
    k = 15
    
    for out_channels in [128, 256]:
        model = GeometricFeatureExtractor(k=k, out_channels=out_channels)
        model.eval()
        
        pos = torch.randn(N, 3)
        batch = torch.zeros(N, dtype=torch.long)
        
        with torch.no_grad():
            feat = model(pos, batch)
            
        # Shape verification
        assert feat.shape == (N, out_channels)
        assert feat.dtype == torch.float32
        
        # L2 norm verification
        norms = torch.norm(feat, p=2, dim=-1)
        expected_norms = torch.ones_like(norms)
        assert torch.allclose(norms, expected_norms, atol=1e-6)

def test_batch_isolation():
    # T-GEOM-06: Aislamiento del Batch (No filtración)
    torch.manual_seed(42)
    N1 = 60
    N2 = 80
    k = 10
    model = GeometricFeatureExtractor(k=k, out_channels=256)
    model.eval()
    
    pos_1 = torch.randn(N1, 3)
    pos_2 = torch.randn(N2, 3)
    
    batch_1 = torch.zeros(N1, dtype=torch.long)
    
    # Run pos_1 in isolation
    with torch.no_grad():
        feat_solo_1 = model(pos_1, batch_1)
        
    # Packaging both clouds into a single batch
    pos_all = torch.cat([pos_1, pos_2], dim=0)
    batch_all = torch.cat([
        torch.zeros(N1, dtype=torch.long),
        torch.ones(N2, dtype=torch.long)
    ])
    
    # Run combined batch
    with torch.no_grad():
        feat_batch_all = model(pos_all, batch_all)
        
    feat_batch_1 = feat_batch_all[:N1]
    
    # Verify no influence from cloud 2
    assert torch.allclose(feat_solo_1, feat_batch_1, atol=1e-6)

def test_gradient_integrity():
    # T-GEOM-07: Estabilidad e Integridad de Gradientes
    torch.manual_seed(42)
    N = 50
    k = 5
    model = GeometricFeatureExtractor(k=k, out_channels=256)
    model.train()  # Put in train mode for backward pass
    
    pos = torch.randn(N, 3)
    batch = torch.zeros(N, dtype=torch.long)
    
    # Run forward pass
    feat_3d = model(pos, batch)
    
    # Calculate scalar loss
    loss = feat_3d.sum()
    
    # Backward pass
    loss.backward()
    
    # Check that all parameters have requires_grad=True and non-null, stable gradients
    for name, param in model.named_parameters():
        assert param.requires_grad, f"Parameter {name} does not require gradients"
        assert param.grad is not None, f"Gradient for parameter {name} is None"
        assert not torch.isnan(param.grad).any(), f"Gradient for parameter {name} contains NaN values"
        assert not torch.isinf(param.grad).any(), f"Gradient for parameter {name} contains infinite values"
