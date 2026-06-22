import pytest
import torch
import torch.nn.functional as F
from gsca.models.losses import CircleLossWithSelfPacedWeighting

def test_perfect_alignment():
    """
    T-LOSS-02: Perfect alignment.
    Check that alpha_p weights are small (equal to margin) and loss is under 0.05.
    """
    loss_fn = CircleLossWithSelfPacedWeighting(gamma=80.0, margin=0.25)
    
    # 2D and 3D descriptors are identical for matching pairs, and orthogonal for non-matching.
    feat_2d = torch.tensor([[[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]]], dtype=torch.float32)
                             
    feat_3d = torch.tensor([[[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]]], dtype=torch.float32)
                             
    gt_mask = torch.tensor([[[True, False, False],
                             [False, True, False],
                             [False, False, True]]], dtype=torch.bool)
                             
    loss = loss_fn(feat_2d, feat_3d, gt_mask)
    
    # Assert Op = 1.25, Sp = 1.0 -> alpha_p = 0.25 (since margin = 0.25)
    assert hasattr(loss_fn, 'last_alpha_p')
    assert torch.allclose(loss_fn.last_alpha_p, torch.tensor(0.25), atol=1e-5)
    assert torch.allclose(loss_fn.last_alpha_n, torch.tensor(0.25), atol=1e-5)
    
    # Check that loss is small
    assert loss.item() < 0.05

def test_maximal_misalignment():
    """
    T-LOSS-02: Maximal misalignment.
    Check that loss is large and gradients propagate with high magnitude when maximum error occurs.
    """
    loss_fn = CircleLossWithSelfPacedWeighting(gamma=80.0, margin=0.25)
    
    # Sp = -1.0, Sn = 1.0
    feat_2d = torch.tensor([[[1.0, 0.0],
                             [1.0, 0.0]]], dtype=torch.float32)
    feat_3d = torch.tensor([[[-1.0, 0.0],
                             [1.0, 0.0]]], dtype=torch.float32)
    gt_mask = torch.tensor([[[True, False],
                             [False, False]]], dtype=torch.bool)
                             
    feat_2d.requires_grad = True
    feat_3d.requires_grad = True
    
    loss = loss_fn(feat_2d, feat_3d, gt_mask)
    loss.backward()
    
    assert loss.item() > 10.0
    assert feat_2d.grad is not None
    assert feat_3d.grad is not None
    assert torch.norm(feat_2d.grad) > 1.0
    assert torch.norm(feat_3d.grad) > 1.0

def test_numerical_stability():
    """
    T-LOSS-02: Numerical stability.
    Check that a huge gamma scale does not cause NaN/inf values.
    """
    loss_fn = CircleLossWithSelfPacedWeighting(gamma=1000.0, margin=0.25)
    
    feat_2d = torch.randn(2, 5, 128)
    feat_3d = torch.randn(2, 5, 128)
    feat_2d = F.normalize(feat_2d, p=2, dim=-1)
    feat_3d = F.normalize(feat_3d, p=2, dim=-1)
    
    gt_mask = torch.rand(2, 5, 5) > 0.5
    
    loss = loss_fn(feat_2d, feat_3d, gt_mask)
    
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)

def test_corner_cases_mask():
    """
    T-LOSS-02: Corner cases.
    Check behavior with all False and all True masks (empty vs full).
    """
    loss_fn = CircleLossWithSelfPacedWeighting(gamma=80.0, margin=0.25)
    
    feat_2d = torch.randn(2, 5, 128, requires_grad=True)
    feat_3d = torch.randn(2, 5, 128, requires_grad=True)
    feat_2d_norm = F.normalize(feat_2d, p=2, dim=-1)
    feat_3d_norm = F.normalize(feat_3d, p=2, dim=-1)
    
    # All False mask (empty)
    mask_empty = torch.zeros(2, 5, 5, dtype=torch.bool)
    loss_empty = loss_fn(feat_2d_norm, feat_3d_norm, mask_empty)
    assert loss_empty.item() == 0.0
    loss_empty.backward()
    
    # Reset grads
    feat_2d.grad = None
    feat_3d.grad = None
    
    # All True mask (full)
    mask_full = torch.ones(2, 5, 5, dtype=torch.bool)
    loss_full = loss_fn(feat_2d_norm, feat_3d_norm, mask_full)
    assert loss_full.item() == 0.0
    loss_full.backward()

def test_gradient_propagation():
    """
    T-LOSS-02: Gradient propagation.
    Verify that gradients are successfully propagated and match the input tensor shape.
    """
    loss_fn = CircleLossWithSelfPacedWeighting(gamma=80.0, margin=0.25)
    
    feat_2d = torch.randn(2, 4, 32, requires_grad=True)
    feat_3d = torch.randn(2, 4, 32, requires_grad=True)
    feat_2d_norm = F.normalize(feat_2d, p=2, dim=-1)
    feat_3d_norm = F.normalize(feat_3d, p=2, dim=-1)
    
    gt_mask = torch.eye(4, dtype=torch.bool).unsqueeze(0).repeat(2, 1, 1)
    
    loss = loss_fn(feat_2d_norm, feat_3d_norm, gt_mask)
    loss.backward()
    
    assert feat_2d.grad is not None
    assert feat_3d.grad is not None
    assert feat_2d.grad.shape == feat_2d.shape
    assert feat_3d.grad.shape == feat_3d.shape
    assert not torch.isnan(feat_2d.grad).any()
    assert not torch.isnan(feat_3d.grad).any()
