import pytest
import torch
import torch.nn as nn
import math
from gsca.training import configure_optimizers

def test_parameter_freezing():
    """
    T-TRAIN-02: Parameter freezing.
    Verify that parameters belonging to DINOv2 (names containing 'dinov2' or 'backbone' without 'adapter')
    are set to requires_grad = False, while adapters, fpn, and dgcnn are trainable.
    """
    class MockGSCAModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.dinov2 = nn.Linear(10, 10)
            self.backbone.adapter = nn.Linear(5, 5)
            self.visual_adapters = nn.Linear(10, 10)
            self.fpn_decoder = nn.Linear(10, 10)
            self.dgcnn = nn.Linear(10, 10)
            
    model = MockGSCAModel()
    optimizer, scheduler = configure_optimizers(model, epochs=10, steps_per_epoch=100)
    
    # Check that backbone.dinov2 parameters are frozen
    for name, param in model.backbone.dinov2.named_parameters():
        assert not param.requires_grad, f"Parameter '{name}' should be frozen"
        
    # Check that others are trainable
    for name, param in model.backbone.adapter.named_parameters():
        assert param.requires_grad, f"Parameter '{name}' should be trainable"
    for name, param in model.visual_adapters.named_parameters():
        assert param.requires_grad, f"Parameter '{name}' should be trainable"
    for name, param in model.fpn_decoder.named_parameters():
        assert param.requires_grad, f"Parameter '{name}' should be trainable"
    for name, param in model.dgcnn.named_parameters():
        assert param.requires_grad, f"Parameter '{name}' should be trainable"
        
    # Verify optimizer parameters
    opt_params = set()
    for group in optimizer.param_groups:
        for p in group['params']:
            opt_params.add(p)
            
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(opt_params) == len(trainable_params), "Optimizer parameters don't match trainable parameters"
    for p in trainable_params:
        assert p in opt_params, "Trainable parameter not in optimizer"


def test_learning_rate_schedule():
    """
    T-TRAIN-02: Learning rate schedule.
    Verify that the learning rate warm-up and cosine decay curves follow the mathematical specs.
    """
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(1))
            
    model = SimpleModel()
    
    epochs = 10
    steps_per_epoch = 100
    optimizer, scheduler = configure_optimizers(model, epochs=epochs, steps_per_epoch=steps_per_epoch)
    
    lrs = []
    for step in range(1001):
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)
        scheduler.step()
        
    # Step 0: lr is approx eta_min (1e-6)
    assert abs(lrs[0] - 1e-6) < 1e-7
    
    # Step 500 (end of warmup at 5 epochs): lr is exactly eta_base (1e-4)
    assert abs(lrs[500] - 1e-4) < 1e-7
    
    # Step 1000 (end of training at 10 epochs): lr is exactly eta_min (1e-6)
    assert abs(lrs[1000] - 1e-6) < 1e-7
    
    # Steps 500 to 1000: Cosine decay curve checks
    # 1. Monotonic decrease
    for i in range(500, 1000):
        assert lrs[i] >= lrs[i+1], f"LR did not decrease at step {i}: {lrs[i]} -> {lrs[i+1]}"
        
    # 2. Non-linear decay (differences are not all equal)
    diffs = [lrs[i] - lrs[i+1] for i in range(500, 1000)]
    assert len(set(diffs)) > 1, "LR decay is linear"
    
    # 3. Symmetry check: eta_{500+x} + eta_{1000-x} == eta_base + eta_min
    for x in range(1, 250):
        sum_lr = lrs[500 + x] + lrs[1000 - x]
        assert abs(sum_lr - (1e-4 + 1e-6)) < 1e-7, f"Symmetry failed for x={x}: {lrs[500+x]} + {lrs[1000-x]} = {sum_lr}"
