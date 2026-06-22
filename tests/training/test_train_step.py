import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from unittest.mock import MagicMock
from gsca.training import train_step, configure_optimizers

def test_train_step_orchestration():
    """
    T-TRAIN-04: Orchestration verification using mocks.
    Verify that all operations are performed in the correct order.
    """
    manager = MagicMock()
    
    model = manager.model
    optimizer = manager.optimizer
    scheduler = manager.scheduler
    loss_fn = manager.loss_fn
    
    # Mock model outputs
    feat_2d_mock = torch.randn(2, 3, 4)
    feat_3d_mock = torch.randn(2, 3, 4)
    model.return_value = (feat_2d_mock, feat_3d_mock)
    
    # Mock loss output
    loss_mock = manager.loss
    loss_fn.return_value = loss_mock
    
    # Mock loss item value so we can check it in stats
    loss_mock.item.return_value = 0.5
    
    # Mock named_modules to avoid issues when checking for backbone
    model.named_modules.return_value = []
    
    # Mock scheduler lr return
    scheduler.get_last_lr.return_value = [0.0001]
    
    # Input batch
    batch = {
        'image': torch.randn(2, 3, 224, 224),
        'point_cloud': torch.randn(2, 100, 3),
        'gt_mask': torch.ones(2, 50, 100, dtype=torch.bool)
    }
    device = torch.device('cpu')
    
    # Execute train_step
    stats = train_step(model, batch, optimizer, scheduler, loss_fn, device)
    
    # Check stats dict
    assert stats == {'loss': 0.5, 'lr': 0.0001}
    
    # Inspect calls on the manager to verify order
    called_names = [call[0] for call in manager.mock_calls]
    
    expected_order = [
        'model.train',
        'optimizer.zero_grad',
        'model',
        'loss_fn',
        'loss.backward',
        'optimizer.step',
        'scheduler.step'
    ]
    
    # Check relative order
    current_idx = -1
    for name in expected_order:
        found = False
        for idx in range(current_idx + 1, len(called_names)):
            if called_names[idx] == name:
                current_idx = idx
                found = True
                break
        assert found, f"Expected call '{name}' was not found in the correct sequence of calls: {called_names}"


class MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_frozen = nn.Linear(4, 4)
        self.conv_trainable = nn.Linear(4, 4)
        
        # Freeze conv_frozen
        for p in self.conv_frozen.parameters():
            p.requires_grad = False
            
    def forward(self, image, point_cloud):
        # Process image input to [B, 4]
        x_img = image.mean(dim=(2, 3)) # [B, 3]
        x_img_pad = F.pad(x_img, (0, 1)) # [B, 4]
        
        f1 = self.conv_frozen(x_img_pad) # [B, 4]
        f2 = self.conv_trainable(x_img_pad) # [B, 4]
        
        # Stack to output shapes [B, 2, 4] for 2D descriptors (so we can have matching/non-matching pairs)
        feat_2d = torch.stack([f1 + f2, f1 - f2], dim=1)
        
        x_pc = point_cloud.mean(dim=1) # [B, 3]
        x_pc_pad = F.pad(x_pc, (0, 1)) # [B, 4]
        # Stack to output shapes [B, 2, 4] for 3D descriptors
        feat_3d = torch.stack([x_pc_pad, -x_pc_pad], dim=1)
        
        return feat_2d, feat_3d


def test_weight_updates():
    """
    T-TRAIN-04: Effective weight updates.
    Verify that only the trainable parameters are updated, and frozen parameters are unmodified.
    """
    model = MiniModel()
    optimizer, scheduler = configure_optimizers(model, epochs=10, steps_per_epoch=10)
    
    from gsca.models.losses import CircleLossWithSelfPacedWeighting
    loss_fn = CircleLossWithSelfPacedWeighting(gamma=10.0, margin=0.25)
    
    # Store deep copies of parameters before step
    frozen_weights_before = copy.deepcopy(model.conv_frozen.weight.data)
    trainable_weights_before = copy.deepcopy(model.conv_trainable.weight.data)
    
    # Inputs: must have a mix of matching and non-matching pairs (e.g. gt_mask has True and False)
    # gt_mask of shape [B, 2, 2]
    batch = {
        'image': torch.randn(2, 3, 8, 8),
        'point_cloud': torch.randn(2, 5, 3),
        'gt_mask': torch.tensor([[[True, False],
                                 [False, True]]], dtype=torch.bool).repeat(2, 1, 1)
    }
    
    # Execute train_step
    train_step(model, batch, optimizer, scheduler, loss_fn, torch.device('cpu'))
    
    # Verify that frozen weights did not change
    assert torch.equal(model.conv_frozen.weight.data, frozen_weights_before)
    
    # Verify that trainable weights did change
    assert not torch.allclose(model.conv_trainable.weight.data, trainable_weights_before)
