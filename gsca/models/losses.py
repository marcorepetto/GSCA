import torch
import torch.nn as nn
import torch.nn.functional as F

class CircleLossWithSelfPacedWeighting(nn.Module):
    """
    Circle Loss with Self-paced Weighting for 2D-3D dense descriptors.
    """
    def __init__(self, gamma: float = 80.0, margin: float = 0.25):
        super().__init__()
        if gamma <= 0:
            raise ValueError("gamma must be greater than 0")
        if not (0.0 < margin < 1.0):
            raise ValueError("margin must be strictly between 0.0 and 1.0")
            
        self.gamma = gamma
        self.margin = margin
        
        # Theoretical values
        self.O_p = 1.0 + margin
        self.O_n = -margin
        self.delta_p = 1.0 - margin
        self.delta_n = margin
        
    def forward(self, feat_2d: torch.Tensor, feat_3d: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_2d: torch.Tensor of shape [B, N_2D, C] (L2 normalized)
            feat_3d: torch.Tensor of shape [B, N_3D, C] (L2 normalized)
            gt_mask: torch.Tensor of shape [B, N_2D, N_3D] (boolean)
        Returns:
            scalar torch.Tensor (shape [])
        """
        # Input validations
        if not isinstance(feat_2d, torch.Tensor):
            raise ValueError("feat_2d must be a torch.Tensor")
        if not isinstance(feat_3d, torch.Tensor):
            raise ValueError("feat_3d must be a torch.Tensor")
        if not isinstance(gt_mask, torch.Tensor):
            raise ValueError("gt_mask must be a torch.Tensor")
            
        if feat_2d.dtype != torch.float32:
            raise ValueError("feat_2d must be float32")
        if feat_3d.dtype != torch.float32:
            raise ValueError("feat_3d must be float32")
        if gt_mask.dtype != torch.bool:
            raise ValueError("gt_mask must be boolean")
            
        if feat_2d.ndim != 3 or feat_3d.ndim != 3 or gt_mask.ndim != 3:
            raise ValueError("All input tensors must have 3 dimensions")
            
        B, N_2D, C = feat_2d.shape
        _, N_3D, _ = feat_3d.shape
        
        if gt_mask.shape != (B, N_2D, N_3D):
            raise ValueError(f"gt_mask shape {gt_mask.shape} does not match (B, N_2D, N_3D) = {(B, N_2D, N_3D)}")
            
        batch_losses = []
        for b in range(B):
            # Compute similarity matrix
            sim_matrix = torch.matmul(feat_2d[b], feat_3d[b].t())
            
            # Separate positive and negative similarities
            s_p = sim_matrix[gt_mask[b]]
            s_n = sim_matrix[~gt_mask[b]]
            
            # Check edge cases
            if s_p.numel() == 0 or s_n.numel() == 0:
                loss_b = torch.tensor(0.0, device=feat_2d.device, dtype=feat_2d.dtype, requires_grad=True)
                batch_losses.append(loss_b)
                continue
                
            # Dynamic weights with gradient detachment
            alpha_p = torch.clamp(self.O_p - s_p.detach(), min=0.0)
            alpha_n = torch.clamp(s_n.detach() - self.O_n, min=0.0)
            self.last_alpha_p = alpha_p
            self.last_alpha_n = alpha_n
            
            # Scaled logits
            logits_p = -self.gamma * alpha_p * (s_p - self.delta_p)
            logits_n = self.gamma * alpha_n * (s_n - self.delta_n)
            
            # Stable logsumexp (normalized by number of pairs to prevent oscillation across batches)
            loss_p = torch.logsumexp(logits_p, dim=0) - torch.log(torch.tensor(s_p.numel(), dtype=feat_2d.dtype, device=feat_2d.device))
            loss_n = torch.logsumexp(logits_n, dim=0) - torch.log(torch.tensor(s_n.numel(), dtype=feat_2d.dtype, device=feat_2d.device))
            
            # Stable loss per batch element: log1p(exp(loss_p + loss_n))
            x = loss_p + loss_n
            loss_b = torch.clamp(x, min=0.0) + torch.log1p(torch.exp(-torch.abs(x)))
            batch_losses.append(loss_b)
            
        return torch.stack(batch_losses).mean()
