import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

class VisualAdapter(nn.Module):
    """
    Parallel bottleneck adapter (PEFT) to inject trainable parameters into
    the frozen DINOv2 ViT blocks.
    """
    def __init__(self, embed_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.act_fn = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming Uniform initialization for down projection
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        if self.down_proj.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.down_proj.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.down_proj.bias, -bound, bound)
        
        # Zero initialization for up projection to preserve identity at start
        nn.init.zeros_(self.up_proj.weight)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, L, embed_dim]
        # Output shape: [B, L, embed_dim]
        return self.up_proj(self.act_fn(self.down_proj(x)))


class AdaptedTransformerBlock(nn.Module):
    """
    Wrapper for a native DINOv2 block to run the VisualAdapter in parallel
    to the MLP layer.
    """
    def __init__(self, original_block: nn.Module, adapter: VisualAdapter):
        super().__init__()
        self.original_block = original_block
        self.adapter = adapter

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.original_block, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run original block's attention branch
        attn_out = self.original_block.attn(self.original_block.norm1(x))
        if hasattr(self.original_block, 'ls1'):
            attn_out = self.original_block.ls1(attn_out)
        if hasattr(self.original_block, 'drop_path1'):
            attn_out = self.original_block.drop_path1(attn_out)
        x_attn = x + attn_out

        # Get norm2 output from original block
        x_norm = self.original_block.norm2(x_attn)

        # Run adapter in parallel to original MLP
        adapter_out = self.adapter(x_norm)

        # Execute original block forward pass on x
        # This handles nested tensors, checkpointing, and stochastic depth natively.
        original_out = self.original_block(x)

        # Return the original block's output combined with the adapter output
        return original_out + adapter_out


def load_and_adapt_dinov2(
    backbone_name: str,
    bottleneck_dim: int,
    intermediate_layers: List[int],
    pretrained: bool = True
) -> nn.Module:
    """
    Loads pre-trained DINOv2, freezes all parameters, and wraps the specified
    intermediate transformer blocks with VisualAdapter modules.
    """
    # Load backbone model
    backbone = torch.hub.load('facebookresearch/dinov2', backbone_name, pretrained=pretrained)
    
    # Freeze all existing parameters globally
    for param in backbone.parameters():
        param.requires_grad = False
        
    # Replace chosen blocks with adapted blocks
    for idx in intermediate_layers:
        original_block = backbone.blocks[idx]
        
        # Get correct embed_dim
        if hasattr(original_block.mlp, 'fc1'):
            embed_dim = original_block.mlp.fc1.in_features
        elif hasattr(original_block.mlp, 'w12'):
            embed_dim = original_block.mlp.w12.in_features
        elif hasattr(original_block.norm1, 'normalized_shape'):
            embed_dim = original_block.norm1.normalized_shape[0]
        else:
            embed_dim = getattr(backbone, 'embed_dim', 384)
            
        adapter = VisualAdapter(embed_dim=embed_dim, bottleneck_dim=bottleneck_dim)
        
        # Ensure only the adapter parameters require gradients
        for param in adapter.parameters():
            param.requires_grad = True
            
        # Wrap the original block
        adapted_block = AdaptedTransformerBlock(original_block, adapter)
        backbone.blocks[idx] = adapted_block
        
    return backbone


class VisualFPN(nn.Module):
    """
    Feature Pyramid Network to reconstruct dense pixel-level descriptors
    from multiscale ViT block feature activations.
    """
    def __init__(self, embed_dim: int, fpn_dim: int, out_dim: int):
        super().__init__()
        # 1x1 projection convolutions to unify channel dimensions
        self.proj0 = nn.Conv2d(embed_dim, fpn_dim, kernel_size=1)
        self.proj1 = nn.Conv2d(embed_dim, fpn_dim, kernel_size=1)
        self.proj2 = nn.Conv2d(embed_dim, fpn_dim, kernel_size=1)
        self.proj3 = nn.Conv2d(embed_dim, fpn_dim, kernel_size=1)
        
        # Scale operators for spatial alignment (grid_size conversion)
        # F0 is at 1/14 scale -> project to 1/4 scale (transpose stride=4)
        self.scale0 = nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=4, stride=4)
        # F1 is at 1/14 scale -> project to 1/8 scale (transpose stride=2)
        self.scale1 = nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=2, stride=2)
        # F2 is at 1/14 scale -> keep native patch scale (Identity)
        # F3 is at 1/14 scale -> project to 1/28 scale (conv stride=2)
        self.scale3 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=2, padding=1)
        
        # Final prediction layer
        self.final_conv = nn.Conv2d(fpn_dim, out_dim, kernel_size=3, padding=1)
        
    def forward(
        self,
        features_list: List[torch.Tensor],
        target_shape: Tuple[int, int]
    ) -> torch.Tensor:
        # features_list: 4 tensors of shape [B, embed_dim, H_p, W_p]
        F0, F1, F2, F3 = features_list
        
        # Apply 1x1 projections and scale transformations
        P1 = self.scale0(self.proj0(F0))
        P2 = self.scale1(self.proj1(F1))
        P3 = self.proj2(F2)
        P4 = self.scale3(self.proj3(F3))
        
        # Top-down fusion with bilinear upsampling
        # P4 -> P3 (1/28 -> 1/14): upsample to exact size of P3
        P4_upsampled = F.interpolate(P4, size=P3.shape[2:], mode='bilinear', align_corners=False)
        P3_out = P3 + P4_upsampled
        
        # P3_out -> P2 (1/14 -> 1/8): upsample exact size of P2
        P3_out_upsampled = F.interpolate(P3_out, size=P2.shape[2:], mode='bilinear', align_corners=False)
        P2_out = P2 + P3_out_upsampled
        
        # P2_out -> P1 (1/8 -> 1/4): upsample exact size of P1
        P2_out_upsampled = F.interpolate(P2_out, size=P1.shape[2:], mode='bilinear', align_corners=False)
        P1_out = P1 + P2_out_upsampled
        
        # Map to final output channel dimension
        out = self.final_conv(P1_out)
        
        # Upsample back to original image resolution
        dense_descriptors = F.interpolate(out, size=target_shape, mode='bilinear', align_corners=False)
        return dense_descriptors


class Visual2DBranch(nn.Module):
    """
    Visual branch (2D) for dense descriptor extraction and bilinear coordinate sampling.
    """
    def __init__(
        self,
        backbone_name: str,
        bottleneck_dim: int,
        intermediate_layers: List[int],
        out_dim: int,
        pretrained: bool = True
    ):
        super().__init__()
        self.intermediate_layers = intermediate_layers
        
        # Initialize adapted backbone
        self.backbone = load_and_adapt_dinov2(
            backbone_name=backbone_name,
            bottleneck_dim=bottleneck_dim,
            intermediate_layers=intermediate_layers,
            pretrained=pretrained
        )
        
        # Identify embed_dim
        if hasattr(self.backbone, 'embed_dim'):
            embed_dim = self.backbone.embed_dim
        elif hasattr(self.backbone.blocks[0].mlp, 'fc1'):
            embed_dim = self.backbone.blocks[0].mlp.fc1.in_features
        elif hasattr(self.backbone.blocks[0].norm1, 'normalized_shape'):
            embed_dim = self.backbone.blocks[0].norm1.normalized_shape[0]
        else:
            embed_dim = 384
            
        # Initialize decodificador VisualFPN (FPN parameters are fully trainable)
        self.fpn = VisualFPN(embed_dim=embed_dim, fpn_dim=256, out_dim=out_dim)
        
    def forward(
        self,
        images: torch.Tensor,
        coords_2d: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # images shape: [B, 3, H, W]
        # coords_2d shape: [B, N, 2] (optional)
        B, C, H, W = images.shape
        
        # 1. Extraction of intermediate layers from DINOv2
        # outputs is a list of features of shape [B, num_patches, embed_dim]
        # get_intermediate_layers already discards the cls_token and register tokens
        outputs = self.backbone.get_intermediate_layers(
            images,
            n=self.intermediate_layers,
            reshape=False,
            norm=True
        )
        
        H_p, W_p = H // 14, W // 14
        features_list = []
        for feat in outputs:
            # Reorder to [B, embed_dim, H_p, W_p]
            feat_spatial = feat.reshape(B, H_p, W_p, -1).permute(0, 3, 1, 2).contiguous()
            features_list.append(feat_spatial)
            
        # 2. Decode features with FPN to retrieve dense pixel-level descriptors
        dense_descriptors = self.fpn(features_list, (H, W))
        
        # 3. Bilinear Sampling
        sampled_descriptors = None
        if coords_2d is not None:
            # Normalize pixel coordinates to [-1.0, 1.0] if necessary
            # coords_2d is [B, N, 2] where coords are (x, y)
            if coords_2d.abs().max() > 1.5:
                x_coords = coords_2d[..., 0]
                y_coords = coords_2d[..., 1]
                x_norm = 2.0 * x_coords / max(W - 1, 1) - 1.0
                y_norm = 2.0 * y_coords / max(H - 1, 1) - 1.0
                coords_norm = torch.stack([x_norm, y_norm], dim=-1)
            else:
                coords_norm = coords_2d
                
            # Perform grid sampling
            # grid shape must be [B, N, 1, 2]
            grid = coords_norm.unsqueeze(2)
            sampled = F.grid_sample(
                dense_descriptors,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            ) # Output shape: [B, out_dim, N, 1]
            
            # Rearrange to [B, N, out_dim]
            sampled_descriptors = sampled.squeeze(-1).permute(0, 2, 1).contiguous()
            
            # Apply L2 normalization along channel dimension
            sampled_descriptors = F.normalize(sampled_descriptors, p=2.0, dim=-1)
            
        return dense_descriptors, sampled_descriptors
