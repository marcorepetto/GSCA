import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv
from torch_geometric.utils import scatter

class GeometricFeatureExtractor(nn.Module):
    """
    Geometric Feature Extractor for 3D point clouds using Dynamic EdgeConv (DGCNN) layers.
    It extracts local geometric features at multiple scales and projects them to a common latent space.
    """
    def __init__(self, k: int = 20, out_channels: int = 256):
        super().__init__()
        
        self.k = k
        self.out_channels = out_channels
        
        # 1. MLP 1 (maps from 6 to 64)
        # PointNet-like edge feature representation mapping from x_i, (x_j - x_i) = 6 channels to 64 channels
        self.mlp1 = nn.Sequential(
            nn.Linear(6, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        # EdgeConv 1: Micro-scale local geometric feature extraction
        self.conv1 = DynamicEdgeConv(nn=self.mlp1, k=k, aggr="max")
        
        # 2. MLP 2 (maps from 128 to 128)
        # Dynamic edge feature representation mapping from x1_i, (x1_j - x1_i) = 128 channels to 128 channels
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        # EdgeConv 2: Mid-scale dynamic feature extraction
        self.conv2 = DynamicEdgeConv(nn=self.mlp2, k=k, aggr="max")
        
        # 3. MLP 3 (maps from 256 to 256)
        # Dynamic edge feature representation mapping from x2_i, (x2_j - x2_i) = 256 channels to 256 channels
        self.mlp3 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        # EdgeConv 3: Macro-scale dynamic feature extraction
        self.conv3 = DynamicEdgeConv(nn=self.mlp3, k=k, aggr="max")
        
        # 4. Red de Proyección Lineal (Global Projection Network)
        # Maps the concatenated multi-scale feature representations (64 + 128 + 256 = 448) to the target dimension
        self.projection = nn.Sequential(
            nn.Linear(448, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, out_channels)
        )
        
    def forward(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # Input Validation (Task T-GEOM-03 hard requirements)
        if not isinstance(pos, torch.Tensor):
            raise ValueError("pos must be a torch.Tensor")
        if not isinstance(batch, torch.Tensor):
            raise ValueError("batch must be a torch.Tensor")
            
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"pos must have shape [N, 3] and ndim == 2, got shape {list(pos.shape)}")
            
        if pos.shape[0] != batch.shape[0]:
            raise ValueError(f"pos and batch must have the same number of points, got {pos.shape[0]} and {batch.shape[0]}")
            
        # Center the coordinates per batch element to guarantee translation invariance
        # Compute the centroid (mean coordinates) for each point cloud in the batch
        centroids = scatter(pos, batch, dim=0, reduce="mean")  # [Num_Clouds_in_Batch, 3]
        pos_centered = pos - centroids[batch]  # [N, 3]
        
        # 1. Micro-scale convolution (using translation-invariant centered coordinates)
        x1 = self.conv1(pos_centered, batch)  # Output: [N, 64]
        
        # 2. Mid-scale convolution
        x2 = self.conv2(x1, batch)  # Output: [N, 128]
        
        # 3. Macro-scale convolution
        x3 = self.conv3(x2, batch)  # Output: [N, 256]
        
        # Multi-scale Fusion (Task T-GEOM-04 hard requirements)
        x_concat = torch.cat([x1, x2, x3], dim=-1)  # Output: [N, 448]
        
        # Linear Projection
        x_proj = self.projection(x_concat)  # Output: [N, out_channels]
        
        # L2 Metric Normalization
        feat_3d = F.normalize(x_proj, p=2, dim=-1)  # Output: [N, out_channels]
        
        return feat_3d
