import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from .Mink import Mink_unet as unet3d
import numpy as np
import open3d as o3d
import os


# =============================================================================
# OFFSET PREDICTION MODULE VARIANTS
# =============================================================================
# This module provides multiple architecture variants for the offset prediction
# head used in PO3AD. The variants are designed for ablation studies to evaluate
# different architectural choices:
#
# 1. 'baseline' - Original simple MLP (3 layers: out_ch -> 16 -> 3)
# 2. 'deep' - Deeper MLP with more capacity (5 layers with gradual reduction)
# 3. 'residual' - Deep MLP with residual/skip connections for better gradients
# 4. 'multi_head' - Multi-head architecture predicting x,y,z independently
# 5. 'attention' - Attention-based feature weighting before prediction
#
# =============================================================================


class ResidualBlock(nn.Module):
    """Residual block with pre-activation (BatchNorm -> PReLU -> Linear).
    
    This block implements the pre-activation residual learning pattern which
    has been shown to improve gradient flow in deep networks.
    """
    def __init__(self, dim, dropout=0.0):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.PReLU(),
            nn.Linear(dim, dim, bias=False),
            nn.BatchNorm1d(dim),
            nn.PReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim, dim, bias=False),
        )
    
    def forward(self, x):
        return x + self.block(x)


class ChannelAttention(nn.Module):
    """Channel attention module for feature weighting.
    
    Learns to weight different feature channels based on their importance
    for offset prediction, similar to SE-Net attention mechanism.
    """
    def __init__(self, dim, reduction=4):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        weights = self.attention(x)
        return x * weights


class MultiHeadOffsetPredictor(nn.Module):
    """Multi-head offset predictor with separate heads for each coordinate.
    
    Instead of predicting all 3 coordinates from a single MLP, this module
    uses separate prediction heads for x, y, and z offsets. This allows
    each head to specialize in predicting one coordinate direction.
    """
    def __init__(self, in_dim, hidden_dim=32, num_layers=2, dropout=0.0):
        super(MultiHeadOffsetPredictor, self).__init__()
        
        # Shared feature transformation
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
        )
        
        # Separate heads for x, y, z
        self.head_x = self._make_head(hidden_dim, num_layers, dropout)
        self.head_y = self._make_head(hidden_dim, num_layers, dropout)
        self.head_z = self._make_head(hidden_dim, num_layers, dropout)
    
    def _make_head(self, hidden_dim, num_layers, dropout):
        # Ensure at least 1 layer (the final output layer)
        num_layers = max(1, num_layers)
        layers = []
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.PReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ])
        layers.append(nn.Linear(hidden_dim, 1, bias=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        shared_feat = self.shared(x)
        offset_x = self.head_x(shared_feat)
        offset_y = self.head_y(shared_feat)
        offset_z = self.head_z(shared_feat)
        return torch.cat([offset_x, offset_y, offset_z], dim=-1)


def create_offset_head(variant, out_channels, hidden_dim=64, num_layers=3, 
                       dropout=0.0, attention_reduction=4):
    """Factory function to create offset prediction head based on variant type.
    
    Args:
        variant (str): Architecture variant name. Options:
            - 'baseline': Original 3-layer MLP (out_ch -> 16 -> 3)
            - 'deep': Deeper MLP with gradual dimension reduction
            - 'residual': Deep MLP with residual connections
            - 'multi_head': Separate prediction heads for x, y, z
            - 'attention': Attention-weighted features before prediction
        out_channels (int): Output dimension from backbone (input to head)
        hidden_dim (int): Hidden layer dimension for deep/residual variants
        num_layers (int): Number of layers for deep/residual variants
        dropout (float): Dropout probability for regularization
        attention_reduction (int): Reduction ratio for attention module
    
    Returns:
        nn.Module: The offset prediction head module
    """
    if variant == 'baseline':
        # Original architecture: simple 3-layer MLP
        return nn.Sequential(
            nn.Linear(out_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Linear(out_channels, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.Linear(16, 3, bias=True)
        )
    
    elif variant == 'deep':
        # Deeper MLP with more capacity and gradual dimension reduction
        layers = [
            nn.Linear(out_channels, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
        ]
        current_dim = hidden_dim
        for i in range(num_layers - 2):
            next_dim = max(16, current_dim // 2)
            layers.extend([
                nn.Linear(current_dim, next_dim, bias=False),
                nn.BatchNorm1d(next_dim),
                nn.PReLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ])
            current_dim = next_dim
        layers.append(nn.Linear(current_dim, 3, bias=True))
        return nn.Sequential(*layers)
    
    elif variant == 'residual':
        # Deep MLP with residual connections for better gradient flow
        layers = [
            nn.Linear(out_channels, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
        ]
        # Add residual blocks
        num_res_blocks = max(1, (num_layers - 2) // 2)
        for _ in range(num_res_blocks):
            layers.append(ResidualBlock(hidden_dim, dropout))
        # Final projection
        layers.extend([
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.PReLU(),
            nn.Linear(16, 3, bias=True),
        ])
        return nn.Sequential(*layers)
    
    elif variant == 'multi_head':
        # Multi-head architecture with separate heads for x, y, z
        return MultiHeadOffsetPredictor(
            in_dim=out_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    
    elif variant == 'attention':
        # Attention-weighted features before prediction
        return nn.Sequential(
            ChannelAttention(out_channels, attention_reduction),
            nn.Linear(out_channels, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, bias=False),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.PReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim // 2, 3, bias=True),
        )
    
    else:
        raise ValueError(f"Unknown offset head variant: {variant}. "
                        f"Options: baseline, deep, residual, multi_head, attention")


class PONet(nn.Module):
    """Point Offset Network for 3D anomaly detection.
    
    The network predicts per-point offset vectors that represent the
    displacement from anomalous positions back to normal surface positions.
    
    Args:
        in_channels (int): Input feature dimension (typically 3 for xyz normals)
        out_channels (int): Backbone output feature dimension
        offset_head_variant (str): Architecture variant for offset prediction head
        offset_hidden_dim (int): Hidden dimension for offset head
        offset_num_layers (int): Number of layers in offset head
        offset_dropout (float): Dropout probability in offset head
        offset_attention_reduction (int): Attention reduction ratio
    """
    def __init__(self, in_channels, out_channels, 
                 offset_head_variant='baseline',
                 offset_hidden_dim=64,
                 offset_num_layers=3,
                 offset_dropout=0.0,
                 offset_attention_reduction=4):
        super(PONet, self).__init__()
        
        self.offset_head_variant = offset_head_variant
        
        self.backbone = unet3d(in_channels=in_channels,
                               out_channels=out_channels, arch='MinkUNet34C')
        
        self.linear_offset = create_offset_head(
            variant=offset_head_variant,
            out_channels=out_channels,
            hidden_dim=offset_hidden_dim,
            num_layers=offset_num_layers,
            dropout=offset_dropout,
            attention_reduction=offset_attention_reduction,
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self,  feat_voxel, xyz_voxel, v2p_v1):
        cuda_cur_device = os.environ['CUDA_VISIBLE_DEVICES']
        inputs = ME.SparseTensor(
            feat_voxel, xyz_voxel, device='cuda:{}'.format(cuda_cur_device))
        voxel_feat = self.backbone(inputs)
        point_feat = voxel_feat.F[v2p_v1]
        pred_offset = self.linear_offset(point_feat)

        return pred_offset

    def test_inference(self, feat_voxel, xyz_voxel, v2p_v1):
        cuda_cur_device = os.environ['CUDA_VISIBLE_DEVICES']
        inputs = ME.SparseTensor(
            feat_voxel, xyz_voxel, device='cuda:{}'.format(cuda_cur_device))
        voxel_feat = self.backbone(inputs)
        point_feat = voxel_feat.F[v2p_v1]
        pred_offset = self.linear_offset(point_feat)

        return pred_offset


def eval_fn(batch, model):
    xyz_voxel = batch['xyz_voxel']
    feat_voxel = batch['feat_voxel']
    v2p_index = batch['v2p_index']

    # So, the model get voxel features and positions as well as the point to voxel index to calculate offset for all points
    with torch.no_grad():
        pred_offset = model.test_inference(feat_voxel, xyz_voxel, v2p_index)

    # The anomaly score is the mean of sum of xyz absolute offset
    sample_score = torch.mean(
        torch.sum(torch.abs(pred_offset.detach().cpu()), dim=-1))

    return sample_score, pred_offset