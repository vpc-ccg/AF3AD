import torch
import torch.nn as nn
import MinkowskiEngine as ME
from .PO3AD.Mink import Mink_unet as unet3d
from .PO3AD.PO3AD import create_offset_head
import numpy as np
import open3d as o3d
import os


class Discriminator(nn.Module):
    """Discriminator network for 3D anomaly detection.
    
    Uses a sparse convolutional U-Net backbone followed by a configurable
    offset prediction head that predicts per-point displacement vectors.
    
    Args:
        args: Configuration object with the following attributes:
            - in_channels (int): Input feature dimension
            - out_channels (int): Backbone output feature dimension
            - offset_head_variant (str): Architecture variant for offset head
              Options: 'baseline', 'deep', 'residual', 'multi_head', 'attention'
            - offset_hidden_dim (int): Hidden dimension for offset head
            - offset_num_layers (int): Number of layers in offset head
            - offset_dropout (float): Dropout probability in offset head
            - offset_attention_reduction (int): Attention reduction ratio
    """
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.backbone = unet3d(in_channels=args.in_channels,
                               out_channels=args.out_channels, arch='MinkUNet34C')
        
        # Get offset head configuration from args with defaults
        offset_head_variant = getattr(args, 'offset_head_variant', 'baseline')
        offset_hidden_dim = getattr(args, 'offset_hidden_dim', 64)
        offset_num_layers = getattr(args, 'offset_num_layers', 3)
        offset_dropout = getattr(args, 'offset_dropout', 0.0)
        offset_attention_reduction = getattr(args, 'offset_attention_reduction', 4)
        
        self.linear_offset = create_offset_head(
            variant=offset_head_variant,
            out_channels=args.out_channels,
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

    def forward(self,  x):
        
        xyz_voxel = x['xyz_voxel']
        feat_voxel = x['feat_voxel']
        v2p_index = x['v2p_index']
        cuda_cur_device = os.environ['CUDA_VISIBLE_DEVICES']
        inputs = ME.SparseTensor(
            feat_voxel, xyz_voxel, device='cuda:{}'.format(cuda_cur_device))
        voxel_feat = self.backbone(inputs)
        point_feat = voxel_feat.F[v2p_index]
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