"""Loss utilities for offset-based anomaly detection."""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class OffsetAnomalyLoss(nn.Module):
    """Offset prediction loss with optional focal-style weighting and regularization.

    Supports baseline PO3AD-style regression, focal regression weighting, an
    auxiliary anomaly classification focal loss based on predicted offset norms,
    and various regularization schemes to reduce edge sensitivity in partial scans.
    
    Regularization options:
        - L1 regularization: Penalizes large predicted offsets (sparsity)
        - L2 regularization: Penalizes squared predicted offsets (smoothness)
        - Smoothness regularization: Penalizes differences between neighboring point offsets
        - Edge-aware weighting: Downweights loss for points near boundaries/edges
    """

    def __init__(
        self,
        use_focal_regression: bool = False,
        use_aux_focal_classification: bool = False,
        focal_gamma: float = 2.0,
        focal_tau: float = 0.01,
        focal_alpha: float = 0.25,
        lambda_aux_focal: float = 0.1,
        # Regularization parameters
        lambda_l1_reg: float = 0.0,
        lambda_l2_reg: float = 0.0,
        lambda_smooth_reg: float = 0.0,
        edge_aware_weight: float = 0.0,
        smoothness_max_sample_points: int = 4096,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if use_focal_regression and use_aux_focal_classification:
            raise ValueError(
                "Enable either focal regression or auxiliary focal classification, not both."
            )
        self.use_focal_regression = use_focal_regression
        self.use_aux_focal_classification = use_aux_focal_classification
        self.focal_gamma = focal_gamma
        self.focal_tau = focal_tau
        self.focal_alpha = focal_alpha
        self.lambda_aux_focal = lambda_aux_focal
        # Regularization
        self.lambda_l1_reg = lambda_l1_reg
        self.lambda_l2_reg = lambda_l2_reg
        self.lambda_smooth_reg = lambda_smooth_reg
        self.edge_aware_weight = edge_aware_weight
        self.smoothness_max_sample_points = smoothness_max_sample_points
        self.eps = eps

    @staticmethod
    def _binary_focal_loss_with_logits(
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Binary focal loss on logits with optional alpha-balancing."""
        targets = targets.type_as(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probas = torch.sigmoid(logits)
        p_t = targets * probas + (1 - targets) * (1 - probas)
        focal = (1 - p_t) ** gamma * bce

        if alpha >= 0:
            alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
            focal = alpha_t * focal

        if reduction == "mean":
            return focal.mean()
        if reduction == "sum":
            return focal.sum()
        return focal

    def forward(
        self,
        gt_offsets: torch.Tensor,
        pred_offset: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        edge_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute offset regression loss with optional focal terms and regularization.

        Args:
            gt_offsets: Ground-truth offsets of shape [N, 3].
            pred_offset: Predicted offsets of shape [N, 3].
            point_coords: Optional point coordinates [N, 3] for smoothness regularization.
            edge_scores: Optional edge scores [N] for edge-aware weighting (higher = more edge-like).

        Returns:
            total_loss: Aggregated scalar loss tensor.
            loss_dict: Dictionary of individual components.
        """
        eps = self.eps
        device = pred_offset.device
        dtype = pred_offset.dtype

        pt_diff = pred_offset - gt_offsets
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)

        gt_norm = torch.norm(gt_offsets, p=2, dim=1)
        gt_unit = gt_offsets / (gt_norm.unsqueeze(-1) + eps)
        pred_norm = torch.norm(pred_offset, p=2, dim=1)
        pred_unit = pred_offset / (pred_norm.unsqueeze(-1) + eps)
        direction_diff = -(gt_unit * pred_unit).sum(-1)

        # Apply edge-aware weighting if enabled and edge_scores provided
        sample_weights = torch.ones(pt_dist.shape[0], device=device, dtype=dtype)
        if self.edge_aware_weight > 0 and edge_scores is not None:
            # Downweight edge points: weight = 1 - edge_aware_weight * normalized_edge_score
            max_edge_score = edge_scores.max()
            if max_edge_score > eps:
                edge_scores_norm = edge_scores / max_edge_score
                sample_weights = 1.0 - self.edge_aware_weight * edge_scores_norm
                sample_weights = sample_weights.clamp(min=0.1)  # Minimum weight of 0.1
            # If max_edge_score is near zero, all points are similar, so keep uniform weights

        if self.use_focal_regression:
            mag = gt_norm
            weight = (mag / (mag + self.focal_tau)).pow(self.focal_gamma)
            weight = weight.detach()
            combined_weight = weight * sample_weights
            offset_norm_loss = (combined_weight * pt_dist).mean()
            offset_dir_loss = (combined_weight * direction_diff).mean()
            focal_reg_loss = offset_norm_loss + offset_dir_loss
            total_loss = focal_reg_loss
        else:
            weighted_pt_dist = sample_weights * pt_dist
            weighted_dir_diff = sample_weights * direction_diff
            offset_norm_loss = weighted_pt_dist.mean()
            offset_dir_loss = weighted_dir_diff.mean()
            focal_reg_loss = torch.zeros((), device=device, dtype=dtype)
            total_loss = offset_norm_loss + offset_dir_loss

        aux_focal_loss = torch.zeros_like(total_loss)
        if self.use_aux_focal_classification:
            anomaly_labels = (gt_norm > 1e-6).float()
            pred_logits = torch.log(pred_norm + eps)
            aux_focal_loss = self._binary_focal_loss_with_logits(
                pred_logits,
                anomaly_labels,
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                reduction="mean",
            )
            total_loss = total_loss + self.lambda_aux_focal * aux_focal_loss

        # L1 regularization on predicted offsets (encourages sparsity)
        l1_reg_loss = torch.zeros((), device=device, dtype=dtype)
        if self.lambda_l1_reg > 0:
            l1_reg_loss = torch.mean(torch.abs(pred_offset))
            total_loss = total_loss + self.lambda_l1_reg * l1_reg_loss

        # L2 regularization on predicted offsets (encourages smaller predictions)
        l2_reg_loss = torch.zeros((), device=device, dtype=dtype)
        if self.lambda_l2_reg > 0:
            l2_reg_loss = torch.mean(pred_offset ** 2)
            total_loss = total_loss + self.lambda_l2_reg * l2_reg_loss

        # Smoothness regularization (encourages spatial consistency)
        smooth_reg_loss = torch.zeros((), device=device, dtype=dtype)
        if self.lambda_smooth_reg > 0 and point_coords is not None:
            smooth_reg_loss = self._compute_smoothness_loss(pred_offset, point_coords)
            total_loss = total_loss + self.lambda_smooth_reg * smooth_reg_loss

        loss_dict: Dict[str, torch.Tensor] = {
            "total": total_loss,
            "offset_norm": offset_norm_loss,
            "offset_dir": offset_dir_loss,
            "focal_reg": focal_reg_loss,
            "aux_focal_cls": aux_focal_loss,
            "l1_reg": l1_reg_loss,
            "l2_reg": l2_reg_loss,
            "smooth_reg": smooth_reg_loss,
        }
        return total_loss, loss_dict

    def _compute_smoothness_loss(
        self,
        pred_offset: torch.Tensor,
        point_coords: torch.Tensor,
        k_neighbors: int = 8,
    ) -> torch.Tensor:
        """Compute smoothness regularization loss.
        
        Encourages neighboring points to have similar predicted offsets,
        which reduces sensitivity to isolated edge/boundary points.
        
        Args:
            pred_offset: Predicted offsets [N, 3]
            point_coords: Point coordinates [N, 3]
            k_neighbors: Number of neighbors to consider
            
        Returns:
            Scalar smoothness loss
        """
        N = point_coords.shape[0]
        device = point_coords.device
        
        # For efficiency, sample a subset of points for smoothness computation
        max_sample_points = min(N, self.smoothness_max_sample_points)
        if N > max_sample_points:
            indices = torch.randperm(N, device=device)[:max_sample_points]
            point_coords = point_coords[indices]
            pred_offset = pred_offset[indices]
            N = max_sample_points
        
        # Compute pairwise distances (efficient batch computation)
        # Using torch.cdist for batch distance computation
        dists = torch.cdist(point_coords, point_coords, p=2)  # [N, N]
        
        # Get k nearest neighbors (excluding self)
        k = min(k_neighbors, N - 1)
        if k < 1:
            return torch.zeros((), device=device, dtype=pred_offset.dtype)
        
        # Get indices of k nearest neighbors (exclude self by using topk on modified distances)
        dists_no_self = dists.clone()
        dists_no_self.fill_diagonal_(float('inf'))
        _, neighbor_idx = torch.topk(dists_no_self, k, dim=1, largest=False)  # [N, k]
        
        # Gather neighbor offsets
        neighbor_offsets = pred_offset[neighbor_idx]  # [N, k, 3]
        
        # Compute offset differences with neighbors
        offset_diff = pred_offset.unsqueeze(1) - neighbor_offsets  # [N, k, 3]
        smoothness_loss = torch.mean(torch.abs(offset_diff))
        
        return smoothness_loss