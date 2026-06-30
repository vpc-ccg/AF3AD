import csv
import json
import logging
import math
import os
import random
import re
import time
from datetime import datetime
from math import cos, pi
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast

from integrations.po3ad.data.factory import get_dataset_class
from integrations.po3ad.utils.losses import OffsetAnomalyLoss




def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range using min-max normalization.
    Args:
        scores: Array of scores to normalize.
    Returns:
        Normalized scores array. Returns zeros if all scores are identical.
    """
    if scores.max() > scores.min():
        return (scores - scores.min()) / (scores.max() - scores.min())
    return np.zeros_like(scores)


# Default fallback value for invalid metrics when computing rewards.
# Using 0.5 represents neutral uncertainty (50% detection rate) when
# metrics cannot be computed due to insufficient class diversity.
_INVALID_METRIC_FALLBACK = 0.5






def _compute_per_action_metrics(
    raw_rollout: Dict[str, Any],
    pred_offset: torch.Tensor,
    gt_offset: torch.Tensor,
    num_actions: int,
    weights: List[float],
) -> torch.Tensor:
    """Compute metrics-based rewards for each action in the rollout batch.
    This function calculates AUC-ROC and AUC-PR metrics at both point and object
    levels for each pseudo anomaly synthesized with the corresponding action.
    These metrics are then combined into a single reward value per action.
    Args:
        raw_rollout: Dictionary containing rollout batch data with 'split_sizes'
            and 'action_map' keys.
        pred_offset: Predicted offsets from discriminator, shape [N_all, 3].
        gt_offset: Ground truth offsets, shape [N_all, 3].
        num_actions: Total number of actions sampled.
        weights: List of 4 weights for combining metrics:
            [object_auc_roc, object_auc_pr, point_auc_roc, point_auc_pr].
    Returns:
        Tensor of shape (num_actions,) containing combined metric rewards.
    """
    device_pt = pred_offset.device
    pred_cpu = pred_offset.detach().cpu().abs().sum(dim=-1).numpy()
    gt_cpu = gt_offset.detach().cpu().abs().sum(dim=-1).numpy()
    point_gt_labels = (gt_cpu > 1e-6).astype(np.int32)

    split_sizes = raw_rollout.get('split_sizes')
    action_map = raw_rollout.get('action_map')

    if split_sizes is None or action_map is None:
        # Fallback: assume equal split per action
        N_all = pred_cpu.shape[0]
        if num_actions == 0 or N_all % num_actions != 0:
            # Return zeros if we can't properly split
            return torch.zeros(num_actions, device=device_pt, dtype=torch.float32)
        N_per_action = N_all // num_actions
        split_sizes_list = [N_per_action] * num_actions
        action_map_list = list(range(num_actions))
    else:
        if isinstance(split_sizes, torch.Tensor):
            split_sizes_list = [int(x) for x in split_sizes.detach().cpu().tolist()]
        else:
            split_sizes_list = [int(x) for x in split_sizes]
        if isinstance(action_map, torch.Tensor):
            action_map_list = [int(x) for x in action_map.detach().cpu().tolist()]
        else:
            action_map_list = [int(x) for x in action_map]
    # Collect metrics per group (each group is a point cloud)
    group_metrics: Dict[int, List[Dict[str, float]]] = {i: [] for i in range(num_actions)}

    start = 0
    for group_idx, size in enumerate(split_sizes_list):
        end = start + size
        end = min(end, pred_cpu.shape[0])
        if end <= start:
            continue

        # Get the action index for this group
        action_idx = action_map_list[group_idx] if group_idx < len(action_map_list) else group_idx

        # Point-level data for this group
        group_pred_scores = pred_cpu[start:end]
        group_gt_labels = point_gt_labels[start:end]

        # Object-level score (mean of point scores)
        object_score = float(group_pred_scores.mean())
        # Object-level label (1 if any point is anomalous, 0 otherwise)
        object_label = int(group_gt_labels.max())

        # Normalize point scores for this group
        group_pred_scores_norm = _normalize_scores(group_pred_scores)
        
        # Compute metrics for this group
        metrics = {
            'object_auc_roc': float('nan'),
            'object_auc_pr': float('nan'),
            'point_auc_roc': float('nan'),
            'point_auc_pr': float('nan'),
            'object_score': object_score,
            'object_label': object_label,
            'point_scores': group_pred_scores_norm,
            'point_labels': group_gt_labels,
        }

        # Point-level metrics can be computed if we have both classes
        if len(np.unique(group_gt_labels)) > 1:
            metrics['point_auc_roc'] = _safe_roc_auc(group_gt_labels, group_pred_scores_norm)
            metrics['point_auc_pr'] = _safe_average_precision(group_gt_labels, group_pred_scores_norm)

        group_metrics[action_idx].append(metrics)
        start = end
    
    # Aggregate metrics per action
    rewards_per_action = torch.zeros(num_actions, device=device_pt, dtype=torch.float32)

    for action_idx in range(num_actions):
        groups = group_metrics[action_idx]
        if not groups:
            continue

        # Collect all point-level and object-level data for this action
        all_object_scores: List[float] = []
        all_object_labels: List[int] = []
        all_point_scores: List[np.ndarray] = []
        all_point_labels: List[np.ndarray] = []

        for g in groups:
            all_object_scores.append(g['object_score'])
            all_object_labels.append(g['object_label'])
            all_point_scores.append(g['point_scores'])
            all_point_labels.append(g['point_labels'])

        # Compute aggregated metrics for this action
        action_metrics = {
            'object_auc_roc': float('nan'),
            'object_auc_pr': float('nan'),
            'point_auc_roc': float('nan'),
            'point_auc_pr': float('nan'),
        }
        
        # Object-level metrics: need at least 2 samples with different labels to compute AUC
        if len(all_object_scores) >= 2 and len(set(all_object_labels)) >= 2:
            obj_scores_arr = np.asarray(all_object_scores, dtype=np.float64)
            obj_labels_arr = np.asarray(all_object_labels, dtype=np.int32)
            action_metrics['object_auc_roc'] = _safe_roc_auc(obj_labels_arr, obj_scores_arr)
            action_metrics['object_auc_pr'] = _safe_average_precision(obj_labels_arr, obj_scores_arr)

        # Point-level metrics
        if all_point_scores and all_point_labels:
            point_scores_arr = np.concatenate(all_point_scores, axis=0)
            point_labels_arr = np.concatenate(all_point_labels, axis=0).astype(np.int32)
            if point_scores_arr.size > 0 and len(np.unique(point_labels_arr)) > 1:
                # Re-normalize after concatenation
                point_scores_norm = _normalize_scores(point_scores_arr)
                action_metrics['point_auc_roc'] = _safe_roc_auc(point_labels_arr, point_scores_norm)
                action_metrics['point_auc_pr'] = _safe_average_precision(point_labels_arr, point_scores_norm)
        
        # Combine metrics into reward using weights
        # Higher metrics = discriminator detects anomalies correctly
        # We want to reward actions that create anomalies the discriminator struggles with
        # So we use (1 - metric) to reward lower detection performance
        # Order matches weights parameter: [object_auc_roc, object_auc_pr, point_auc_roc, point_auc_pr]
        # When metrics are invalid/NaN, use fallback representing 50% random detection rate
        fallback = 1.0 - _INVALID_METRIC_FALLBACK
        reward_components = [
            (1.0 - action_metrics['object_auc_roc']) if np.isfinite(action_metrics['object_auc_roc']) else fallback,
            (1.0 - action_metrics['object_auc_pr']) if np.isfinite(action_metrics['object_auc_pr']) else fallback,
            (1.0 - action_metrics['point_auc_roc']) if np.isfinite(action_metrics['point_auc_roc']) else fallback,
            (1.0 - action_metrics['point_auc_pr']) if np.isfinite(action_metrics['point_auc_pr']) else fallback,
        ]
        
        # Weighted combination of metrics
        combined_reward = sum(w * r for w, r in zip(weights, reward_components))
        rewards_per_action[action_idx] = combined_reward

    return rewards_per_action







def _safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or scores.size == 0 or np.all(labels == labels[0]):
        return float("nan")
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


def _safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or scores.size == 0 or np.sum(labels) == 0:
        return float("nan")
    try:
        return float(average_precision_score(labels, scores))
    except ValueError:
        return float("nan")


def _format_metric(value: float) -> str:
    return f"{value:.4f}" if value is not None and np.isfinite(value) else "nan"


def _accumulate_metric_data(
    raw_rollout: Dict[str, Any],
    pred_offset: torch.Tensor,
    gt_offset: torch.Tensor,
    object_scores: List[float],
    object_labels: List[int],
    point_scores: List[np.ndarray],
    point_labels: List[np.ndarray],
    points_collected: int,
    max_points: Optional[int],
) -> int:
    pred_cpu = pred_offset.detach().cpu().abs().sum(dim=-1).numpy()
    gt_cpu = gt_offset.detach().cpu().abs().sum(dim=-1).numpy()
    point_gt_labels = (gt_cpu > 1e-6).astype(np.int32)

    split_sizes = raw_rollout.get('split_sizes')
    if split_sizes is None:
        split_sizes_list: List[int] = [int(pred_cpu.shape[0])]
    else:
        if isinstance(split_sizes, torch.Tensor):
            split_sizes_list = [int(x) for x in split_sizes.detach().cpu().tolist()]
        else:
            split_sizes_list = [int(x) for x in split_sizes]

    start = 0
    for size in split_sizes_list:
        end = start + size
        end = min(end, pred_cpu.shape[0])
        if end <= start:
            continue
        object_scores.append(float(pred_cpu[start:end].mean()))
        object_labels.append(int(point_gt_labels[start:end].max()))
        start = end

    if max_points is not None and points_collected >= max_points:
        return points_collected

    preds_local = pred_cpu
    labels_local = point_gt_labels

    if max_points is not None:
        remaining = max_points - points_collected
        if remaining <= 0:
            return points_collected
        sample_size = min(remaining, preds_local.shape[0])
        if sample_size < preds_local.shape[0]:
            idx = np.random.choice(preds_local.shape[0], sample_size, replace=False)
            preds_local = preds_local[idx]
            labels_local = labels_local[idx]
        points_collected += sample_size
    else:
        points_collected += preds_local.shape[0]

    point_scores.append(preds_local)
    point_labels.append(labels_local)
    return points_collected


def _compute_epoch_metrics(
    object_scores: List[float],
    object_labels: List[int],
    point_scores: List[np.ndarray],
    point_labels: List[np.ndarray],
) -> Dict[str, float]:
    
    # initialize metrics with NaN
    metrics = {
        'object_auc_roc': float('nan'),
        'object_auc_pr': float('nan'),
        'point_auc_roc': float('nan'),
        'point_auc_pr': float('nan'),
    }

    # compute object-level metrics
    if object_scores and object_labels:
        obj_scores_arr = np.asarray(object_scores, dtype=np.float64)
        obj_labels_arr = np.asarray(object_labels, dtype=np.int32)
        metrics['object_auc_roc'] = _safe_roc_auc(obj_labels_arr, obj_scores_arr)
        metrics['object_auc_pr'] = _safe_average_precision(obj_labels_arr, obj_scores_arr)

    # compute point-level metrics
    if point_scores and point_labels:
        point_scores_arr = np.concatenate(point_scores, axis=0)
        point_labels_arr = np.concatenate(point_labels, axis=0).astype(np.int32)
        if point_scores_arr.size > 0:
            point_scores_norm = _normalize_scores(point_scores_arr)
            metrics['point_auc_roc'] = _safe_roc_auc(point_labels_arr, point_scores_norm)
            metrics['point_auc_pr'] = _safe_average_precision(point_labels_arr, point_scores_norm)

    return metrics


def _eval_batch(batch, model):
    xyz_voxel = batch['xyz_voxel']
    feat_voxel = batch['feat_voxel']
    v2p_index = batch['v2p_index']

    with torch.no_grad():
        pred_offset = model.test_inference(feat_voxel, xyz_voxel, v2p_index)

    sample_score = torch.mean(torch.sum(torch.abs(pred_offset.detach().cpu()), dim=-1))
    return sample_score, pred_offset


def _evaluate_validation_set(
    args,
    discriminator: nn.Module,
    dataset,
    val_loader,
):
    if val_loader is None:
        return None

    prev_mode = discriminator.training
    discriminator.eval()

    normal_tag = getattr(dataset, 'normal_tag', None)
    delimiter = getattr(dataset, 'gt_delimiter', ',')

    object_scores: List[float] = []
    object_labels: List[int] = []
    point_scores: List[np.ndarray] = []
    point_labels: List[np.ndarray] = []

    for batch in val_loader:
        torch.cuda.empty_cache()
        sample_name = Path(batch['fn'][0]).stem

        if normal_tag and normal_tag in sample_name:
            gt_mask = np.zeros(batch['xyz_original'].shape[0])
        else:
            gt_path = dataset._resolve_gt_path(sample_name)
            kwargs = {}
            if delimiter is not None:
                kwargs['delimiter'] = delimiter
            gt_mask = np.loadtxt(gt_path, **kwargs)[:, 3:].squeeze(1)

        score, pred_offset = _eval_batch(batch, discriminator)
        pred_mask = pred_offset.detach().cpu().abs().sum(dim=-1).numpy()

        object_scores.append(score.item())
        object_labels.append(int(batch['labels'].detach().cpu().numpy()[0]))
        point_scores.append(pred_mask)
        point_labels.append(gt_mask)

    metrics = _compute_epoch_metrics(
        object_scores,
        object_labels,
        point_scores,
        point_labels,
    )

    if prev_mode:
        discriminator.train()

    return metrics





def fix_seed(seed):
    """Fixing seeds for consistency.

    Args:
        seed (_type_): _description_
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    # PyTorch and CUDA RNGs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Seed all GPUs
    
    # Enable deterministic behavior
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Enable deterministic algorithms (requires env var for some ops)
    # Set CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8
    # Use setdefault to avoid overriding user-specified configurations
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    # try:
    #     torch.use_deterministic_algorithms(True)
    # except (RuntimeError, AttributeError):
    #     # Fallback for older PyTorch versions that don't support this function
    #     pass
    
    

def _create_optimizer(params, optimizer_name, lr, betas, momentum, weight_decay):
    """
    Helper function to create an optimizer based on the name.

    Args:
        params: Model parameters to optimize.
        optimizer_name: Name of the optimizer.
        lr: Learning rate.
        betas: Tuple of (beta1, beta2) for Adam-like optimizers.
        momentum: Momentum for SGD.
        weight_decay: Weight decay for regularization.

    Returns:
        torch.optim.Optimizer: The created optimizer.

    Raises:
        ValueError: If the optimizer name is not recognized.
    """
    if optimizer_name == 'Adam':
        return optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'Adagrad':
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'Adadelta':
        return optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'RAdam':
        return optim.RAdam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif optimizer_name == 'NAdam':
        return optim.NAdam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif optimizer_name == 'Adamax':
        return optim.Adamax(params, lr=lr, betas=betas, weight_decay=weight_decay)
    elif optimizer_name == 'ASGD':
        return optim.ASGD(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. "
                         f"Supported optimizers: Adam, SGD, AdamW, RMSprop, Adagrad, Adadelta, RAdam, NAdam, Adamax, ASGD")



def create_optimizers(args, discriminator):
    """
    Creates optimizers for generator and discriminator models.

    Args:
        args: Namespace or dict containing optimizer parameters.
        discriminator: Discriminator model.

    Returns:
        opt_D: Discriminator optimizer.
    """


    disc_params = filter(lambda p: p.requires_grad, discriminator.parameters())
    discriminator_betas = (args.discriminator_beta1, args.discriminator_beta2)
    opt_D = _create_optimizer(
        disc_params,
        args.optimizer,
        args.lr_D,
        discriminator_betas,
        args.momentum,
        args.weight_decay
    )

    return  opt_D


def resume_from_checkpoint(
    args, run_id, logger, device,
    discriminator, opt_D, scaler_D
):
    """
    Loads checkpoint and restores model, optimizer, and scaler states.

    Returns:
        start_epoch (int), best_d_loss (float), best_epoch (int)
    """
    resume_ckpt = getattr(args, 'resume_checkpoint', '')
    start_epoch, best_d_loss, best_epoch = 0, float('inf'), 0

    if resume_ckpt:
        ckpt_path = Path(resume_ckpt)
        if not ckpt_path.is_file():
            logger.warning(
                "[%s] Resume checkpoint %s not found; starting from scratch.",
                run_id,
                ckpt_path,
            )
        else:
            logger.info("[%s] Loading checkpoint from %s", run_id, ckpt_path)
            ckpt = torch.load(str(ckpt_path), map_location=device)
            discriminator.load_state_dict(ckpt['discriminator'])
            
            opt_D.load_state_dict(ckpt['opt_D'])
            
            if 'scaler_D' in ckpt and ckpt['scaler_D']:
                scaler_D.load_state_dict(ckpt['scaler_D'])

            start_epoch = int(ckpt.get('epoch', 0))
            metrics_state = ckpt.get('metrics', {}) or {}
            best_d_loss = float(metrics_state.get('best_d_loss', metrics_state.get('d_loss', float('inf'))))
            best_epoch = int(metrics_state.get('best_epoch', start_epoch))
            logger.info(
                "[%s] Resumed training from epoch %d (best d_loss=%.4f at epoch %d)",
                run_id,
                start_epoch,
                best_d_loss,
                best_epoch,
            )
    return start_epoch, best_d_loss, best_epoch


def log_action_stats(actions, num_presets, writer, epoch):
    """
    Logs action frequency statistics and histogram to TensorBoard.

    Args:
        actions (torch.Tensor): Tensor of action indices.
        num_presets (int): Number of possible presets/actions.
        writer (SummaryWriter): TensorBoard writer instance.
        epoch (int): Current epoch or step.
    """
    with torch.no_grad():
        actions_long = actions.to(torch.long)
        counts = torch.bincount(actions_long, minlength=num_presets)
        num_sampled_actions = max(actions_long.numel(), 1)
        freqs = counts.float() / num_sampled_actions
        for preset_idx in range(num_presets):
            writer.add_scalar(
                f"acts/preset_{preset_idx}_freq",
                freqs[preset_idx].item(),
                epoch,
            )
        writer.add_histogram(
            "dist/preset_indices",
            actions_long.to(torch.float32),
            epoch,
        )


def save_anomaly_samples_wrapper(
    args,
    epoch,
    it,
    saved_samples_this_epoch,
    data_samples,
    samples_root,
    logger,
    save_fn,
):
    """
    Conditionally saves anomaly samples and logs the event.

    Args:
        args: Argument namespace with sample_export_freq, sample_export_max,
              sample_export_all, and sample_export_annotated.
        epoch (int): Current epoch.
        it (int): Current iteration.
        saved_samples_this_epoch (bool): Flag if samples were saved this epoch.
        data_samples: Data to save.
        samples_root: Output directory.
        logger: Logger instance.
        save_fn: Function to save samples.

    Returns:
        bool: Updated saved_samples_this_epoch flag.
    """
    if (
        args.sample_export_freq > 0
        and (epoch) % args.sample_export_freq == 0
        and not saved_samples_this_epoch
    ):
        saved_paths = save_fn(
            data_samples,
            epoch_index=epoch + 1,
            iter=it + 1,
            output_root=samples_root,
            max_samples=args.sample_export_max,
            logger=logger,
            export_all=getattr(args, 'sample_export_all', False),
            export_annotated=getattr(args, 'sample_export_annotated', False),
        )
        if saved_paths:
            saved_samples_this_epoch = True
            logger.info(
                "Saved %d anomaly synthesis samples for epoch %d to %s",
                len(saved_paths),
                epoch + 1,
                samples_root / f"epoch_{epoch + 1:04d}",
            )
    return saved_samples_this_epoch


# Epoch counts from 0 to N-1
def cosine_lr_after_step(optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6):
    if epoch < step_epoch:
        lr = base_lr
    else:
        lr = clip + 0.5 * (base_lr - clip) * (1 + cos(pi *
                                                      ((epoch - step_epoch) / (total_epochs - step_epoch))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def _sanitize_stub(name: str) -> str:
    """Create a filesystem-friendly name from the provided string."""
    if not name:
        return "sample"
    # Replace separators and unsupported characters with underscores.
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name)


def _write_point_cloud_ply(path: Path, points: np.ndarray) -> None:
    """Write an ASCII PLY file containing only XYZ coordinates."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be of shape (N, 3)")

    path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        if points.shape[0] > 0:
            np.savetxt(f, points, fmt="%.6f %.6f %.6f")


def _flatten_anomaly_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert a nested anomaly cfg dictionary into flat CSV-friendly keys."""

    flat: Dict[str, Any] = {}
    if not cfg:
        return flat

    for key, value in cfg.items():
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                value = value.item()
            else:
                value = value.detach().cpu().tolist()

        if isinstance(value, (list, tuple)):
            simple_elements: List[Any] = []
            for element in value:
                if isinstance(element, np.generic):
                    element = element.item()
                if isinstance(element, torch.Tensor):
                    element = element.item() if element.numel(
                    ) == 1 else element.detach().cpu().tolist()
                simple_elements.append(element)

            if all(isinstance(elem, (int, float, bool, type(None))) for elem in simple_elements):
                for idx, elem in enumerate(simple_elements):
                    flat[f"{key}_{idx}"] = elem
            else:
                flat[key] = json.dumps(simple_elements)
        elif isinstance(value, dict):
            flat[key] = json.dumps(value)
        else:
            flat[key] = value

    return flat


def _write_point_cloud_ply_with_colors(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write an ASCII PLY file containing XYZ coordinates and RGB colors.
    
    Args:
        path: Output file path.
        points: Array of shape (N, 3) containing XYZ coordinates.
        colors: Array of shape (N, 3) containing RGB colors in [0, 255] range.
    """
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be of shape (N, 3)")
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError("colors must be of shape (N, 3)")
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have the same number of rows")

    path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        if points.shape[0] > 0:
            # Combine points and colors into a single array for vectorized writing
            combined = np.column_stack([points, colors])
            np.savetxt(f, combined, fmt="%.6f %.6f %.6f %d %d %d")


def _compute_anomaly_colors(anomaly_scores: np.ndarray) -> np.ndarray:
    """
    Blue (normal) -> Yellow (medium anomaly) -> Red (high anomaly)
    """
    scores = anomaly_scores.astype(np.float32)
    max_score = scores.max()

    if max_score > 1e-8:
        v = scores / max_score
    else:
        v = np.zeros_like(scores)

    N = len(v)
    rgb = np.zeros((N, 3), dtype=np.float32)

    # Low anomaly: blue -> yellow (0.0 ~ 0.5)
    mask_low = v < 0.5
    t = v[mask_low] / 0.5
    rgb[mask_low, 0] = t            # R: 0 -> 1
    rgb[mask_low, 1] = t            # G: 0 -> 1
    rgb[mask_low, 2] = 1.0 - t      # B: 1 -> 0

    # High anomaly: yellow -> red (0.5 ~ 1.0)
    mask_high = v >= 0.5
    t = (v[mask_high] - 0.5) / 0.5
    rgb[mask_high, 0] = 1.0          # R stays 1
    rgb[mask_high, 1] = 1.0 - t      # G: 1 -> 0
    rgb[mask_high, 2] = 0.0          # B stays 0

    return (rgb * 255).astype(np.uint8)



def save_anomaly_samples(
    raw_rollout: Dict[str, Any],
    epoch_index: int,
    iter: int,
    output_root: Path,
    max_samples: int = 5,
    logger: Optional[logging.Logger] = None,
    export_all: bool = False,
    export_annotated: bool = False,
) -> List[Path]:
    """Persist synthesized anomaly samples to PLY files.

    Args:
        raw_rollout: Dictionary containing rollout batch data.
        epoch_index: Current epoch index for naming the output directory.
        iter: Current iteration index for naming output files.
        output_root: Root directory for saving samples.
        max_samples: Maximum number of samples to export (ignored if export_all=True).
        logger: Optional logger for logging messages.
        export_all: If True, export all samples instead of limiting to max_samples.
        export_annotated: If True, export samples with colored anomaly annotations.

    Returns:
        List of paths to the files that were written.
    """

    if "xyz_shifted" not in raw_rollout:
        return []

    xyz_shifted = raw_rollout["xyz_shifted"].detach().cpu()
    if xyz_shifted.ndim != 2 or xyz_shifted.size(1) != 3:
        if logger:
            logger.warning("Skipping sample export: unexpected xyz_shifted shape %s", tuple(
                xyz_shifted.shape))
        return []

    split_sizes = raw_rollout.get("split_sizes")
    if split_sizes is None:
        if logger:
            logger.warning(
                "Skipping sample export: missing split_sizes in rollout batch")
        return []

    if isinstance(split_sizes, torch.Tensor):
        split_sizes_list = split_sizes.detach().cpu().tolist()
    elif isinstance(split_sizes, np.ndarray):
        split_sizes_list = split_sizes.tolist()
    else:
        split_sizes_list = list(split_sizes)

    try:
        segments_shifted = torch.split(
            xyz_shifted, tuple(int(s) for s in split_sizes_list))
    except RuntimeError as exc:
        if logger:
            logger.warning("Skipping sample export: %s", exc)
        return []

    xyz_original = raw_rollout.get("xyz_original")
    segments_original: Optional[Tuple[torch.Tensor, ...]] = None
    if isinstance(xyz_original, torch.Tensor) and xyz_original.numel() == xyz_shifted.numel():
        segments_original = torch.split(
            xyz_original.detach().cpu(), tuple(int(s) for s in split_sizes_list))

    # Get anomaly offset for colored annotation
    batch_offset = raw_rollout.get("batch_offset")
    segments_offset: Optional[Tuple[torch.Tensor, ...]] = None
    if export_annotated and batch_offset is not None:
        if isinstance(batch_offset, torch.Tensor):
            batch_offset_cpu = batch_offset.detach().cpu()
            # Compute anomaly magnitude (L2 norm of offset vector)
            if batch_offset_cpu.ndim == 2 and batch_offset_cpu.size(1) == 3:
                try:
                    segments_offset = torch.split(
                        batch_offset_cpu, tuple(int(s) for s in split_sizes_list))
                except RuntimeError as exc:
                    if logger:
                        logger.warning("Could not split batch_offset for annotation: %s", exc)

    fn_list = raw_rollout.get("fn") or []
    action_map = raw_rollout.get("action_map")
    if isinstance(action_map, torch.Tensor):
        action_map_list = action_map.detach().cpu().tolist()
    elif isinstance(action_map, np.ndarray):
        action_map_list = action_map.tolist()
    elif isinstance(action_map, list):
        action_map_list = action_map
    else:
        action_map_list = []

    cfg_records_raw = raw_rollout.get("anomaly_cfg_params")
    if isinstance(cfg_records_raw, list):
        cfg_records = cfg_records_raw
    elif isinstance(cfg_records_raw, tuple):
        cfg_records = list(cfg_records_raw)
    else:
        cfg_records = []

    samples_dir = output_root / f"epoch_{epoch_index:04d}"
    samples_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    csv_rows: List[Dict[str, Any]] = []

    # Determine the number of samples to export
    num_segments = len(segments_shifted)
    num_to_export = num_segments if export_all else min(max_samples, num_segments)

    for idx, pts_shifted in enumerate(segments_shifted[:num_to_export]):
        pts_np = pts_shifted.numpy()
        base_name = fn_list[idx] if idx < len(fn_list) else f"sample_{idx:02d}"
        base_name = _sanitize_stub(base_name)
        if idx < len(action_map_list):
            base_name = f"{base_name}_act{int(action_map_list[idx])}"

        anomaly_path = samples_dir / \
            f"{base_name}_anomaly_iter_{str(iter)}.ply"
        
        # Export with or without colored annotation
        if export_annotated and segments_offset is not None and idx < len(segments_offset):
            # Compute anomaly magnitude (L2 norm of offset)
            offset_np = segments_offset[idx].numpy()
            anomaly_scores = np.linalg.norm(offset_np, axis=-1)
            colors = _compute_anomaly_colors(anomaly_scores)
            _write_point_cloud_ply_with_colors(anomaly_path, pts_np, colors)
        else:
            _write_point_cloud_ply(anomaly_path, pts_np)
        saved_paths.append(anomaly_path)

        row: Dict[str, Any] = {
            "sample_index": idx,
            "file_name": fn_list[idx] if idx < len(fn_list) else base_name,
            "saved_stub": base_name,
        }
        if idx < len(action_map_list):
            row["action_map"] = int(action_map_list[idx])

        if idx < len(cfg_records):
            cfg_entry = cfg_records[idx]
            if isinstance(cfg_entry, dict):
                row.update(_flatten_anomaly_cfg(cfg_entry))

        csv_rows.append(row)

        if segments_original is not None and idx < len(segments_original):
            orig_np = segments_original[idx].numpy()
            orig_path = samples_dir / f"{base_name}_original.ply"
            _write_point_cloud_ply(orig_path, orig_np)
            saved_paths.append(orig_path)

    if csv_rows:
        base_fields = ["sample_index", "file_name", "saved_stub", "action_map"]
        field_set = set()
        for row in csv_rows:
            field_set.update(row.keys())
        ordered_fields = [field for field in base_fields if field in field_set]
        for key in sorted(field_set - set(ordered_fields)):
            ordered_fields.append(key)

        csv_path = samples_dir / "anomaly_params.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=ordered_fields)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)

        if logger:
            logger.info(
                "Saved anomaly synthesizer parameters for %d samples to %s",
                len(csv_rows),
                csv_path,
            )

    return saved_paths


def calculate_loss(
    gt_offsets: torch.Tensor,
    pred_offset: torch.Tensor,
    loss_variant: str = "baseline",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    focal_tau: float = 0.01,
    lambda_aux_focal: float = 0.1,
    # Regularization parameters
    lambda_l1_reg: float = 0.0,
    lambda_l2_reg: float = 0.0,
    lambda_smooth_reg: float = 0.0,
    edge_aware_weight: float = 0.0,
    point_coords: Optional[torch.Tensor] = None,
    edge_scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict, Dict, Dict, torch.Tensor, torch.Tensor]:
    """Calculate offset prediction loss with optional regularization.
    
    Args:
        gt_offsets: Ground truth offset vectors [N, 3]
        pred_offset: Predicted offset vectors [N, 3]
        loss_variant: Loss variant ('baseline', 'focal_cls', 'focal_reg')
        focal_alpha: Alpha for focal loss
        focal_gamma: Gamma for focal loss
        focal_tau: Tau for focal regression
        lambda_aux_focal: Weight for auxiliary focal classification loss
        lambda_l1_reg: L1 regularization weight (sparsity)
        lambda_l2_reg: L2 regularization weight (smoothness)
        lambda_smooth_reg: Smoothness regularization weight (spatial consistency)
        edge_aware_weight: Edge-aware weighting factor (downweights edge points)
        point_coords: Point coordinates for smoothness regularization [N, 3]
        edge_scores: Edge scores for edge-aware weighting [N]
        
    Returns:
        loss: Total loss
        pred: Empty dict (for compatibility)
        visual_dict: Visualization values
        meter_dict: Metric values with counts
        offset_norm_loss: Offset norm component
        offset_dir_loss: Offset direction component
    """
    variant = loss_variant.lower()
    if variant == "baseline":
        loss_fn = OffsetAnomalyLoss(
            lambda_l1_reg=lambda_l1_reg,
            lambda_l2_reg=lambda_l2_reg,
            lambda_smooth_reg=lambda_smooth_reg,
            edge_aware_weight=edge_aware_weight,
            eps=1e-8,
        )
    elif variant == "focal_cls":
        loss_fn = OffsetAnomalyLoss(
            use_aux_focal_classification=True,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            lambda_aux_focal=lambda_aux_focal,
            lambda_l1_reg=lambda_l1_reg,
            lambda_l2_reg=lambda_l2_reg,
            lambda_smooth_reg=lambda_smooth_reg,
            edge_aware_weight=edge_aware_weight,
            eps=1e-8,
        )
    elif variant == "focal_reg":
        loss_fn = OffsetAnomalyLoss(
            use_focal_regression=True,
            focal_gamma=focal_gamma,
            focal_tau=focal_tau,
            lambda_l1_reg=lambda_l1_reg,
            lambda_l2_reg=lambda_l2_reg,
            lambda_smooth_reg=lambda_smooth_reg,
            edge_aware_weight=edge_aware_weight,
            eps=1e-8,
        )
    else:
        raise ValueError(
            "loss_variant must be one of {'baseline', 'focal_cls', 'focal_reg'}"
        )

    loss, loss_dict = loss_fn(gt_offsets, pred_offset, point_coords=point_coords, edge_scores=edge_scores)
    offset_norm_loss = loss_dict["offset_norm"]
    offset_dir_loss = loss_dict["offset_dir"]

    with torch.no_grad():
        pred: Dict[str, Any] = {}
        visual_dict: Dict[str, float] = {}
        meter_dict: Dict[str, Tuple[float, int]] = {}

        visual_dict['loss'] = float(loss.detach().cpu())
        meter_dict['loss'] = (visual_dict['loss'], pred_offset.shape[0])
        visual_dict['offset_norm'] = float(offset_norm_loss.detach().cpu())
        meter_dict['offset_norm'] = (
            visual_dict['offset_norm'],
            pred_offset.shape[0],
        )
        visual_dict['offset_dir'] = float(offset_dir_loss.detach().cpu())
        meter_dict['offset_dir'] = (
            visual_dict['offset_dir'],
            pred_offset.shape[0],
        )

        if variant == "focal_cls":
            focal_loss_val = float(loss_dict["aux_focal_cls"].detach().cpu())
            visual_dict['focal_cls'] = focal_loss_val
            meter_dict['focal_cls'] = (focal_loss_val, pred_offset.shape[0])
        elif variant == "focal_reg":
            focal_reg_val = float(loss_dict["focal_reg"].detach().cpu())
            visual_dict['focal_reg'] = focal_reg_val
            meter_dict['focal_reg'] = (focal_reg_val, pred_offset.shape[0])

        # Log regularization losses if enabled
        if lambda_l1_reg > 0:
            l1_val = float(loss_dict["l1_reg"].detach().cpu())
            visual_dict['l1_reg'] = l1_val
            meter_dict['l1_reg'] = (l1_val, pred_offset.shape[0])
        if lambda_l2_reg > 0:
            l2_val = float(loss_dict["l2_reg"].detach().cpu())
            visual_dict['l2_reg'] = l2_val
            meter_dict['l2_reg'] = (l2_val, pred_offset.shape[0])
        if lambda_smooth_reg > 0 and point_coords is not None:
            smooth_val = float(loss_dict["smooth_reg"].detach().cpu())
            visual_dict['smooth_reg'] = smooth_val
            meter_dict['smooth_reg'] = (smooth_val, pred_offset.shape[0])

    return loss, pred, visual_dict, meter_dict, offset_norm_loss, offset_dir_loss


def prepare_data(args):
    """Prepare dataset and data loaders.

    Args:
        args (_type_): _description_

    Returns:
        _type_: _description_
    """
    DatasetCls = get_dataset_class(args.dataset)
    dataset = DatasetCls(args)
    dataset.trainLoader()
    val_loader = None
    if getattr(args, 'validation_eval_freq', 0) > 0:
        dataset.valLoader()
        val_loader = getattr(dataset, 'val_data_loader', None)
    return dataset, dataset.train_data_loader, dataset.train_random_data_loader, dataset.train_rollout_data_loader, val_loader


def build_run_id(args) -> str:
    """Prepare a unique run ID based on timestamp and experiment name.

    Args:
        args (_type_): _description_

    Returns:
        str: _description_
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp = getattr(args, "exp_name", None)
    return f"{ts}_{exp}" if exp else ts


def prepare_dirs(args, run_id: str):
    """Prepare logging and checkpoint directories.

    Args:
        args (_type_): _description_
        run_id (str): _description_

    Returns:
        _type_: _description_
    """
    base = Path(args.logpath)
    run_dir = base / run_id
    logs_dir = run_dir / "logs"
    ckpt_dir = run_dir / "ckpts"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, logs_dir, ckpt_dir


def attach_file_handler(logger: logging.Logger, logfile: Path):
    """Attach a file handler to the provided logger.

    Args:
        logger (logging.Logger): _description_
        logfile (Path): _description_
    """
    fh = logging.FileHandler(str(logfile))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)


def save_hparams(run_dir: Path, args):
    """Save hyperparameters to JSON.

    Args:
        run_dir (Path): _description_
        args (_type_): _description_
    """
    hp_path = run_dir / "hparams.json"
    try:
        with open(hp_path, "w") as f:
            json.dump(vars(args), f, indent=2, default=str)
    except Exception as e:
        print(f"Could not save hparams: {e}")


def save_checkpoint(
    ckpt_dir: Path,
    run_id: str,
    tag: str,
    discriminator,
    opt_D,
    epoch: int,
    scaler_D: GradScaler,
    metrics: Optional[Dict[str, Any]] = None,
):
    if tag == "final":
        filename = ckpt_dir / f"{run_id}_final.pth"
    elif tag == "best":
        filename = ckpt_dir / f"{run_id}_best.pth"
    else:
        filename = ckpt_dir / f"{run_id}_{tag}_e{epoch:04d}.pth"
    payload = {
        "epoch": epoch,
        "discriminator": discriminator.state_dict(),
        "opt_D": opt_D.state_dict(),
        "scaler_D": scaler_D.state_dict(),
        "run_id": run_id,
        "tag": tag,
        "time": time.time(),
    }

    if metrics is not None:
        payload["metrics"] = metrics

    torch.save(payload, str(filename))




# def fps(data, number):
#     '''
#         data B N 3
#         number int
#     '''
#     fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
#     fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
#     return fps_data


# Adapted from https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
class CosineAnnealingWarmupRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        first_cycle_steps,
        cycle_mult=1.0,
        max_lr=0.1,
        min_lr=0.001,
        warmup_steps=0,
        gamma=1.0,
        last_epoch=-1,
    ):
        if first_cycle_steps <= 0:
            raise ValueError("first_cycle_steps must be positive")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if warmup_steps >= first_cycle_steps:
            raise ValueError("warmup_steps must be less than first_cycle_steps")

        self.first_cycle_steps = int(first_cycle_steps)
        self.cycle_mult = float(cycle_mult)
        self.base_max_lr = float(max_lr)
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = int(warmup_steps)
        self.gamma = float(gamma)
        self.cur_cycle_steps = int(first_cycle_steps)
        self.cycle = 0
        self.step_in_cycle = last_epoch

        for param_group in optimizer.param_groups:
            param_group["lr"] = self.min_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return [self.min_lr for _ in self.base_lrs]
        if self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr
                for base_lr in self.base_lrs
            ]
        return [
            base_lr
            + (self.max_lr - base_lr)
            * (1 + cos(pi * (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps)))
            / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                    self.cur_cycle_steps = self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult**n - 1) / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = int(self.first_cycle_steps * self.cycle_mult**n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


def cosine_lr_with_warmup(
    optimizer,
    base_lr,
    epoch,
    total_epochs,
    warmup_epochs=0,
    min_lr=1e-6,
):
    if total_epochs <= 0:
        return base_lr

    warmup_epochs = max(0, int(warmup_epochs))
    if warmup_epochs > 0 and epoch < warmup_epochs:
        lr = base_lr * float(epoch + 1) / float(warmup_epochs)
    elif total_epochs <= warmup_epochs:
        lr = base_lr
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def get_optimizer_lr(optimizer):
    for param_group in optimizer.param_groups:
        return float(param_group.get("lr", 0.0))
    return 0.0

def _evaluate_training_pseudo_anomalies(
    args,
    discriminator: nn.Module,
    rollout_loader,
    dataset,
    num_batches: int = 5,
) -> Optional[Dict[str, float]]:
    """Evaluate model on pseudo anomalous training samples to check overfitting.
    
    This function generates a batch of pseudo anomalous samples (similar to those used 
    during training) and evaluates the model's performance on them. This helps determine
    if the model is learning/overfitting on the training distribution.
    
    Args:
        args: Training arguments
        discriminator: The discriminator model to evaluate
        rollout_loader: DataLoader for generating pseudo anomalous samples
        dataset: Dataset object with rollout_param_queue
        num_batches: Number of batches to evaluate on
        
    Returns:
        Dictionary containing metrics with keys:
            - 'object_auc_roc': Object-level ROC AUC score (float)
            - 'object_auc_pr': Object-level Average Precision (float)
            - 'point_auc_roc': Point-level ROC AUC score (float)
            - 'point_auc_pr': Point-level Average Precision (float)
        Returns None if evaluation fails.
    """
    if rollout_loader is None:
        return None
    
    # Set model to eval mode but remember previous mode
    prev_mode = discriminator.training
    discriminator.eval()
    
    # Prepare metric accumulators
    object_scores: List[float] = []
    object_labels: List[int] = []
    point_scores: List[np.ndarray] = []
    point_labels: List[np.ndarray] = []
    
    # Create an iterator for the rollout loader
    rollout_iter = iter(rollout_loader)
    
    try:
        amp_enabled = torch.cuda.is_available()
        
        for batch_idx in range(num_batches):
            # Generate a batch of pseudo anomalous samples
            # Use uniform sampling or default parameters for unbiased evaluation
            params = {
                'force_single_sample': getattr(args, 'rollout_force_single_sample', True),
                'rollout_voxel_scale': getattr(args, 'rollout_voxel_scale', 1.0),
                'max_points_per_group': getattr(args, 'rollout_max_points_per_group', 250000),
            }
            dataset.rollout_param_queue.put(params)
            
            try:
                batch = next(rollout_iter)
            except StopIteration:
                # Reinitialize iterator if we run out
                rollout_iter = iter(rollout_loader)
                batch = next(rollout_iter)
            
            # Evaluate with no gradient computation
            # Note: AMP is used if CUDA is available to match training behavior
            with torch.no_grad(), autocast(enabled=amp_enabled):
                pred_offset = discriminator(batch)
                gt_offset = batch['batch_offset'].to(pred_offset.device, non_blocking=True)
            
            # Accumulate metrics data
            # Note: points_collected starts at 0 for each evaluation run since we want
            # to collect all points from the evaluation batches (max_points=None).
            # This is intentional as training evaluation should use fresh samples each time.
            _ = _accumulate_metric_data(
                batch,
                pred_offset,
                gt_offset,
                object_scores,
                object_labels,
                point_scores,
                point_labels,
                points_collected=0,
                max_points=None,  # Collect all points for training evaluation
            )
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Error during training evaluation: {e}")
        if prev_mode:
            discriminator.train()
        return None
    
    # Compute metrics
    metrics = _compute_epoch_metrics(
        object_scores,
        object_labels,
        point_scores,
        point_labels,
    )
    
    # Restore previous mode
    if prev_mode:
        discriminator.train()
    
    return metrics
