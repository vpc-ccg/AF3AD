import multiprocessing as mp
import math
import glob
import torch
import random
import numpy as np
import open3d as o3d
import scipy.ndimage
import scipy.interpolate
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
import integrations.po3ad.data.AnomalyShapeNet.transform as aug_transform
import os
import re
import gc
import sys
from pathlib import Path
from collections import OrderedDict

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Optional, Tuple

from af3ad import (
    AnomalyPreset,
    PseudoAnomalySynthesizer,
    RandomFactory,
    SmartAnomaly_Cfg,
    normalize_synthesis_policy,
)
from integrations.po3ad.utils.visualize import save_pc_plotly_html
DEBUG = False

# Index of the micro dimple field preset in the AnomalyPreset.presets list.
_MICRO_DIMPLE_PRESET_IDX = 9
_RANDOMFACTORY_POLICIES = {
    "randomfactory_raw",
    "randomfactory_filtered",
    "preset_randomfactory_mix",
}
_AF3AD_POLICY_ALIASES = {
    "plain": "plain",
    "default": "plain",
    "random_factory": "random_factory",
    "randomfactory": "random_factory",
}


def _normalize_af3ad_policy(policy: str) -> str:
    normalized = str(policy).strip().lower().replace("-", "_")
    try:
        return _AF3AD_POLICY_ALIASES[normalized]
    except KeyError as exc:
        valid = "', '".join(sorted(_AF3AD_POLICY_ALIASES))
        raise ValueError(
            f"Unsupported AF3AD policy '{policy}'. Expected one of '{valid}'."
        ) from exc


# ----------------------
# Queue Manager for Dataloader Workers
# ----------------------

# To inject params into dataloader workers
manager = mp.Manager()
standard_param_queue = manager.Queue()
random_param_queue = manager.Queue()
rollout_param_queue = manager.Queue()
# Backwards compatibility: retain the original name for the standard queue
param_queue = standard_param_queue
# Contribution: Writing dataloader collate function as a closure to capture shared config that can be modified externally through training loop, without any signi


class BoundedCache:
    """Simple LRU cache with maximum size to prevent unbounded memory growth."""

    def __init__(self, maxsize=100):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self._total_sets = 0  # Track total set operations
        self._evictions = 0   # Track eviction count

    def get(self, key):
        """Retrieve value from cache, moving it to end (most recently used)."""
        if key not in self.cache:
            return None
        # Move to end to mark as recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key, value):
        """Store value in cache, evicting oldest if at capacity."""
        self._total_sets += 1
        if key in self.cache:
            # Update existing, move to end
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            # Evict oldest BEFORE adding new entry if at capacity
            if len(self.cache) >= self.maxsize:
                evicted_key, evicted_value = self.cache.popitem(last=False)
                self._evictions += 1
                # Help Python GC by explicitly deleting evicted data
                del evicted_value
            self.cache[key] = value

    def clear(self):
        """Clear all cached data to free memory."""
        self.cache.clear()
        # Force garbage collection after clearing cache
        gc.collect()

    def size(self):
        """Return the number of items currently cached."""
        return len(self.cache)

    def get_stats(self):
        """Return cache statistics for monitoring."""
        return {
            'size': len(self.cache),
            'maxsize': self.maxsize,
            'total_sets': self._total_sets,
            'evictions': self._evictions,
            'utilization': len(self.cache) / self.maxsize if self.maxsize > 0 else 0
        }

    def estimate_memory_bytes(self):
        """Estimate memory usage of cached data."""
        total_bytes = 0
        for key, value in self.cache.items():
            # Estimate key size
            total_bytes += sys.getsizeof(key)
            # Estimate value size (handle tuples of numpy arrays)
            if isinstance(value, tuple):
                for item in value:
                    if isinstance(item, np.ndarray):
                        total_bytes += item.nbytes
                    else:
                        total_bytes += sys.getsizeof(item)
            elif isinstance(value, np.ndarray):
                total_bytes += value.nbytes
            else:
                total_bytes += sys.getsizeof(value)
        return total_bytes

@dataclass
class CollateBundle:
    """Container for the different training collate entrypoints."""

    standard: Callable
    rollout: Callable
    random: Callable
    dispatch: Callable

    def __call__(self, batch_indices):
        return self.dispatch(batch_indices)


def _strength_preset_scales(preset_name: str) -> Tuple[float, float]:
    """Return radius and magnitude multipliers for coarse strength presets."""

    preset = str(preset_name).strip().lower()
    if preset == "small":
        return 0.85, 0.75
    if preset == "large":
        return 1.20, 1.35
    # "default" and "medium" intentionally keep the current distribution.
    return 1.0, 1.0


def _apply_strength_controls(cfg: SmartAnomaly_Cfg, args) -> SmartAnomaly_Cfg:
    """Apply lightweight radius and magnitude scaling to a sampled anomaly cfg."""

    radius_scale, magnitude_scale = _strength_preset_scales(
        getattr(args, "anomaly_strength_preset", "default")
    )
    radius_scale *= float(getattr(args, "anomaly_radius_multiplier", 1.0))
    magnitude_scale *= float(getattr(args, "anomaly_scale_multiplier", 1.0))

    if cfg.R is not None:
        cfg.R = float(cfg.R) * radius_scale
    cfg.B = float(cfg.B) * magnitude_scale
    return cfg


def _mean_knn_distance(points: np.ndarray, k: int = 16) -> np.ndarray:
    """Return per-point mean kNN spacing for a small point cloud."""

    points = np.asarray(points, dtype=np.float32)
    n_points = int(points.shape[0])
    if n_points <= 1:
        return np.zeros(n_points, dtype=np.float32)

    k_eff = min(int(k), n_points - 1)
    diff = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    nearest = np.partition(distances, kth=k_eff, axis=1)[:, 1:k_eff + 1]
    return nearest.mean(axis=1).astype(np.float32)


def _normalize_displaced_region_density(
    xyz: np.ndarray,
    new_xyz: np.ndarray,
    anomaly_mask: np.ndarray,
    *,
    k: int = 16,
    max_scale_delta: float = 0.25,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Rescale only displaced points so local patch spacing matches the source patch."""

    mask = np.asarray(anomaly_mask).astype(bool)
    stats = {
        "density_normalized": 0.0,
        "density_norm_affected_points": float(mask.sum()),
        "density_norm_spacing_before": 0.0,
        "density_norm_spacing_after_before": 0.0,
        "density_norm_spacing_after": 0.0,
        "density_norm_scale": 1.0,
    }
    if int(mask.sum()) <= 1:
        return new_xyz, stats

    before_patch = np.asarray(xyz[mask], dtype=np.float32)
    after_patch = np.asarray(new_xyz[mask], dtype=np.float32)
    spacing_before = float(_mean_knn_distance(before_patch, k=k).mean())
    spacing_after_before = float(_mean_knn_distance(after_patch, k=k).mean())
    stats["density_norm_spacing_before"] = spacing_before
    stats["density_norm_spacing_after_before"] = spacing_after_before

    eps = 1e-8
    if spacing_before <= eps or spacing_after_before <= eps:
        stats["density_norm_spacing_after"] = spacing_after_before
        return new_xyz, stats

    raw_scale = spacing_before / spacing_after_before
    min_scale = max(0.0, 1.0 - float(max_scale_delta))
    max_scale = 1.0 + float(max_scale_delta)
    scale = float(np.clip(raw_scale, min_scale, max_scale))

    normalized_xyz = new_xyz.copy()
    center = after_patch.mean(axis=0, keepdims=True)
    normalized_patch = center + scale * (after_patch - center)
    normalized_xyz[mask] = normalized_patch.astype(normalized_xyz.dtype, copy=False)

    spacing_after = float(_mean_knn_distance(normalized_xyz[mask], k=k).mean())
    stats.update({
        "density_normalized": 1.0,
        "density_norm_spacing_after": spacing_after,
        "density_norm_scale": scale,
    })
    return normalized_xyz, stats


def _compute_sample_anomaly_stats(
    xyz: np.ndarray,
    new_xyz: np.ndarray,
    gt_offset: np.ndarray,
    anomaly_mask: np.ndarray,
    args,
) -> Dict[str, float]:
    """Summarize pseudo-anomaly strength for one synthesized training sample."""

    eps = float(getattr(args, "anomaly_strength_eps", 1e-6))
    displacement = np.linalg.norm(gt_offset, axis=1)
    changed_mask = displacement > eps
    changed_points = int(changed_mask.sum())
    total_points = max(int(gt_offset.shape[0]), 1)
    changed_fraction = float(changed_points) / float(total_points)

    if changed_points > 0:
        changed_disp = displacement[changed_mask]
        mean_disp = float(changed_disp.mean())
        max_disp = float(changed_disp.max())
        changed_xyz = new_xyz[changed_mask]
        bbox_extent = np.maximum(
            changed_xyz.max(axis=0) - changed_xyz.min(axis=0),
            0.0,
        )
        bbox_volume = float(np.prod(bbox_extent))
    else:
        mean_disp = 0.0
        max_disp = 0.0
        bbox_volume = 0.0

    anomaly_l1 = float(np.abs(gt_offset).sum())
    anomaly_l2 = float(np.linalg.norm(gt_offset.reshape(-1)))
    strength_score = float(anomaly_l2 / np.sqrt(float(total_points)))
    anomaly_mask_size = int(np.asarray(anomaly_mask).sum())

    return {
        "changed_points": changed_points,
        "changed_fraction": changed_fraction,
        "mean_displacement": mean_disp,
        "max_displacement": max_disp,
        "anomaly_l1": anomaly_l1,
        "anomaly_l2": anomaly_l2,
        "anomaly_mask_size": anomaly_mask_size,
        "bbox_volume": bbox_volume,
        "strength_score": strength_score,
    }


def _sample_meets_strength_thresholds(stats: Dict[str, float], args) -> bool:
    """Return whether a pseudo anomaly satisfies the configured strength floors."""

    if int(getattr(args, "min_changed_points", 0)) > 0:
        if stats["changed_points"] < int(getattr(args, "min_changed_points", 0)):
            return False
    if float(getattr(args, "min_changed_fraction", 0.0)) > 0.0:
        if stats["changed_fraction"] < float(getattr(args, "min_changed_fraction", 0.0)):
            return False
    if float(getattr(args, "min_anomaly_l2", 0.0)) > 0.0:
        if stats["anomaly_l2"] < float(getattr(args, "min_anomaly_l2", 0.0)):
            return False
    return True


def make_collate(dataset_object, param_queue):
    import queue as _queue
    import numpy as np
    import torch
    import MinkowskiEngine as ME
    import open3d as o3d

    queue_timeout = 2

    def _get_params():
        """ Reads a dict from the queue

        Raises:
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        try:
            params = param_queue.get(timeout=queue_timeout)
        except _queue.Empty:
            params = {}
        if not isinstance(params, dict):
            raise TypeError(f"Expected parameter dict, got {type(params)!r}")
        return params

    # Seed for deterministic preset sampling; falls back to torch worker seed when
    # a manual seed is not provided in the config.
    fallback_seed = getattr(dataset_object.global_cfg, 'manual_seed', None)
    # Cache of per-worker RNGs so each DataLoader worker reuses its own generator.
    worker_rngs = {}

    def _get_fallback_rng():
        """Return a per-worker RNG seeded deterministically."""

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else None
        rng = worker_rngs.get(worker_id)
        if rng is None:
            # Use manual seed when set so experiments are reproducible; otherwise
            # mirror PyTorch's worker seed so the RNG stays stable per worker.
            if fallback_seed is not None:
                seed = int(fallback_seed)
            else:
                seed = int(torch.initial_seed())
            rng = np.random.default_rng(seed)
            # Persist the generator for this worker id to avoid reseeding mid-epoch.
            worker_rngs[worker_id] = rng
        return rng

    num_presets = max(int(getattr(dataset_object, 'num_presets', 0)), 1)
    # Get preset probabilities for weighted sampling
    # Falls back to uniform distribution if not configured on the dataset object
    preset_probs = getattr(dataset_object, 'preset_probs', None)
    if preset_probs is None:
        preset_probs = np.ones(num_presets, dtype=np.float64) / num_presets

    def _sample_random_actions(num_actions=1):
        """Sample random preset indices when none are provided, using weighted probabilities."""

        if num_presets <= 0:
            raise ValueError("Dataset has no anomaly presets configured.")
        rng = _get_fallback_rng()
        # Use weighted sampling based on preset_probs
        return rng.choice(num_presets, size=(num_actions,), p=preset_probs).astype(np.int64)

    def _normalize_rollout_actions(params):
        """Ensure rollout actions is a 1-D array of preset indices."""

        actions = params.get('actions') if params is not None else None
        if actions is None:
            num_actions = 1
            if isinstance(params, dict):
                for key in ('num_actions', 'rollout_K', 'K'):
                    maybe = params.get(key)
                    if maybe is not None:
                        try:
                            num_actions = max(int(maybe), 1)
                        except (TypeError, ValueError):
                            num_actions = 1
                        break
            actions = _sample_random_actions(num_actions)
        else:
            actions = np.asarray(actions)
            if actions.ndim == 0:
                actions = actions.reshape(1)
            elif actions.ndim == 2 and actions.shape[1] == 1:
                actions = actions.reshape(-1)
        if actions.ndim != 1:
            raise ValueError(
                f"`actions` must be a 1-D array of preset indices, got shape {actions.shape}")
        return actions.astype(np.int64, copy=False)

    def _select_best_action(params):
        """Select the preset index to apply for standard batches."""

        actions = params.get('actions')
        if actions is None:
            return _sample_random_actions(num_actions=1)

        actions = np.asarray(actions)
        if actions.ndim == 0:
            actions = actions.reshape(1)
        elif actions.ndim == 2 and actions.shape[1] == 1:
            actions = actions.reshape(-1)
        elif actions.ndim != 1:
            raise ValueError(
                f"`actions` must be 1-D when provided, got shape {actions.shape}")

        actions = actions.astype(np.int64, copy=False)
        rewards = params.get('rewards', None)
        if rewards is not None:
            rewards = np.asarray(rewards).reshape(-1)
            if rewards.shape[0] != actions.shape[0]:
                raise ValueError(
                    f"`rewards` length ({rewards.shape}) does not match actions ({actions.shape})")
            best_idx = int(np.argmax(rewards))
        else:
            best_idx = int(params.get('best_action_idx', 0))

        if not 0 <= best_idx < actions.shape[0]:
            raise ValueError(
                f"`best_action_idx` {best_idx} out of range for {actions.shape[0]} actions")

        return actions[best_idx:best_idx + 1]

    def _collate_impl(id_list, params, actions_arr, rollout_mode, N_fixed: int = 2048, generator: torch.Generator = None, random_mode: bool = False):
        actions_arr = np.asarray(actions_arr, dtype=np.int64)
        if actions_arr.ndim != 1:
            raise ValueError(
                f"Expected actions of shape (N,), got {actions_arr.shape}")
        batch_synthesis_policy = None
        if (
            random_mode
            and bool(
                getattr(
                    dataset_object.global_cfg,
                    'batch_policy_sampling',
                    False,
                )
            )
        ):
            batch_synthesis_policy = str(np.random.choice((
                'af3ad_presets',
                'randomfactory_raw',
                'randomfactory_filtered',
            )))
        batch_preset_idx = None
        if batch_synthesis_policy == 'af3ad_presets':
            if actions_arr.size == 0:
                raise ValueError(
                    "Cannot select a batch preset from an empty action array"
                )
            batch_preset_idx = int(actions_arr[0])
        # --- RNG for reproducibility (optional) ---
        if generator is None:
            generator = torch.Generator()
            # uncomment for deterministic sampling
            generator.manual_seed(
                int(getattr(dataset_object.global_cfg, 'manual_seed', None)))

        sample_indices = list(id_list)
        if rollout_mode:
            force_single_sample = bool(params.get('force_single_sample', True))
            if force_single_sample and len(sample_indices) > 1:
                sample_indices = [sample_indices[0]]
            rollout_voxel_scale = float(params.get('rollout_voxel_scale', 1.0))
            this_voxel_size = float(
                dataset_object.voxel_size) * rollout_voxel_scale
        else:
            this_voxel_size = float(dataset_object.voxel_size)

        max_points_per_group = params.get('max_points_per_group', None)
        if max_points_per_group is not None:
            max_points_per_group = int(max_points_per_group)

        file_name = []
        xyz_voxel = []
        feat_voxel = []
        xyz_original = []
        xyz_shifted = []
        normal_original = []
        reward_anomaly_masks = []
        v2p_index_batch = []
        total_voxel_num = 0
        batch_count = [0]
        total_point_num = 0
        gt_offset_list = []
        track_reward_data = bool(
            getattr(dataset_object.global_cfg, 'reward_tracking', False)
        )

        # --- keep per-item tensors for BN3 outputs for point cloud feature extraction ---
        xyz_original_per_item = []
        xyz_shifted_per_item = []
        offset_per_item = []

        # Track shift_indices for both rollout and standard modes
        shift_indices_list = []

        # Store patch point clouds for FPFH feature extraction
        patch_xyz_list = []

        split_sizes = []
        action_map = []
        anomaly_cfg_records = []
        preset_indices = []
        anomaly_changed_points = []
        anomaly_changed_fraction = []
        anomaly_mean_displacement = []
        anomaly_max_displacement = []
        anomaly_l1 = []
        anomaly_l2 = []
        anomaly_mask_size = []
        anomaly_bbox_volume = []
        anomaly_strength_score = []
        anomaly_below_threshold = []
        anomaly_regen_attempts = []
        anomaly_regen_exhausted = []
        diayn_valid_mask = []

        # This loop is on point clouds, so in each iteration we are working with one point cloud (In rollout mode, there will be only one point cloud with multiple actions)
        for sample_idx, idx in enumerate(sample_indices):
            # Set name variables
            fn_path = dataset_object.train_file_list[idx]
            base_name = dataset_object.train_file_list[idx]

            # Load point cloud and preprocess
            coord, vertex_normals = dataset_object._load_train_point_cloud(
                fn_path)
            # Initialize mask as -1 (no mask)
            mask = np.ones(coord.shape[0]) * -1

            # Apply any dataset-specific augmentations and compositions
            Point_dict = {
                'coord': coord,
                'normal': vertex_normals,
                'mask': mask
            }
            Point_dict, centers = dataset_object.train_aug_compose(Point_dict)

            # Apply edge segment cutout if enabled (removes edge segments like cutout in images)
            # This must be applied after SphereCropMask since it uses the segment centers
            edge_cutout_transform = getattr(
                dataset_object, 'RandomEdgeSegmentCutout', None)
            if edge_cutout_transform is not None:
                Point_dict, centers = edge_cutout_transform(
                    Point_dict, centers)

            # Extract base data arrays
            xyz_base = Point_dict['coord'].astype(np.float32)
            normal_base = Point_dict['normal'].astype(np.float32)
            mask = Point_dict['mask'].astype(np.int32)

            # Transform mask values to expected range by giving valid values to mask indicies which are out of bound
            mask[mask == (dataset_object.mask_num + 1)
                 ] = dataset_object.mask_num - 1

            if random_mode:
                action_count = 1
            else:
                action_count = actions_arr.shape[0]

            # For loop over actions (in rollout mode) or single action (standard mode)
            for action_idx in range(action_count):

                # Get index of preset to use. Batch policy sampling deliberately
                # reuses one preset for every sample in a preset batch.
                # TODO: if training in RL mode, when training the discriminator, the random_mode is disabled and best actions are selected
                action_idx = action_idx if not random_mode else sample_idx
                preset_idx = (
                    batch_preset_idx
                    if batch_preset_idx is not None
                    else int(actions_arr[action_idx])
                )
                if not 0 <= preset_idx < len(dataset_object.anomaly_presets):
                    raise ValueError(
                        f"Preset index {preset_idx} out of range (have {len(dataset_object.anomaly_presets)})"
                    )

                max_regen_attempts = max(
                    0, int(getattr(dataset_object.global_cfg, "max_regen_attempts", 0))
                )
                intact_ratio = float(getattr(dataset_object, 'intact_ratio', 0.0))
                weak_anomaly_fallback = str(
                    getattr(dataset_object.global_cfg, "weak_anomaly_fallback", "keep_last")
                ).strip().lower()

                final_xyz = None
                final_new_xyz = None
                final_gt_offset = None
                final_normals = None
                final_anomaly_mask = None
                final_cfg_dict = None
                final_shift_xyz_base = None
                final_shift_index = None
                final_stats = None
                final_below_threshold = False
                regen_attempts_used = 0
                regen_exhausted = False

                for attempt_idx in range(max_regen_attempts + 1):
                    regen_attempts_used = attempt_idx
                    anomlay_cfg = None
                    policy_metadata = None
                    synthesis_policy = (
                        batch_synthesis_policy
                        or getattr(
                            dataset_object,
                            "synthesis_policy",
                            "af3ad_presets",
                        )
                    )
                    preset_randomfactory_prob = float(
                        getattr(dataset_object, "preset_randomfactory_prob", 0.5)
                    )
                    mixed_policy_active = synthesis_policy == "preset_randomfactory_mix"
                    mixed_use_preset = (
                        mixed_policy_active
                        and np.random.rand() < preset_randomfactory_prob
                    )
                    randomfactory_v1_active = (
                        dataset_object.global_cfg.smart_anomaly
                        and synthesis_policy in _RANDOMFACTORY_POLICIES
                        and not mixed_use_preset
                    )
                    if (
                        (synthesis_policy == "af3ad_presets" or mixed_use_preset)
                        and dataset_object.af3ad_policy == "plain"
                    ):
                        anomlay_cfg = _apply_strength_controls(
                            SmartAnomaly_Cfg(**asdict(dataset_object.anomaly_presets[preset_idx]())),
                            dataset_object.global_cfg,
                        )

                    # Randomly select indices for shifting and extract corresponding
                    # data from xyz_base, normal_base, and centers.
                    actual_num_segments = len(centers)
                    mask_range = np.arange(0, actual_num_segments)
                    use_prefetched_shift = (
                        attempt_idx == 0
                        and dataset_object.use_prefetched_shift_indices
                        and dataset_object.prefetched_shift_indices is not None
                        and idx in dataset_object.prefetched_shift_indices
                        and getattr(dataset_object, 'RandomEdgeSegmentCutout', None) is None
                    )
                    if use_prefetched_shift:
                        shift_index = dataset_object.prefetched_shift_indices[idx]
                    else:
                        shift_index = np.random.choice(
                            mask_range,
                            (
                                1
                                if randomfactory_v1_active
                                else min(
                                    max(
                                        1,
                                        int(getattr(
                                            dataset_object.global_cfg,
                                            "pseudo_anomaly_count",
                                            1,
                                        )),
                                    ),
                                    actual_num_segments,
                                )
                            ),
                            replace=False,
                        )

                    shift_index = np.asarray(shift_index, dtype=np.int64).reshape(-1)
                    shift_index_list = [int(x) for x in shift_index]
                    mask_used = mask.copy()
                    mask_used[np.isin(mask_used, shift_index)] = -1
                    shift_xyz_base = xyz_base[mask_used == -1].copy()
                    shift_normal_base = normal_base[mask_used == -1].copy()
                    center = centers[shift_index[0]]

                    xyz = xyz_base.copy()
                    normals = normal_base.copy()
                    mlocal = mask_used
                    shift_xyz = shift_xyz_base
                    shift_normal = shift_normal_base

                    # Keep the first try compatible with the intact-ratio path.
                    keep_intact = attempt_idx == 0 and (np.random.rand() < intact_ratio)
                    random_factory_generated = False
                    if keep_intact:
                        shifted_xyz = shift_xyz.copy()
                    elif dataset_object.global_cfg.smart_anomaly:
                        if randomfactory_v1_active:
                            filtered = synthesis_policy == "randomfactory_filtered"
                            max_rf_attempts = (
                                dataset_object.random_factory.config.max_resample_attempts
                                if filtered
                                else 1
                            )
                            selected_new_xyz = None
                            selected_metadata = None
                            selected_anomaly_mask = None

                            for rf_attempt in range(1, max_rf_attempts + 1):
                                rf_cfg, sampled_metadata = dataset_object.random_factory.sample_config()
                                shifted_candidate = dataset_object.generate_pseudo_anomaly(
                                    shift_xyz,
                                    shift_normal,
                                    center,
                                    anomlay_cfg=rf_cfg,
                                )
                                candidate_new_xyz = xyz.copy()
                                candidate_new_xyz[mlocal == -1] = shifted_candidate
                                rf_stats = dataset_object.random_factory.compute_stats(
                                    xyz,
                                    candidate_new_xyz,
                                )
                                accepted, reject_reasons = dataset_object.random_factory.passes_filter(
                                    rf_stats
                                )
                                if not filtered:
                                    accepted = True
                                    reject_reasons = ()

                                fallback = bool(filtered and not accepted and rf_attempt == max_rf_attempts)
                                candidate_metadata = dataset_object.random_factory.build_metadata(
                                    synthesis_policy,
                                    sampled_metadata,
                                    rf_stats,
                                    attempt_count=rf_attempt,
                                    accepted=accepted,
                                    rejected_reasons=reject_reasons,
                                    fallback=fallback,
                                    center=center,
                                )
                                candidate_metadata["synthesis_policy"] = synthesis_policy
                                candidate_metadata["synthesis_source"] = "randomfactory"
                                if mixed_policy_active:
                                    candidate_metadata["preset_randomfactory_prob"] = preset_randomfactory_prob
                                candidate_metadata["selected_patch_index"] = int(shift_index[0])
                                candidate_metadata["selected_patch_indices"] = [
                                    int(x) for x in np.asarray(shift_index).reshape(-1)
                                ]
                                candidate_metadata["selected_patch_point_count"] = int(
                                    shift_xyz_base.shape[0]
                                )

                                if accepted or fallback:
                                    selected_new_xyz = candidate_new_xyz
                                    selected_metadata = candidate_metadata
                                    selected_anomaly_mask = (
                                        np.linalg.norm(candidate_new_xyz - xyz, axis=1)
                                        > dataset_object.random_factory.config.eps
                                    )
                                    break

                            if selected_new_xyz is None or selected_metadata is None:
                                raise RuntimeError("RandomFactory did not generate a candidate sample.")

                            new_xyz = selected_new_xyz
                            anomaly_mask = selected_anomaly_mask
                            policy_metadata = selected_metadata
                            random_factory_generated = True
                        else:
                            composed_new_xyz = xyz.copy()
                            pseudo_entries = []
                            selected_patch_point_counts = []
                            selected_preset_indices = []
                            selected_preset_names = []
                            selected_centers = []

                            for pseudo_idx, patch_idx in enumerate(shift_index_list):
                                patch_mask = mask == patch_idx
                                patch_xyz = composed_new_xyz[patch_mask].copy()
                                patch_normal = normal_base[patch_mask].copy()
                                patch_center = centers[patch_idx]
                                patch_center_arr = (
                                    np.asarray(patch_center, dtype=np.float32)
                                    .reshape(-1)[:3]
                                )
                                if batch_preset_idx is not None:
                                    this_preset_idx = batch_preset_idx
                                elif pseudo_idx == 0:
                                    this_preset_idx = preset_idx
                                else:
                                    this_preset_idx = int(
                                        np.random.choice(
                                            dataset_object.num_presets,
                                            p=dataset_object.preset_probs,
                                        )
                                    )

                                this_cfg = _apply_strength_controls(
                                    SmartAnomaly_Cfg(**asdict(dataset_object.anomaly_presets[this_preset_idx]())),
                                    dataset_object.global_cfg,
                                )
                                micro_count = getattr(
                                    dataset_object.global_cfg, 'micro_dimple_count', 5
                                ) if this_preset_idx == _MICRO_DIMPLE_PRESET_IDX else 1

                                if micro_count > 1:
                                    shifted_patch = patch_xyz.copy()
                                    for _d in range(micro_count):
                                        dimple_pt_idx = np.random.randint(0, len(shifted_patch))
                                        dimple_center = shifted_patch[dimple_pt_idx].copy()
                                        dimple_cfg = _apply_strength_controls(
                                            SmartAnomaly_Cfg(**asdict(dataset_object.anomaly_presets[this_preset_idx]())),
                                            dataset_object.global_cfg,
                                        )
                                        shifted_patch = dataset_object.generate_pseudo_anomaly(
                                            shifted_patch,
                                            patch_normal,
                                            dimple_center,
                                            anomlay_cfg=dimple_cfg,
                                        )
                                    entry_cfg = dimple_cfg
                                else:
                                    shifted_patch = dataset_object.generate_pseudo_anomaly(
                                        patch_xyz,
                                        patch_normal,
                                        patch_center_arr,
                                        anomlay_cfg=this_cfg,
                                    )
                                    entry_cfg = this_cfg

                                composed_new_xyz[patch_mask] = shifted_patch
                                entry_dict = asdict(entry_cfg)
                                radii_val = entry_dict.get('radii')
                                if isinstance(radii_val, (list, tuple)):
                                    entry_dict['radii'] = [float(x) for x in radii_val]
                                elif radii_val is None:
                                    entry_dict['radii'] = []
                                else:
                                    entry_dict['radii'] = [float(radii_val)]
                                entry_dict.update({
                                    'pseudo_anomaly_index': int(pseudo_idx),
                                    'preset_index': int(this_preset_idx),
                                    'preset_name': dataset_object.anomaly_presets[this_preset_idx].__name__,
                                    'selected_patch_index': int(patch_idx),
                                    'selected_patch_point_count': int(patch_xyz.shape[0]),
                                    'selected_patch_center_x': float(patch_center_arr[0]),
                                    'selected_patch_center_y': float(patch_center_arr[1]),
                                    'selected_patch_center_z': float(patch_center_arr[2]),
                                })
                                pseudo_entries.append(entry_dict)
                                selected_patch_point_counts.append(int(patch_xyz.shape[0]))
                                selected_preset_indices.append(int(this_preset_idx))
                                selected_preset_names.append(dataset_object.anomaly_presets[this_preset_idx].__name__)
                                selected_centers.append(
                                    patch_center_arr.astype(float).tolist()
                                )

                            shifted_xyz = composed_new_xyz[mlocal == -1]
                            anomlay_cfg = None
                            policy_metadata = {
                                'policy': 'af3ad_presets',
                                'synthesis_policy': synthesis_policy,
                                'synthesis_source': 'preset_composition',
                                'pseudo_anomaly_count': int(len(shift_index_list)),
                                'num_pseudo_anomalies': int(len(shift_index_list)),
                                'N': int(len(shift_index_list)),
                                'preset_indices': selected_preset_indices,
                                'presets': selected_preset_names,
                                'pseudo_anomaly_entries': pseudo_entries,
                                'selected_patch_indices': shift_index_list,
                                'selected_patch_point_counts': selected_patch_point_counts,
                                'selected_patch_centers': selected_centers,
                            }
                    else:
                        shifted_xyz = dataset_object.generate_pseudo_anomaly_original(
                            shift_xyz,
                            shift_normal,
                            center,
                        )

                    if not random_factory_generated:
                        new_xyz = xyz.copy()
                        new_xyz[mlocal == -1] = shifted_xyz
                        anomaly_mask = (mlocal == -1)

                    # Optionally limit number of points per group for rollout memory control.
                    if max_points_per_group is not None and new_xyz.shape[0] > max_points_per_group:
                        sel = np.random.choice(
                            new_xyz.shape[0], max_points_per_group, replace=False
                        )
                        new_xyz = new_xyz[sel]
                        xyz = xyz[sel]
                        normals = normals[sel]
                        anomaly_mask = anomaly_mask[sel]

                    density_norm_stats = {}
                    if bool(getattr(dataset_object.global_cfg, 'normalize_pseudo_anomaly_density', False)):
                        new_xyz, density_norm_stats = _normalize_displaced_region_density(
                            xyz,
                            new_xyz,
                            anomaly_mask,
                        )

                    gt_offset = new_xyz - xyz  # (Ni, 3)
                    sample_stats = _compute_sample_anomaly_stats(
                        xyz,
                        new_xyz,
                        gt_offset,
                        anomaly_mask,
                        dataset_object.global_cfg,
                    )
                    sample_meets_threshold = _sample_meets_strength_thresholds(
                        sample_stats,
                        dataset_object.global_cfg,
                    )

                    if anomlay_cfg is not None:
                        cfg_dict = asdict(anomlay_cfg)
                        radii_val = cfg_dict.get('radii')
                        if isinstance(radii_val, (list, tuple)):
                            cfg_dict['radii'] = [float(x) for x in radii_val]
                        elif radii_val is None:
                            cfg_dict['radii'] = []
                        else:
                            cfg_dict['radii'] = [float(radii_val)]
                        R_value = cfg_dict.get('R')
                        cfg_dict['R_value'] = float(R_value) if R_value is not None else None
                        B_value = cfg_dict.get('B')
                        cfg_dict['B_value'] = float(B_value) if B_value is not None else None
                    else:
                        cfg_dict = {
                            'policy': dataset_object.af3ad_policy,
                            'radii': [],
                            'R_value': None,
                            'B_value': None,
                        }
                        if policy_metadata:
                            cfg_dict.update(policy_metadata)
                    if mixed_policy_active and not policy_metadata:
                        cfg_dict['policy'] = 'af3ad_presets'
                        cfg_dict['synthesis_policy'] = synthesis_policy
                        cfg_dict['synthesis_source'] = 'preset' if mixed_use_preset else 'intact'
                        cfg_dict['preset_randomfactory_prob'] = preset_randomfactory_prob
                    if density_norm_stats:
                        cfg_dict.update(density_norm_stats)
                    cfg_dict['preset_index'] = int(preset_idx)
                    center_arr = np.asarray(center, dtype=np.float32).reshape(-1)
                    patch_centroid = (
                        shift_xyz_base.mean(axis=0)
                        if shift_xyz_base.size
                        else center_arr[:3]
                    )
                    patch_centroid = np.asarray(patch_centroid, dtype=np.float32).reshape(-1)
                    cfg_dict['selected_patch_index'] = shift_index_list[0]
                    cfg_dict['selected_patch_indices'] = shift_index_list
                    cfg_dict['selected_patch_point_count'] = int(shift_xyz_base.shape[0])
                    if center_arr.size >= 3:
                        cfg_dict.setdefault('selected_patch_center_x', float(center_arr[0]))
                        cfg_dict.setdefault('selected_patch_center_y', float(center_arr[1]))
                        cfg_dict.setdefault('selected_patch_center_z', float(center_arr[2]))
                    if patch_centroid.size >= 3:
                        cfg_dict['selected_patch_centroid_x'] = float(patch_centroid[0])
                        cfg_dict['selected_patch_centroid_y'] = float(patch_centroid[1])
                        cfg_dict['selected_patch_centroid_z'] = float(patch_centroid[2])
                    cfg_dict['selected_patch_shared_across_presets'] = bool(
                        len(set(shift_index_list)) < len(shift_index_list)
                    )

                    final_xyz = xyz
                    final_new_xyz = new_xyz
                    final_gt_offset = gt_offset
                    final_normals = normals
                    final_anomaly_mask = anomaly_mask
                    final_cfg_dict = cfg_dict
                    final_shift_xyz_base = shift_xyz_base
                    final_shift_index = shift_index
                    final_stats = sample_stats
                    final_below_threshold = not sample_meets_threshold

                    if sample_meets_threshold:
                        break

                if (
                    final_xyz is None
                    or final_new_xyz is None
                    or final_gt_offset is None
                    or final_normals is None
                    or final_anomaly_mask is None
                    or final_stats is None
                ):
                    raise RuntimeError("Pseudo anomaly generation did not produce a valid sample.")

                if final_below_threshold and max_regen_attempts > 0:
                    regen_exhausted = True

                diayn_valid = not (
                    final_below_threshold and weak_anomaly_fallback == "skip_diayn"
                )
                final_stats["below_threshold"] = float(final_below_threshold)
                final_stats["regen_attempts"] = float(regen_attempts_used)
                final_stats["regen_exhausted"] = float(regen_exhausted)

                xyz = final_xyz
                new_xyz = final_new_xyz
                gt_offset = final_gt_offset
                normals = final_normals
                anomaly_mask = final_anomaly_mask
                cfg_dict = final_cfg_dict
                shift_xyz_base = final_shift_xyz_base
                shift_index = final_shift_index

                # Refine file name
                file_name.append(f"{base_name}::act{action_idx}")

                # tensors per-item (for both legacy flat concat and BN3 sampling).
                xyz_t = torch.from_numpy(xyz)           # (Ni, 3)
                new_xyz_t = torch.from_numpy(new_xyz)   # (Ni, 3)
                gt_off_t = torch.from_numpy(gt_offset)  # (Ni, 3)

                xyz_original.append(xyz_t)
                xyz_shifted.append(new_xyz_t)
                gt_offset_list.append(gt_off_t)
                if track_reward_data:
                    normal_original.append(torch.from_numpy(normals))
                    reward_anomaly_masks.append(
                        torch.from_numpy(
                            anomaly_mask.astype(np.bool_, copy=False)
                        )
                    )

                # Keep per-item copies for BN3 outputs. These will be used to build B*N_fixed*3 later.
                xyz_original_per_item.append(xyz_t)
                xyz_shifted_per_item.append(new_xyz_t)
                offset_per_item.append(gt_off_t)

                # Voxelization using MinkowskiEngine
                quantized_coords, feats_all, index, inverse_index = ME.utils.sparse_quantize(
                    new_xyz, new_xyz,
                    quantization_size=this_voxel_size,
                    return_index=True,
                    return_inverse=True
                )

                v2p_index = inverse_index + total_voxel_num
                total_voxel_num += index.shape[0]

                total_point_num += inverse_index.shape[0]
                batch_count.append(total_point_num)

                xyz_voxel.append(quantized_coords)
                feat_voxel.append(feats_all)
                v2p_index_batch.append(v2p_index)

                # Track shift_index for both modes
                shift_indices_list.append(int(shift_index[0]))

                # Store patch point cloud for FPFH feature extraction in RL state
                patch_xyz_list.append(torch.from_numpy(
                    shift_xyz_base.copy()).to(torch.float32))

                split_sizes.append(int(inverse_index.shape[0]))
                action_map.append(int(action_idx))
                anomaly_cfg_records.append(cfg_dict)
                preset_indices.append(int(preset_idx))
                anomaly_changed_points.append(float(final_stats["changed_points"]))
                anomaly_changed_fraction.append(float(final_stats["changed_fraction"]))
                anomaly_mean_displacement.append(float(final_stats["mean_displacement"]))
                anomaly_max_displacement.append(float(final_stats["max_displacement"]))
                anomaly_l1.append(float(final_stats["anomaly_l1"]))
                anomaly_l2.append(float(final_stats["anomaly_l2"]))
                anomaly_mask_size.append(float(final_stats["anomaly_mask_size"]))
                anomaly_bbox_volume.append(float(final_stats["bbox_volume"]))
                anomaly_strength_score.append(float(final_stats["strength_score"]))
                anomaly_below_threshold.append(float(final_stats["below_threshold"]))
                anomaly_regen_attempts.append(float(final_stats["regen_attempts"]))
                anomaly_regen_exhausted.append(float(final_stats["regen_exhausted"]))
                diayn_valid_mask.append(float(diayn_valid))

        # MinkowskiEngine collate
        xyz_voxel_batch, feat_voxel_batch = ME.utils.sparse_collate(
            xyz_voxel, feat_voxel)
        xyz_original_cat = torch.cat(xyz_original, 0).to(torch.float32)
        xyz_shifted_cat = torch.cat(xyz_shifted, 0).to(torch.float32)
        v2p_index_batch = torch.cat(v2p_index_batch, 0).to(torch.int64)
        batch_count = torch.from_numpy(np.array(batch_count, dtype=np.int64))
        batch_offset = torch.cat(gt_offset_list, 0).to(torch.float32)

        # ==================================================
        # Build fixed-size (B, N_fixed, 3) tensors for point cloud features extraction
        # ==================================================
        def _sample_fixed(idx_len: int, N: int, gen: torch.Generator, device: torch.device):
            if idx_len >= N:
                return torch.randperm(idx_len, generator=gen, device=device)[:N]
            # repeat-to-fill strategy
            num_full = N // idx_len
            rem = N - num_full * idx_len
            base = torch.arange(idx_len, device=device)
            if rem > 0:
                extra = torch.randint(low=0, high=idx_len, size=(
                    rem,), generator=gen, device=device)
                return torch.cat([base.repeat(num_full), extra], dim=0)
            else:
                return base.repeat(num_full)

        B = len(xyz_original_per_item)
        device = xyz_original_cat.device  # tensors above are on CPU; preserve that

        bn3_xyz_list = []
        bn3_xyz_shifted_list = []
        bn3_offset_list = []
        bn3_indices = []
        orig_lengths = []

        for i in range(B):
            pts = xyz_original_per_item[i].to(device)    # (Ni,3)
            pts_new = xyz_shifted_per_item[i].to(device)  # (Ni,3)
            pts_off = offset_per_item[i].to(device)      # (Ni,3)
            Ni = pts.shape[0]
            orig_lengths.append(Ni)

            idx_sel = _sample_fixed(Ni, N_fixed, generator, device)
            bn3_indices.append(idx_sel)

            bn3_xyz_list.append(pts[idx_sel].unsqueeze(0))       # (1,N,3)
            bn3_xyz_shifted_list.append(pts_new[idx_sel].unsqueeze(0))
            bn3_offset_list.append(pts_off[idx_sel].unsqueeze(0))

        xyz_bn3 = torch.cat(bn3_xyz_list, dim=0).to(
            torch.float32)             # (B,N,3)
        xyz_shifted_bn3 = torch.cat(bn3_xyz_shifted_list, dim=0).to(
            torch.float32)  # (B,N,3)
        offset_bn3 = torch.cat(bn3_offset_list, dim=0).to(
            torch.float32)       # (B,N,3)
        bn3_indices = torch.stack(bn3_indices, dim=0).to(
            torch.int64)          # (B,N)
        orig_lengths = torch.tensor(
            orig_lengths, dtype=torch.int32)           # (B,)

        if (
            batch_preset_idx is not None
            and any(index != batch_preset_idx for index in preset_indices)
        ):
            raise RuntimeError(
                "Batch preset invariant violated: a preset batch contains "
                "more than one preset index"
            )

        out = {
            'xyz_voxel': xyz_voxel_batch,
            'feat_voxel': feat_voxel_batch,
            'xyz_original': xyz_original_cat,    # legacy flat concat
            'fn': file_name,
            'v2p_index': v2p_index_batch,
            'xyz_shifted': xyz_shifted_cat,
            'batch_count': batch_count,
            'batch_offset': batch_offset,

            # # --- new BN3 tensors --- # TODO: Uncomment if needed for RL
            # 'xyz_bn3': xyz_bn3,                         # (B, N_fixed, 3)
            # 'xyz_shifted_bn3': xyz_shifted_bn3,         # (B, N_fixed, 3)
            # 'offset_bn3': offset_bn3,                   # (B, N_fixed, 3)
            # # (B, N_fixed) per-item indices
            # 'bn3_indices': bn3_indices,
            # 'orig_lengths': orig_lengths,               # (B,)
            'N_fixed': int(N_fixed),
            # --- shift_indices for RL state ---
            # --- shift_indices for RL state ---
            'shift_indices': torch.as_tensor(shift_indices_list, dtype=torch.long),

            # --- patch point clouds for FPFH feature extraction ---
            # List of tensors, each of shape (N_patch_i, 3)
            'patch_xyz': patch_xyz_list,
            'split_sizes': torch.as_tensor(split_sizes, dtype=torch.long),
            'action_map': torch.as_tensor(action_map, dtype=torch.long),
            'anomaly_cfg_params': anomaly_cfg_records,
            'preset_indices': torch.as_tensor(preset_indices, dtype=torch.long),
            'anomaly_changed_points': torch.as_tensor(anomaly_changed_points, dtype=torch.float32),
            'anomaly_changed_fraction': torch.as_tensor(anomaly_changed_fraction, dtype=torch.float32),
            'anomaly_mean_displacement': torch.as_tensor(anomaly_mean_displacement, dtype=torch.float32),
            'anomaly_max_displacement': torch.as_tensor(anomaly_max_displacement, dtype=torch.float32),
            'anomaly_l1': torch.as_tensor(anomaly_l1, dtype=torch.float32),
            'anomaly_l2': torch.as_tensor(anomaly_l2, dtype=torch.float32),
            'anomaly_mask_size': torch.as_tensor(anomaly_mask_size, dtype=torch.float32),
            'anomaly_bbox_volume': torch.as_tensor(anomaly_bbox_volume, dtype=torch.float32),
            'anomaly_strength_score': torch.as_tensor(anomaly_strength_score, dtype=torch.float32),
            'anomaly_below_threshold': torch.as_tensor(anomaly_below_threshold, dtype=torch.float32),
            'anomaly_regen_attempts': torch.as_tensor(anomaly_regen_attempts, dtype=torch.float32),
            'anomaly_regen_exhausted': torch.as_tensor(anomaly_regen_exhausted, dtype=torch.float32),
            'diayn_valid_mask': torch.as_tensor(diayn_valid_mask, dtype=torch.float32),
            'batch_synthesis_policy': (
                batch_synthesis_policy
                or getattr(
                    dataset_object,
                    'synthesis_policy',
                    'af3ad_presets',
                )
            ),
            'batch_preset_index': batch_preset_idx,
        }
        if track_reward_data:
            out['normal_original'] = torch.cat(
                normal_original, 0
            ).to(torch.float32)
            out['reward_anomaly_mask'] = torch.cat(
                reward_anomaly_masks, 0
            ).to(torch.bool)

        return out

    def _resolve_mode(params):
        mode = params.get('collate_mode')
        if mode is not None:
            mode = str(mode).lower()
            if mode in {"rollout", "standard", "random"}:
                return mode
            raise ValueError(
                f"Unknown collate_mode '{mode}'. Expected 'rollout' or 'standard' or 'random'.")

        return 'random'  # Random mode is for training without RL; always sample random actions
        # actions_hint = params.get('actions', None)
        # if actions_hint is not None:
        #     actions_hint = np.asarray(actions_hint)
        #     if actions_hint.ndim == 2 and actions_hint.shape[0] > 1:
        #         return 'rollout'

        # return fallback

    def _dispatch_with_mode(id_list, params, forced_mode=None):
        params_local = dict(params) if isinstance(
            params, dict) else dict(params)
        if forced_mode is not None:
            params_local['collate_mode'] = forced_mode

        mode = _resolve_mode(params_local)
        if mode == 'rollout':
            actions_arr = _normalize_rollout_actions(params_local)
            return _collate_impl(id_list, params_local, actions_arr, rollout_mode=True)
        elif mode == 'standard':
            actions_arr = _select_best_action(params_local)
            return _collate_impl(id_list, params_local, actions_arr, rollout_mode=False)
        elif mode == 'random':
            if params_local.get('actions') is None:
                # Use weighted sampling based on preset_probs
                actions_arr = np.random.choice(
                    num_presets, size=(len(id_list),), p=preset_probs).astype(np.int64)
            else:
                actions_arr = np.asarray(
                    params_local.get('actions'), dtype=np.int64)
            return _collate_impl(id_list, params_local, actions_arr, rollout_mode=False, random_mode=True)

        raise ValueError(
            f"Unknown collate_mode '{mode}'. Expected 'rollout' or 'standard' or 'random'.")

    def _dispatch_auto(id_list):
        params = _get_params()
        return _dispatch_with_mode(id_list, params)

    def _dispatch_rollout(id_list):
        params = _get_params()
        return _dispatch_with_mode(id_list, params, forced_mode='rollout')

    def _dispatch_standard(id_list):
        params = _get_params()
        return _dispatch_with_mode(id_list, params, forced_mode='standard')

    def _dispatch_random(id_list):
        params = _get_params()
        return _dispatch_with_mode(id_list, params, forced_mode='random')

    return CollateBundle(
        standard=_dispatch_standard,
        rollout=_dispatch_rollout,
        random=_dispatch_random,
        dispatch=_dispatch_auto,
    )


class Dataset:
    def __init__(self, cfg):
        self.global_cfg = cfg
        self.batch_size = cfg.batch_size
        self.rollout_batch_size = getattr(cfg, 'rollout_batch_size', 1)
        self.dataset_workers = cfg.num_works
        self.data_repeat = cfg.data_repeat
        self.voxel_size = cfg.voxel_size
        self.mask_num = cfg.mask_num
        cache_dataset = getattr(cfg, 'cache_dataset', False)
        cache_test_set = getattr(cfg, 'cache_test_set', None)
        self.cache_dataset = bool(cache_dataset)
        self.cache_test_set = self.cache_dataset if cache_test_set is None else bool(
            cache_test_set)

        self.category = cfg.category
        self.category_list = self._list_categories()
        assert self.category in self.category_list

        # Build file lists BEFORE cache initialization to determine proper cache size
        self.train_file_list = self._build_train_file_list()
        self.test_file_list = self._build_test_file_list()

        # Calculate cache sizes based on actual number of unique files
        # The training list may have duplicates due to data_repeat, so count unique files
        unique_train_files = len(set(self.train_file_list))
        unique_test_files = len(set(self.test_file_list))

        # Cache size configuration
        # With persistent_workers, each worker process gets its own cache instance
        # Set cache size to hold all unique files plus a small buffer
        # Distribute unique files across workers, with a minimum per worker
        cache_buffer = 5  # Small buffer for safety
        self.cache_size_per_worker = max(
            50,  # Minimum per worker
            (unique_train_files + cache_buffer) // max(1, self.dataset_workers) + 1
        )
        self.test_cache_size_per_worker = max(
            25,  # Minimum per worker
            (unique_test_files + cache_buffer) // max(1, self.dataset_workers) + 1
        )

        # Initialize caches (will be re-initialized in worker processes)
        self._train_cache = BoundedCache(maxsize=self.cache_size_per_worker)
        self._test_cache = BoundedCache(
            maxsize=self.test_cache_size_per_worker)
        self.standard_param_queue = standard_param_queue
        self.random_param_queue = random_param_queue
        self.rollout_param_queue = rollout_param_queue
        self.anomaly_presets = AnomalyPreset(self.global_cfg).presets
        self.num_presets = len(self.anomaly_presets)
        self.synthesis_policy = normalize_synthesis_policy(
            getattr(cfg, 'synthesis_policy', 'af3ad_presets')
        )
        self.af3ad_policy = _normalize_af3ad_policy(
            getattr(cfg, 'af3ad_policy', 'plain')
        )
        if self.af3ad_policy == "random_factory":
            if self.synthesis_policy == "af3ad_presets":
                self.synthesis_policy = "randomfactory_raw"
            self.af3ad_policy = "plain"
        self.preset_randomfactory_prob = min(
            1.0,
            max(0.0, float(getattr(cfg, 'preset_randomfactory_prob', 0.5))),
        )

        # Parse preset distribution weights from config
        # Default to uniform weights (1.0 each) if not specified
        self.preset_weights = []
        for i in range(self.num_presets):
            weight = getattr(cfg, f'preset_{i}_weight', 1.0)
            self.preset_weights.append(float(weight))
        # Normalize to probabilities
        total_weight = sum(self.preset_weights)
        if total_weight > 0:
            self.preset_probs = np.array(
                [w / total_weight for w in self.preset_weights], dtype=np.float64)
        else:
            # Fallback to uniform if all weights are 0
            self.preset_probs = np.ones(
                self.num_presets, dtype=np.float64) / self.num_presets

        self.validation_suffixes = self._parse_validation_suffixes(
            getattr(cfg, 'validation_suffixes', ''))
        if self.global_cfg.validation:
            self.validation_file_list = self._build_validation_file_list()
        else:
            self.validation_file_list = []
        self._eval_file_list = self.test_file_list

        self.normal_tag = getattr(cfg, 'normal_tag', 'positive')
        self.gt_delimiter = getattr(cfg, 'gt_delimiter', ',')
        self.gt_mask_dir = self._default_gt_mask_dir()

        transform_module = self._get_transform_module()
        self.NormalizeCoord = transform_module.NormalizeCoord()
        self.CenterShift = transform_module.CenterShift(apply_z=True)
        self.RandomRotate_z = transform_module.RandomRotate(
            angle=[-1, 1], axis="z", center=[0, 0, 0], p=1.0)
        self.RandomRotate_y = transform_module.RandomRotate(
            angle=[-1, 1], axis="y", p=1.0)
        self.RandomRotate_x = transform_module.RandomRotate(
            angle=[-1, 1], axis="x", p=1.0)
        self.SphereCropMask = transform_module.SphereCropMask(
            part_num=self.mask_num)

        self.train_aug_compose = transform_module.Compose([self.CenterShift, self.RandomRotate_z, self.RandomRotate_y, self.RandomRotate_x,
                                                           self.NormalizeCoord, self.SphereCropMask])

        self.test_aug_compose = transform_module.Compose(
            [self.CenterShift, self.NormalizeCoord])

        # Initialize storage for pre-fetched shift indices
        # Disabled by default to reduce memory usage with large data_repeat values
        self.prefetched_shift_indices = None
        self.use_prefetched_shift_indices = getattr(
            cfg, 'use_prefetched_shift_indices', False)

        # Worker cache monitoring - track load counts for periodic logging
        self._worker_load_count = 0
        self._worker_log_interval = getattr(
            cfg, 'worker_cache_log_interval', 100)  # Log every N loads

        self.binary_anomaly_label = getattr(cfg, 'binary_anomaly_label', False)
        self.af3ad_synthesizer = PseudoAnomalySynthesizer(
            args=self.global_cfg,
            binary_anomaly_label=self.binary_anomaly_label,
        )
        self.random_factory = RandomFactory(self.global_cfg)
        # Fraction of training samples to keep intact (without anomaly)
        self.intact_ratio = getattr(cfg, 'intact_ratio', 0.0)

    def _worker_init_fn_(self, worker_id):
        """Initialize worker process with fresh caches and seeds.

        This is critical for preventing memory leaks with persistent_workers=True.
        Each worker process gets its own cache instance to ensure bounded memory usage.
        """
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        np.random.seed(np_seed)
        random.seed(np_seed)

        # Re-initialize caches in worker process to ensure they're independent
        # This prevents memory leaks when using persistent_workers=True
        if self.cache_dataset:
            self._train_cache = BoundedCache(
                maxsize=self.cache_size_per_worker)
        if self.cache_test_set:
            self._test_cache = BoundedCache(
                maxsize=self.test_cache_size_per_worker)

        # Reset worker load counter
        self._worker_load_count = 0

    # ------------------------------------------------------------------
    # Dataset configuration helpers (override in subclasses)
    # ------------------------------------------------------------------
    def _get_transform_module(self):
        """Return the transform module used to build augmentation pipelines."""
        return aug_transform

    def _list_categories(self):
        root = Path(self.global_cfg.dataset_base_dir + '/pcd')
        if not root.exists():
            return []
        return sorted([p.name for p in root.iterdir() if p.is_dir()])

    def _train_file_glob(self):
        return str(Path(self.global_cfg.dataset_base_dir + '/obj') / self.category / '*.obj')

    def _train_file_filter(self, candidates):
        pattern = re.compile(r'template')
        return sorted([fn for fn in candidates if pattern.search(fn)])

    def _build_train_file_list(self):
        data_list = glob.glob(self._train_file_glob())
        train_files = self._train_file_filter(data_list)
        return train_files * self.data_repeat

    def _test_file_glob(self):
        return str(Path(self.global_cfg.dataset_base_dir + '/pcd') / self.category / 'test' / '*.pcd')

    def _build_test_file_list(self):
        test_files = glob.glob(self._test_file_glob())
        test_files.sort()
        return test_files

    def _parse_validation_suffixes(self, suffixes_raw):
        if suffixes_raw is None:
            return None
        if isinstance(suffixes_raw, (list, tuple, set)):
            parsed = {int(s) for s in suffixes_raw}
            return parsed if parsed else None
        suffixes_str = str(suffixes_raw).strip()
        if suffixes_str.lower() in {'', 'auto', 'none'}:
            return None
        try:
            parts = suffixes_str.split(',')
            parsed = {int(p.strip()) for p in parts if p.strip()}
            return parsed if parsed else None
        except ValueError:
            return None

    def _extract_suffix_index(self, path_str: str):
        match = re.search(r'(\d+)$', Path(path_str).stem)
        if match is None:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _build_validation_file_list(self):
        if not self.test_file_list:
            return []

        indexed = []
        for fn in self.test_file_list:
            idx = self._extract_suffix_index(fn)
            if idx is not None:
                indexed.append((idx, fn))

        if not indexed:
            return []

        available_suffixes = sorted({idx for idx, _ in indexed})
        target_suffixes = set()
        if self.validation_suffixes:
            target_suffixes = {
                idx for idx in available_suffixes if idx in self.validation_suffixes}
        if not target_suffixes:
            target_suffixes = set(available_suffixes[:2])

        val_files = [fn for idx, fn in indexed if idx in target_suffixes]
        val_files.sort()

        print(
            f"Validation samples ({len(val_files)}) using suffixes {sorted(target_suffixes)}:")
        for fn in val_files:
            print(f"  - {Path(fn).name}")

        return val_files

    def _default_gt_mask_dir(self):
        return Path(self.global_cfg.dataset_base_dir + '/pcd') / self.category / 'GT'

    def _resolve_gt_path(self, sample_name: str) -> Path:
        return Path(self.gt_mask_dir) / f'{sample_name}.txt'

    def _read_anomalous_points(self, gt_path: Path) -> np.ndarray:
        kwargs = {}
        if self.gt_delimiter is not None:
            kwargs['delimiter'] = self.gt_delimiter
        data = np.loadtxt(gt_path, **kwargs)
        return data[:, 0:3]

    def _load_normal_point_cloud(self, fn_path: str) -> np.ndarray:
        if self.cache_test_set:
            cached = self._test_cache.get((fn_path, 'normal'))
            if cached is not None:
                return cached.copy()
        pcd = o3d.io.read_point_cloud(fn_path)
        points = np.asarray(pcd.points)
        if self.cache_test_set:
            self._test_cache.set((fn_path, 'normal'), points)
        return points

    def _load_anomalous_point_cloud(self, fn_path: str) -> np.ndarray:
        sample_name = Path(fn_path).stem
        gt_path = self._resolve_gt_path(sample_name)
        cache_key = (fn_path, 'anomaly')
        if self.cache_test_set:
            cached = self._test_cache.get(cache_key)
            if cached is not None:
                return cached.copy()
        points = self._read_anomalous_points(gt_path)
        if self.cache_test_set:
            self._test_cache.set(cache_key, points)
        return points

    def _load_train_point_cloud(self, fn_path: str):
        if self.cache_dataset:
            cached = self._train_cache.get(fn_path)
            if cached is not None:
                coord_cached, normal_cached = cached
                return coord_cached.copy(), normal_cached.copy()

        obj = o3d.io.read_triangle_mesh(fn_path)
        obj.compute_vertex_normals()                         # Compute normals
        # Extract vertices (N, 3)
        coord = np.asarray(obj.vertices)
        # Extract normals (N, 3)
        vertex_normals = np.asarray(obj.vertex_normals)

        if self.cache_dataset:
            self._train_cache.set(fn_path, (coord, vertex_normals))

            # Periodically log worker cache stats to track memory usage
            self._worker_load_count += 1
            if self._worker_log_interval > 0 and self._worker_load_count % self._worker_log_interval == 0:
                self._log_worker_cache_stats()

        return coord, vertex_normals

    def _load_test_point_cloud(self, fn_path: str):
        is_normal = self.normal_tag and self.normal_tag in Path(fn_path).name
        if is_normal:
            coord = self._load_normal_point_cloud(fn_path)
            label = 0
        else:
            coord = self._load_anomalous_point_cloud(fn_path)
            label = 1
        return coord, label

    def clear_cache(self, recreate=True):
        """
        Clear all cached data in the current process to free memory.

        IMPORTANT: With persistent_workers=True, each worker has its own cache instance.
        This method only clears the cache in the process where it's called (typically the
        main process). Worker caches are independent and will be managed automatically
        through the BoundedCache LRU eviction policy.

        For effective memory management with persistent workers:
        1. Use smaller cache sizes (already configured per worker)
        2. Worker caches auto-evict old entries via LRU when reaching maxsize
        3. Call this method periodically to clear the main process cache

        Args:
            recreate: If True, completely destroys and recreates cache objects.
                     This is more aggressive and helps ensure all memory is freed.

        Returns:
            dict: Information about cleared cache sizes
        """
        train_size = self._train_cache.size() if self.cache_dataset else 0
        test_size = self._test_cache.size() if self.cache_test_set else 0

        if self.cache_dataset:
            if recreate:
                # Completely destroy and recreate cache to ensure memory is freed
                # Set to None first to break any circular references, then delete
                old_cache = self._train_cache
                self._train_cache = None
                del old_cache
                # Create new cache object
                self._train_cache = BoundedCache(
                    maxsize=self.cache_size_per_worker)
            else:
                self._train_cache.clear()

        if self.cache_test_set:
            if recreate:
                # Completely destroy and recreate cache to ensure memory is freed
                # Set to None first to break any circular references, then delete
                old_cache = self._test_cache
                self._test_cache = None
                del old_cache
                # Create new cache object
                self._test_cache = BoundedCache(
                    maxsize=self.test_cache_size_per_worker)
            else:
                self._test_cache.clear()

        return {
            'train_cache_cleared': train_size,
            'test_cache_cleared': test_size,
            'recreated': recreate,
        }

    # TODO: Remove
    def get_cache_stats(self):
        """
        Get detailed cache statistics for monitoring.

        Returns dict with cache sizes, memory usage, and other metrics.
        Only returns stats for the current process (main or worker).
        """
        stats = {
            'process_type': 'main',  # Will be 'worker' if called from worker
            'cache_dataset_enabled': self.cache_dataset,
            'cache_test_set_enabled': self.cache_test_set,
        }

        # Check if we're in a worker process
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            stats['process_type'] = 'worker'
            stats['worker_id'] = worker_info.id

        if self.cache_dataset:
            train_stats = self._train_cache.get_stats()
            train_memory = self._train_cache.estimate_memory_bytes()
            stats['train_cache'] = {
                **train_stats,
                'memory_mb': train_memory / (1024 * 1024),
            }
        else:
            stats['train_cache'] = None

        if self.cache_test_set:
            test_stats = self._test_cache.get_stats()
            test_memory = self._test_cache.estimate_memory_bytes()
            stats['test_cache'] = {
                **test_stats,
                'memory_mb': test_memory / (1024 * 1024),
            }
        else:
            stats['test_cache'] = None

        # Add dataset configuration info
        stats['config'] = {
            'num_workers': self.dataset_workers,
            'cache_size_per_worker': self.cache_size_per_worker,
            'test_cache_size_per_worker': self.test_cache_size_per_worker,
            'data_repeat': self.data_repeat,
            'unique_train_files': len(set(self.train_file_list)),
            'total_train_entries': len(self.train_file_list),
        }

        return stats

    # TODO: Remove
    def _log_worker_cache_stats(self):

        """
        Log cache statistics from within a worker process.

        This method is called periodically during data loading to monitor
        worker cache behavior. It logs directly to stdout/stderr since
        logger instances may not work properly across process boundaries.
        """
        import logging
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return  # Not in a worker

        worker_id = worker_info.id
        stats = self.get_cache_stats()

        # Create a simple logger that writes to stderr for worker processes
        # This ensures we can see worker logs even if main logger doesn't capture them
        print(
            f"[WORKER {worker_id}] Cache Stats after {self._worker_load_count} loads:", file=sys.stderr)

        if stats['train_cache']:
            tc = stats['train_cache']
            print(
                f"[WORKER {worker_id}]   Train Cache: {tc['size']}/{tc['maxsize']} "
                f"({tc['utilization']*100:.1f}% full, {tc['memory_mb']:.2f} MB, "
                f"{tc['evictions']} evictions, {tc['total_sets']} total sets)",
                file=sys.stderr
            )

        if stats['test_cache']:
            tc = stats['test_cache']
            print(
                f"[WORKER {worker_id}]   Test Cache: {tc['size']}/{tc['maxsize']} "
                f"({tc['utilization']*100:.1f}% full, {tc['memory_mb']:.2f} MB)",
                file=sys.stderr
            )

    def prefetch_shift_indices(self, seed=None):
        """
        Pre-fetch shift indices for all training samples.

        This method generates shift_index values for each training sample ahead of time,
        which can be used during training for reproducibility and for RL state construction.

        Args:
            seed: Random seed for reproducibility. If None, uses the manual_seed from config,
                  or defaults to 42 if manual_seed is not set.

        Returns:
            dict: A dictionary mapping sample indices to their pre-fetched shift indices.
                  Format: {sample_idx: shift_index}
        """
        # Default seed for reproducibility when manual_seed is not configured
        DEFAULT_PREFETCH_SEED = 42

        if seed is None:
            seed = getattr(self.global_cfg, 'manual_seed',
                           DEFAULT_PREFETCH_SEED)

        rng = np.random.default_rng(seed)

        # Calculate mask_range (same as in collate function)
        # mask_range = np.arange(0, self.mask_num // 2)
        mask_range = np.arange(0, self.mask_num)
        num_shift = 1

        # Pre-allocate dictionary for better performance
        num_samples = len(self.train_file_list)
        prefetched_indices = {}

        # Pre-fetch shift indices for all training samples
        for idx in range(num_samples):
            # Generate one shift_index per sample
            shift_index = rng.choice(mask_range, num_shift, replace=False)
            prefetched_indices[idx] = shift_index

        self.prefetched_shift_indices = prefetched_indices
        return prefetched_indices

    def trainLoader(self):
        # Creates training dataset indecies.
        train_set = list(range(len(self.train_file_list)))

        # Pre-fetch shift indices if enabled
        if self.use_prefetched_shift_indices:
            self.prefetch_shift_indices()

        standard_collate = make_collate(self, standard_param_queue)
        random_collate = make_collate(self, random_param_queue)
        rollout_collate = make_collate(self, rollout_param_queue)

        self.train_standard_collate = standard_collate
        self.train_random_collate = random_collate
        self.train_rollout_collate = rollout_collate

        # Create generator for deterministic shuffling
        generator = None
        manual_seed = getattr(self.global_cfg, 'manual_seed', None)
        if manual_seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(manual_seed))

        self.train_data_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            collate_fn=standard_collate.standard,
            num_workers=self.dataset_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn_,
            persistent_workers=True,  # recommended for speed; safe with Manager proxy
            # minimize memory usage (1 batch prefetched per worker)
            prefetch_factor=1,
            generator=generator,
        )

        self.train_random_data_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            collate_fn=random_collate.random,
            num_workers=self.dataset_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn_,
            persistent_workers=True,  # recommended for speed; safe with Manager proxy
            # minimize memory usage (1 batch prefetched per worker)
            prefetch_factor=1,
            generator=generator,
        )

        # Create separate generator for rollout loader
        rollout_generator = None
        if manual_seed is not None:
            rollout_generator = torch.Generator()
            # Different seed to avoid same order
            rollout_generator.manual_seed(int(manual_seed) + 1)

        rollout_batch_size = max(
            1, int(getattr(self, 'rollout_batch_size', 1)))
        self.train_rollout_data_loader = DataLoader(
            train_set,
            batch_size=rollout_batch_size,
            collate_fn=rollout_collate.rollout,
            num_workers=self.dataset_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn_,
            persistent_workers=True,
            prefetch_factor=1,
            generator=rollout_generator,
        )

    def testLoader(self):
        # Creates test dataset indecies.
        self._eval_file_list = self.test_file_list
        test_set = list(range(len(self.test_file_list)))

        # Initializes the test data loader with the specified parameters and custom collate function. Note that collate_fn is a custom function (self.testMerge) to merge and preprocess data for each batch.
        self.test_data_loader = DataLoader(test_set, batch_size=1, collate_fn=self.testMerge,
                                           num_workers=self.dataset_workers,
                                           shuffle=False, sampler=None,
                                           drop_last=False, pin_memory=False,
                                           worker_init_fn=self._worker_init_fn_)

    def valLoader(self):
        # Creates validation dataset indices based on a pre-filtered file list.
        self._eval_file_list = self.validation_file_list
        val_set = list(range(len(self.validation_file_list)))

        if len(val_set) == 0:
            self.val_data_loader = None
            return

        self.val_data_loader = DataLoader(val_set, batch_size=1, collate_fn=self.testMerge,
                                          num_workers=self.dataset_workers,
                                          shuffle=False, sampler=None,
                                          drop_last=False, pin_memory=False,
                                          worker_init_fn=self._worker_init_fn_)

    def generate_pseudo_anomaly_original(self, points, normals, center, distance_to_move=0.08):
        return self.af3ad_synthesizer.generate_original(
            points,
            normals,
            center,
            distance_to_move=distance_to_move,
        )

    def generate_pseudo_anomaly(self, points, normals, center, anomlay_cfg=None):
        return self.af3ad_synthesizer.generate(
            points,
            normals,
            center,
            anomaly_cfg=anomlay_cfg,
        )

    def generate_pseudo_anomaly_with_policy(self, points, normals, center):
        return self.af3ad_synthesizer.generate_with_policy(
            points,
            normals,
            center,
            policy=self.af3ad_policy,
            return_metadata=True,
        )

    def testMerge(self, id, N_fixed: int = 2048, generator: torch.Generator = None):
        file_list = getattr(self, '_eval_file_list', self.test_file_list)
        file_name = []
        xyz_voxel = []
        feat_voxel = []
        xyz_original_per_sample = []     # keep per-sample for BN3
        xyz_original_cat = []            # old flat concat for backward-compat
        v2p_index_batch = []
        labels = []

        total_voxel_num = 0
        total_point_num = 0
        batch_count = [0]

        # Optional RNG for reproducibility
        if generator is None:
            generator = torch.Generator()
            # you can set a seed here if you want determinism:
            # generator.manual_seed(0)

        for i, idx in enumerate(id):
            fn_path = file_list[idx]
            file_name.append(fn_path)

            coord, label = self._load_test_point_cloud(fn_path)

            # ---- Data aug
            Point_dict = {'coord': coord}
            Point_dict = self.test_aug_compose(Point_dict)

            # ---- numpy -> float32
            xyz = Point_dict['coord'].astype(np.float32)

            # ---- Quantize for ME sparse path (unchanged)
            quantized_coords, feats_all, index, inverse_index = ME.utils.sparse_quantize(
                xyz, xyz,
                quantization_size=self.voxel_size,
                return_index=True,
                return_inverse=True
            )

            v2p_index = inverse_index + total_voxel_num
            total_voxel_num += index.shape[0]
            total_point_num += inverse_index.shape[0]
            batch_count.append(total_point_num)

            # ---- Accumulate for ME batch
            xyz_voxel.append(quantized_coords)
            feat_voxel.append(feats_all)
            xyz_t = torch.from_numpy(xyz)          # (Ni, 3)
            xyz_original_per_sample.append(xyz_t)  # save per-sample
            xyz_original_cat.append(xyz_t)         # legacy flat concat
            v2p_index_batch.append(v2p_index)

            labels.append(label)

        # ---- Collate ME sparse (unchanged)
        xyz_voxel_batch, feat_voxel_batch = ME.utils.sparse_collate(
            xyz_voxel, feat_voxel)
        xyz_original = torch.cat(xyz_original_cat, 0).to(torch.float32)
        v2p_index_batch = torch.cat(v2p_index_batch, 0).to(torch.int64)
        labels = torch.from_numpy(np.array(labels))
        batch_count = torch.from_numpy(np.array(batch_count))

        # =======================
        # Build BN3 tensors here
        # =======================
        B = len(xyz_original_per_sample)
        device = xyz_original.device  # keep on CPU by default; move later if you prefer
        bn3_list = []
        # indices into each sample’s original points (before BN3 sampling)
        bn3_indices = []
        orig_lengths = []

        for i in range(B):
            pts = xyz_original_per_sample[i].to(device)              # (Ni, 3)
            Ni = pts.shape[0]
            orig_lengths.append(Ni)

            if Ni >= N_fixed:
                # random sample without replacement
                idx_sel = torch.randperm(
                    Ni, generator=generator, device=pts.device)[:N_fixed]
                sampled = pts[idx_sel]
            else:
                # repeat points to reach N_fixed (keeps all points, unbiased random fill)
                num_full = N_fixed // Ni
                rem = N_fixed - num_full * Ni
                base = torch.arange(Ni, device=pts.device)
                if rem > 0:
                    extra = torch.randint(low=0, high=Ni, size=(
                        rem,), generator=generator, device=pts.device)
                    idx_sel = torch.cat([base.repeat(num_full), extra], dim=0)
                else:
                    idx_sel = base.repeat(num_full)
                sampled = pts[idx_sel]

            bn3_list.append(sampled.unsqueeze(0))  # (1, N_fixed, 3)
            bn3_indices.append(idx_sel)            # (N_fixed,)

        xyz_bn3 = torch.cat(bn3_list, dim=0).to(
            torch.float32)       # (B, N_fixed, 3)
        bn3_indices = torch.stack(
            bn3_indices, dim=0)                # (B, N_fixed)
        orig_lengths = torch.tensor(orig_lengths, dtype=torch.int32)  # (B,)

        # You can optionally move BN3 to GPU here:
        # xyz_bn3 = xyz_bn3.cuda(non_blocking=True)
        # bn3_indices = bn3_indices.cuda(non_blocking=True)
        # orig_lengths = orig_lengths.cuda(non_blocking=True)

        return {
            # existing outputs (unchanged)
            'xyz_voxel': xyz_voxel_batch,
            'feat_voxel': feat_voxel_batch,
            'xyz_original': xyz_original,         # flat concat (legacy)
            'fn': file_name,
            'v2p_index': v2p_index_batch,
            'labels': labels,
            'batch_count': batch_count,

            # new fixed-size per-sample outputs
            'xyz_bn3': xyz_bn3,                   # (B, N_fixed, 3)
            # (B, N_fixed) indices into per-sample original points
            'bn3_indices': bn3_indices,
            'orig_lengths': orig_lengths,         # (B,)
            'N_fixed': int(N_fixed),
        }
