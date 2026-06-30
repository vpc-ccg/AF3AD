"""Real3D-AD dataset preprocessing utilities.

This module adapts the AnomalyShapeNet preprocessing pipeline so the new
PO3AD training code can work with Real3D without duplicating the large
amount of shared logic. Only the dataset discovery rules and a handful
of defaults differ between the two datasets, so we subclass the base
`Dataset` implementation and override the pieces that depend on file
structure or naming conventions.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Iterable, List
from integrations.po3ad.data.Real3D.downsampling import apply_downsampling
import numpy as np
import open3d as o3d
from typing import Iterable, List, Optional
import re
import warnings


from integrations.po3ad.data.AnomalyShapeNet.preprocessing import (
    Dataset as _BaseDataset,
    SmartAnomaly_Cfg,
    AnomalyPreset,
    make_collate,
    manager,
    standard_param_queue,
    rollout_param_queue,
    param_queue,
)

import integrations.po3ad.data.Real3D.transform as aug_transform

__all__ = [
    "Dataset",
    "SmartAnomaly_Cfg",
    "AnomalyPreset",
    "make_collate",
    "manager",
    "standard_param_queue",
    "rollout_param_queue",
    "param_queue",
]


class Dataset(_BaseDataset):
    """Real3D dataset that reuses the AnomalyShapeNet augmentation logic."""
    def __init__(self, cfg):
        
        # Select training data directory based on train_data_type argument
        self._train_data_type = getattr(cfg, 'train_data_type', 'cut')
        if self._train_data_type == 'pcd':
            self._train_mesh_root = Path(cfg.dataset_base_dir+"/Real3D-AD-PCD")
        elif self._train_data_type == 'cut':
            self._train_mesh_root = Path(cfg.dataset_base_dir+"/Real3D-AD-PLY-CUT")
        elif self._train_data_type == 'cut_full':
            self._train_mesh_root = Path(cfg.dataset_base_dir+"/Real3D-AD-PLY-CUT+FULL")
        else:
            # Default to cut for backwards compatibility
            self._train_mesh_root = Path(cfg.dataset_base_dir+"/Real3D-AD-PLY-CUT")
        
        self._test_root = Path(cfg.dataset_base_dir+"/Real3D-AD-PCD")
        print(f"Training data type: {self._train_data_type}, path: {self._train_mesh_root}")
        super().__init__(cfg)

        # Override defaults that differ from AnomalyShapeNet when the caller
        # did not provide explicit overrides in the config object.
        if not hasattr(cfg, "normal_tag"):
            self.normal_tag = "good"
        if not hasattr(cfg, "gt_delimiter"):
            self.gt_delimiter = None
        self.gt_mask_dir = self._default_gt_mask_dir()
        
        # Store downsampling configuration for training          
        self.downsample_mode = getattr(cfg, 'downsample_mode', 'none')
        self.downsample_ratio = getattr(cfg, 'downsample_ratio', 0.5)
        self.downsample_voxel_size = getattr(cfg, 'downsample_voxel_size', None)
        self.downsample_voxel_size_multiplier = getattr(cfg, 'downsample_voxel_size_multiplier', 2.0)
        self.downsample_target_points = getattr(cfg, 'downsample_target_points', None)
        self.downsample_recompute_normals = getattr(cfg, 'downsample_recompute_normals', True)
        self.downsample_random_seed = getattr(cfg, 'downsample_random_seed', 42)
        
        # Test set downsampling configuration
        # When enabled, applies voxel downsampling to test point clouds while 
        # maintaining proper point-to-label correspondence using KD-Tree
        self.test_downsample_enabled = getattr(cfg, 'test_downsample_enabled', False)
        self.test_downsample_voxel_size = getattr(cfg, 'test_downsample_voxel_size', 0.005)

        # Random plane cut augmentation (partial scan simulation)
        # Applied with probability p to simulate partial scans during training
        self.plane_cut_enabled = getattr(cfg, 'plane_cut_enabled', True)
        if self.plane_cut_enabled:
            self.RandomPlaneCut = aug_transform.RandomPlaneCut(
                p=getattr(cfg, 'plane_cut_prob', 0.7),
                r_min=getattr(cfg, 'plane_cut_r_min', 0.3),
                r_max=getattr(cfg, 'plane_cut_r_max', 0.9),
                horizontal_prob=getattr(cfg, 'plane_cut_horizontal_prob', 0.7),
                horizontal_angle_max=getattr(cfg, 'plane_cut_horizontal_angle_max', 20.0),
                min_points=getattr(cfg, 'plane_cut_min_points', 1024),
            )
            # Rebuild training augmentation with RandomPlaneCut
            self.train_aug_compose = aug_transform.Compose([
                self.RandomPlaneCut,  # Apply plane cut first (before centering)
                self.CenterShift,
                self.RandomRotate_z,
                self.RandomRotate_y,
                self.RandomRotate_x,
                self.NormalizeCoord,
                self.SphereCropMask,
            ])
        
        # Random edge segment cutout augmentation
        # Applied after SphereCropMask to remove edge segments (similar to cutout in images)
        self.edge_cutout_enabled = getattr(cfg, 'edge_cutout_enabled', False)
        if self.edge_cutout_enabled:
            self.RandomEdgeSegmentCutout = aug_transform.RandomEdgeSegmentCutout(
                p=getattr(cfg, 'edge_cutout_prob', 0.5),
                max_segments=getattr(cfg, 'edge_cutout_max_segments', 3),
                k_neighbors=getattr(cfg, 'edge_cutout_k_neighbors', 5),
                edge_threshold_percentile=getattr(cfg, 'edge_cutout_threshold_percentile', 70.0),
                min_points=getattr(cfg, 'edge_cutout_min_points', 1024),
            )
        else:
            self.RandomEdgeSegmentCutout = None
        
        print(self.train_aug_compose)
        
        
    # ------------------------------------------------------------------
    # Overrides for dataset-specific configuration
    # ------------------------------------------------------------------
    def _get_transform_module(self):
        return aug_transform

    def _list_categories(self) -> List[str]:
        """ Gets the list of categories available in both training and test sets.

        Returns:
            List[str]: _description_
        """
        ply_categories = self._collect_dir_names(self._train_mesh_root)
        pcd_categories = self._collect_dir_names(self._test_root)
        if ply_categories and pcd_categories:
            categories = sorted(set(ply_categories) & set(pcd_categories))
        else:
            categories = ply_categories or pcd_categories
        return categories

    def _collect_dir_names(self, root: Path) -> List[str]:
        """Collects the names of all subdirectories in the given root directory.

        Args:
            root (Path): _description_

        Returns:
            List[str]: _description_
        """
        if not root.exists():
            return []
        return sorted([p.name for p in root.iterdir() if p.is_dir()])

    def _train_file_glob(self) -> str:
        if self._train_data_type == 'pcd':
            # For PCD data, files are in category/train/*.pcd
            return str(self._train_mesh_root / self.category / "train" / "*.pcd")
        else:
            # For PLY-CUT and PLY-CUT+FULL, files are in category/*.ply
            return str(self._train_mesh_root / self.category / "*.ply")

    def _train_file_filter(self, candidates: Iterable[str]):
        if self._train_data_type == 'pcd':
            # For PCD data, filter for good/normal samples (template files)
            # Real3D-AD train folder contains *_template*.pcd files as training data
            pattern = re.compile(r"template")
            return sorted([fn for fn in candidates if pattern.search(fn)])
        else:
            # For PLY data, use the template pattern
            pattern = self._train_template_pattern
            return sorted([fn for fn in candidates if pattern.search(fn)])

    @property
    def _train_template_pattern(self):
        # Real3D templates follow the same naming convention as the original
        # code (e.g. ``*_template*.ply``), so we keep the same regex.
        return re.compile(r"template")

    def _build_train_file_list(self):
        """Builds a lits of training files for the current category.

        Returns:
            _type_: _description_
        """
        data_list = glob.glob(self._train_file_glob())
        train_files = self._train_file_filter(data_list)
        return train_files * self.data_repeat

    def _test_file_glob(self) -> str:
        return str(self._test_root / self.category / "test" / "*.pcd")

    def _default_gt_mask_dir(self) -> Path:
        return self._test_root / self.category / "gt"
    
    def _load_train_point_cloud(self, fn_path: str):
        """Override the default function in AnomalyShapeNet Dataset class to add point cloud downsampling support for Real3D dataset.
        
        This method loads a training point cloud (PLY mesh or PCD), applies optional
        downsampling based on config parameters, and returns vertices and normals.
        """
        # Check cache first
        if self.cache_dataset:
            cached = self._train_cache.get(fn_path)
            if cached is not None:
                coord_cached, normal_cached = cached
                return coord_cached.copy(), normal_cached.copy()

        # Load based on file type
        if fn_path.endswith('.pcd'):
            # Load PCD file (point cloud)
            pcd = o3d.io.read_point_cloud(fn_path)
            coord = np.asarray(pcd.points)
            # Estimate normals if not present
            if not pcd.has_normals():
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30))
            vertex_normals = np.asarray(pcd.normals)
        else:
            # Load PLY mesh
            obj = o3d.io.read_triangle_mesh(fn_path)
            obj.compute_vertex_normals()
            coord = np.asarray(obj.vertices)
            vertex_normals = np.asarray(obj.vertex_normals)

        # Apply downsampling if configured
        if self.downsample_mode != 'none':
            coord, vertex_normals = apply_downsampling(
                points=coord,
                normals=vertex_normals,
                downsample_mode=self.downsample_mode,
                downsample_ratio=self.downsample_ratio,
                voxel_size=self.downsample_voxel_size,
                voxel_size_multiplier=self.downsample_voxel_size_multiplier,
                target_num_points=self.downsample_target_points,
                recompute_normals_after_downsample=self.downsample_recompute_normals,
                random_seed=self.downsample_random_seed,
            )

        # Cache if enabled
        if self.cache_dataset:
            self._train_cache.set(fn_path, (coord, vertex_normals))

        return coord, vertex_normals
    
    
    def _load_test_point_cloud(self, fn_path: str):
        """Override test point cloud loading for Real3D-AD dataset.
        
        Real3D-AD test data has a special two-part structure where the 4th column
        contains labels (0 = normal points, 1 = anomalous points). This method
        handles loading, optional voxel downsampling, and ground truth generation.
        
        When test_downsample_enabled is True:
        1. Separates points by label (normal vs anomalous)
        2. Combines and applies voxel downsampling  
        3. Re-associates each downsampled point with its original label using KD-Tree
        4. Generates binary ground truth mask
        
        Returns:
            tuple: (coordinates, sample_label, point_gt_mask)
                - coordinates: (N, 3) point cloud after optional downsampling
                - sample_label: 0 for normal samples, 1 for anomalous samples
                - point_gt_mask: (N,) binary mask, 1.0 for anomalous points
        """
        is_normal = self.normal_tag and self.normal_tag in Path(fn_path).name
        
        if is_normal:
            # For normal samples, load from PCD file
            pcd = o3d.io.read_point_cloud(fn_path)
            points = np.asarray(pcd.points)
            sample_label = 0
            gt_mask = np.zeros(points.shape[0], dtype=np.float32)
            
            # Apply downsampling if enabled (normal samples have no anomaly labels)
            if self.test_downsample_enabled:
                pcd_ds = pcd.voxel_down_sample(voxel_size=self.test_downsample_voxel_size)
                points = np.asarray(pcd_ds.points)
                gt_mask = np.zeros(points.shape[0], dtype=np.float32)
        else:
            # For anomalous samples, load from GT text file with point labels
            sample_name = Path(fn_path).stem
            gt_path = self._resolve_gt_path(sample_name)
            
            # Load points with labels (4th column is the anomaly label)
            input_data = np.loadtxt(gt_path)
            input_points = input_data[:, 0:3]  # XYZ coordinates
            input_labels = input_data[:, 3]    # Anomaly labels (0 or 1)
            
            sample_label = 1
            
            if self.test_downsample_enabled:
                # Separate points based on label
                idx_normal = input_labels == 0  # Normal/reference points
                idx_anomaly = input_labels == 1  # Anomalous points
                
                normal_points = input_points[idx_normal]
                anomaly_points = input_points[idx_anomaly]
                
                # Create two separate point clouds
                pcd_normal = o3d.geometry.PointCloud()
                pcd_normal.points = o3d.utility.Vector3dVector(normal_points)
                
                pcd_anomaly = o3d.geometry.PointCloud()
                pcd_anomaly.points = o3d.utility.Vector3dVector(anomaly_points)
                
                # Combine for downsampling
                pcd_combined = pcd_normal + pcd_anomaly
                pcd_combined_ds = pcd_combined.voxel_down_sample(
                    voxel_size=self.test_downsample_voxel_size
                )
                
                # Get downsampled points
                points_ds = np.asarray(pcd_combined_ds.points)
                
                # Use KD-Tree to find which original point each downsampled point
                # corresponds to, maintaining point-to-label correspondence
                pc_len_normal = len(normal_points)  # Number of normal points
                pcd_tree = o3d.geometry.KDTreeFlann(pcd_combined)
                
                # Pre-allocate arrays for better performance
                # We'll use boolean masks instead of list append operations
                num_ds_points = len(points_ds)
                is_normal_mask = np.zeros(num_ds_points, dtype=bool)
                
                # Vectorized-friendly loop using KD-Tree search
                for i, pt in enumerate(points_ds):
                    [k, idx, _] = pcd_tree.search_knn_vector_3d(pt, 1)
                    if idx[0] < pc_len_normal:
                        is_normal_mask[i] = True
                
                # Separate points using boolean indexing (more efficient)
                normal_points_new = points_ds[is_normal_mask]
                anomaly_points_new = points_ds[~is_normal_mask]
                
                # Combine downsampled points (normal first, then anomalous)
                if len(normal_points_new) > 0 and len(anomaly_points_new) > 0:
                    points = np.vstack([normal_points_new, anomaly_points_new])
                elif len(normal_points_new) > 0:
                    points = normal_points_new
                elif len(anomaly_points_new) > 0:
                    points = anomaly_points_new
                else:
                    # Fallback: use original points if downsampling failed
                    warnings.warn(
                        f"Test downsampling produced no valid points for {fn_path}. "
                        "Using original points instead. Consider adjusting voxel_size.",
                        RuntimeWarning
                    )
                    points = input_points
                
                # Create binary ground truth labels
                # Normal points are first, anomalous points follow
                gt_mask = np.zeros(points.shape[0], dtype=np.float32)
                if len(normal_points_new) > 0:
                    gt_mask[len(normal_points_new):] = 1.0
                else:
                    # All points are anomalous
                    gt_mask[:] = 1.0
            else:
                # No downsampling - use original points and labels
                points = input_points
                gt_mask = np.where(input_labels > 0.5, 1.0, 0.0).astype(np.float32)
        
        return points, sample_label, gt_mask
    
    
    def _extract_suffix_index(self, path_str: str) -> Optional[int]:
        """Extract sample index from Real3D test file names.

        Real3D test files use a prefix pattern like ``142_good.pcd`` or 
        ``142_good_cut.pcd`` where the number is at the beginning. This
        overrides the base class method which expects numbers at the end.

        Args:
            path_str: Path to the test file.

        Returns:
            The extracted numeric prefix, or None if not found.
        """
        stem = Path(path_str).stem  # e.g. "142_good" or "142_good_cut"
        match = re.match(r'^(\d+)', stem)
        if match is None:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None
        
    def testMerge(self, id, N_fixed: int = 2048, generator=None):
        """Override testMerge to include point-level ground truth masks.
        
        Real3D-AD test data requires special handling because:
        1. The _load_test_point_cloud returns (coord, label, gt_mask)
        2. When downsampling is enabled, the gt_mask is generated based on
           KD-Tree nearest neighbor association
        
        Returns:
            dict with additional 'gt_masks' key containing point-level ground truth
        """
        import torch
        import MinkowskiEngine as ME
        
        file_list = getattr(self, '_eval_file_list', self.test_file_list)
        file_name = []
        xyz_voxel = []
        feat_voxel = []
        xyz_original_per_sample = []
        xyz_original_cat = []
        v2p_index_batch = []
        labels = []
        gt_masks = []  # Point-level ground truth masks
        
        total_voxel_num = 0
        total_point_num = 0
        batch_count = [0]
        
        if generator is None:
            generator = torch.Generator()
        
        for i, idx in enumerate(id):
            fn_path = file_list[idx]
            file_name.append(fn_path)
            
            # Real3D _load_test_point_cloud returns (coord, label, gt_mask)
            coord, label, gt_mask = self._load_test_point_cloud(fn_path)
            
            # Data augmentation
            Point_dict = {'coord': coord}
            Point_dict = self.test_aug_compose(Point_dict)
            
            xyz = Point_dict['coord'].astype(np.float32)
            
            # Quantize for MinkowskiEngine sparse path
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
            
            xyz_voxel.append(quantized_coords)
            feat_voxel.append(feats_all)
            xyz_t = torch.from_numpy(xyz)
            xyz_original_per_sample.append(xyz_t)
            xyz_original_cat.append(xyz_t)
            v2p_index_batch.append(v2p_index)
            
            labels.append(label)
            gt_masks.append(torch.from_numpy(gt_mask))
        
        # Collate MinkowskiEngine sparse tensors
        xyz_voxel_batch, feat_voxel_batch = ME.utils.sparse_collate(
            xyz_voxel, feat_voxel)
        xyz_original = torch.cat(xyz_original_cat, 0).to(torch.float32)
        v2p_index_batch = torch.cat(v2p_index_batch, 0).to(torch.int64)
        labels = torch.from_numpy(np.array(labels))
        batch_count = torch.from_numpy(np.array(batch_count))
        gt_masks_cat = torch.cat(gt_masks, 0).to(torch.float32)
        
        # Build BN3 tensors
        B = len(xyz_original_per_sample)
        device = xyz_original.device
        bn3_list = []
        bn3_indices = []
        orig_lengths = []
        
        for i in range(B):
            pts = xyz_original_per_sample[i].to(device)
            Ni = pts.shape[0]
            orig_lengths.append(Ni)
            
            if Ni >= N_fixed:
                idx_sel = torch.randperm(
                    Ni, generator=generator, device=pts.device)[:N_fixed]
                sampled = pts[idx_sel]
            else:
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
            
            bn3_list.append(sampled.unsqueeze(0))
            bn3_indices.append(idx_sel)
        
        xyz_bn3 = torch.cat(bn3_list, dim=0).to(torch.float32)
        bn3_indices = torch.stack(bn3_indices, dim=0)
        orig_lengths = torch.tensor(orig_lengths, dtype=torch.int32)
        
        return {
            'xyz_voxel': xyz_voxel_batch,
            'feat_voxel': feat_voxel_batch,
            'xyz_original': xyz_original,
            'fn': file_name,
            'v2p_index': v2p_index_batch,
            'labels': labels,
            'batch_count': batch_count,
            'gt_masks': gt_masks_cat,  # Point-level ground truth
            'xyz_bn3': xyz_bn3,
            'bn3_indices': bn3_indices,
            'orig_lengths': orig_lengths,
            'N_fixed': int(N_fixed),
        }
