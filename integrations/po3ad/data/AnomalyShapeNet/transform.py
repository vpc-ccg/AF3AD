import random
import numbers
import scipy
import scipy.ndimage
import scipy.interpolate
import scipy.stats
import numpy as np
import torch
import copy

class NormalizeCoord(object):
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            # modified from pointnet2
            centroid = np.mean(data_dict["coord"], axis=0)
            data_dict["coord"] -= centroid
            m = np.max(np.sqrt(np.sum(data_dict["coord"] ** 2, axis=1)))
            data_dict["coord"] = data_dict["coord"] / m
        return data_dict


class CenterShift(object):
    def __init__(self, apply_z=True):
        self.apply_z = apply_z

    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            x_min, y_min, z_min = data_dict["coord"].min(axis=0)
            x_max, y_max, _ = data_dict["coord"].max(axis=0)
            if self.apply_z:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
            else:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
            data_dict["coord"] -= shift
        return data_dict


class RandomRotate(object):
    def __init__(self, angle=None, center=None, axis="z", always_apply=False, p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


class SphereCropMask(object):
    def __init__(self, part_num=64):
        self.part_num = part_num

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        part_point_num = data_dict["coord"].shape[0] // self.part_num
        centers = []
        for p_i in range(self.part_num):
            null_mask = np.argwhere(data_dict["mask"] == -1)
            center = data_dict["coord"][null_mask[np.random.randint(null_mask.shape[0])]]
            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"][null_mask.reshape(-1)] - center), 1))[:part_point_num]
            data_dict['mask'][null_mask[idx_crop]] = p_i
            centers.append(center)
        data_dict['mask'][data_dict['mask'] == -1] = self.part_num + 1
        return data_dict, centers


class Compose(object):
    def __init__(self, aug_list= []):
        self.transforms = []
        for cur_aug in aug_list:
            self.transforms.append(cur_aug)

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict
    
class RandomPlaneCut:
    """Random plane-based cutting augmentation for point clouds.
    
    Simulates partial scans by cutting the point cloud with a random plane.
    Supports bias towards horizontal cuts (plane normal close to Z axis)
    to match Real3D test set distribution.
    
    Args:
        p: Probability of applying the cut (vs keeping full object).
        r_min: Minimum retention ratio (fraction of points to keep).
        r_max: Maximum retention ratio.
        horizontal_prob: Probability of horizontal cut vs diverse direction.
        horizontal_angle_max: Max angle (degrees) from Z axis for horizontal cuts.
        min_points: Minimum points to retain; if cut results in fewer, skip cut.
        coarse_edge_prob: Probability of applying coarse (rough/jagged) edges.
        coarse_edge_noise: Noise magnitude for coarse edges (relative to object size).
    """
    
    def __init__(
        self,
        p: float = 0.7,
        r_min: float = 0.3,
        r_max: float = 0.9,
        horizontal_prob: float = 0.7,
        horizontal_angle_max: float = 20.0,
        min_points: int = 1024,
        coarse_edge_prob: float = 0.3,
        coarse_edge_noise: float = 0.05,
    ):
        self.p = p
        self.r_min = r_min
        self.r_max = r_max
        self.horizontal_prob = horizontal_prob
        self.horizontal_angle_max = horizontal_angle_max
        self.min_points = min_points
        self.coarse_edge_prob = coarse_edge_prob
        self.coarse_edge_noise = coarse_edge_noise
    
    def __call__(self, data_dict):
        if np.random.random() > self.p:
            return data_dict
        
        if "coord" not in data_dict:
            return data_dict
        
        coord = data_dict["coord"]
        n_points = len(coord)
        
        if n_points < self.min_points:
            return data_dict
        
        # Sample cut plane normal
        if np.random.random() < self.horizontal_prob:
            # Horizontal cut: normal close to Z axis
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.radians(self.horizontal_angle_max))
            normal = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
        else:
            # Diverse cut: uniform on sphere (Marsaglia method)
            while True:
                u, v = np.random.uniform(-1, 1, size=2)
                s_sq = u * u + v * v
                if s_sq < 1:
                    break
            sqrt_term = np.sqrt(1 - s_sq)
            normal = np.array([2 * u * sqrt_term, 2 * v * sqrt_term, 1 - 2 * s_sq])
        
        # Flip normal randomly to avoid directional bias
        if np.random.random() < 0.5:
            normal = -normal
        
        # Project points onto normal
        projections = coord @ normal
        
        # Sample retention ratio and compute threshold
        target_ratio = np.random.uniform(self.r_min, self.r_max)
        sorted_proj = np.sort(projections)
        cutoff_idx = int((1 - target_ratio) * n_points)
        cutoff_idx = max(0, min(cutoff_idx, n_points - 1))
        threshold = sorted_proj[cutoff_idx]
        
        # # Apply coarse edge noise (irregular/jagged boundary)
        # if self.coarse_edge_prob > 0 and np.random.random() < self.coarse_edge_prob:
        #     # Compute object scale for noise magnitude
        #     proj_range = sorted_proj[-1] - sorted_proj[0]
        #     noise_scale = proj_range * self.coarse_edge_noise
        #     # Add per-point noise to projections for irregular boundary
        #     noise = np.random.normal(0, noise_scale, n_points)
        #     projections = projections + noise
        
        # Create mask
        mask = projections >= threshold
        n_retained = np.sum(mask)
        
        # Skip if too few points retained
        if n_retained < self.min_points:
            return data_dict
        
        # Apply mask to all point-wise arrays
        data_dict["coord"] = coord[mask]
        if "normal" in data_dict:
            data_dict["normal"] = data_dict["normal"][mask]
        if "color" in data_dict:
            data_dict["color"] = data_dict["color"][mask]
        if "mask" in data_dict and len(data_dict["mask"]) == n_points:
            data_dict["mask"] = data_dict["mask"][mask]
        
        return data_dict


class RandomEdgeSegmentCutout:
    """Random cutout augmentation that removes edge segments from point clouds.
    
    This augmentation is similar to cutout in images. It identifies segments
    that are on the edges of the point cloud (using KNN distance to other segment
    centers) and randomly removes up to max_segments of them. This helps make
    training data more similar to test data which may have missing edge regions.
    
    The edge detection works by:
    1. Using the segment centers from SphereCropMask segmentation
    2. Computing the mean KNN distance for each segment center
    3. Segments with high KNN distances (above the specified percentile threshold) 
       are considered edges
    
    This augmentation should be applied AFTER SphereCropMask since it uses
    the segment centers produced by that transform.
    
    Args:
        p: Probability of applying the cutout augmentation.
        max_segments: Maximum number of edge segments to remove (1 to max_segments).
        k_neighbors: Number of neighbors to use for KNN distance computation.
        edge_threshold_percentile: Percentile threshold for identifying edge segments.
            Segments with mean KNN distance above this percentile are considered edges.
        min_points: Minimum points to retain; if cutout results in fewer, skip cutout.
    """
    
    def __init__(
        self,
        p: float = 0.5,
        max_segments: int = 3,
        k_neighbors: int = 5,
        edge_threshold_percentile: float = 70.0,
        min_points: int = 1024,
    ):
        self.p = p
        self.max_segments = max_segments
        self.k_neighbors = k_neighbors
        self.edge_threshold_percentile = edge_threshold_percentile
        self.min_points = min_points
    
    def _compute_knn_distances(self, centers: np.ndarray, k: int) -> np.ndarray:
        """Compute mean KNN distance for each segment center.
        
        Args:
            centers: Array of shape (N, 3) containing segment centers.
            k: Number of nearest neighbors to consider.
            
        Returns:
            Array of shape (N,) containing mean KNN distance for each center.
        """
        n_centers = len(centers)
        if n_centers <= k:
            k = max(1, n_centers - 1)
        
        # Compute pairwise distances between all centers
        # dist_matrix[i, j] = distance between center i and center j
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (N, N, 3)
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)
        
        # For each center, find mean distance to k nearest neighbors (excluding self)
        mean_knn_distances = np.zeros(n_centers)
        for i in range(n_centers):
            # Get distances from center i to all others
            distances = dist_matrix[i]
            # Sort and take k+1 smallest (first one is self with distance 0)
            sorted_distances = np.sort(distances)
            # Take distances 1 to k+1 (skip self)
            knn_distances = sorted_distances[1:k+1]
            mean_knn_distances[i] = np.mean(knn_distances)
        
        return mean_knn_distances
    
    def _find_edge_segments(self, centers: np.ndarray) -> np.ndarray:
        """Find segment indices that are on the edges of the point cloud.
        
        Edge segments are identified as those with high mean KNN distance
        (above the specified percentile threshold).
        
        Args:
            centers: Array of shape (N, 3) containing segment centers.
            
        Returns:
            Array of segment indices that are considered edges.
        """
        if len(centers) < 2:
            return np.array([], dtype=np.int32)
        
        # Compute mean KNN distances for all centers
        mean_knn_distances = self._compute_knn_distances(centers, self.k_neighbors)
        
        # Find threshold based on percentile
        threshold = np.percentile(mean_knn_distances, self.edge_threshold_percentile)
        
        # Segments with distance above threshold are edges
        edge_indices = np.where(mean_knn_distances >= threshold)[0]
        
        return edge_indices.astype(np.int32)
    
    def __call__(self, data_dict, centers):
        """Apply random edge segment cutout.
        
        Args:
            data_dict: Dictionary containing 'coord', 'normal', and 'mask' arrays.
            centers: List of segment centers from SphereCropMask.
            
        Returns:
            Tuple of (modified data_dict, modified centers).
        """
        if np.random.random() > self.p:
            return data_dict, centers
        
        if "coord" not in data_dict or "mask" not in data_dict:
            return data_dict, centers
        
        coord = data_dict["coord"]
        mask = data_dict["mask"]
        n_points = len(coord)
        
        if n_points < self.min_points:
            return data_dict, centers
        
        # Convert centers list to numpy array
        centers_array = np.array([c.flatten() for c in centers])  # (num_segments, 3)
        
        if len(centers_array) < 2:
            return data_dict, centers
        
        # Find edge segments
        edge_indices = self._find_edge_segments(centers_array)
        
        if len(edge_indices) == 0:
            return data_dict, centers
        
        # Randomly select number of segments to remove (1 to max_segments)
        num_to_remove = np.random.randint(1, min(self.max_segments, len(edge_indices)) + 1)
        
        # Randomly select which edge segments to remove
        segments_to_remove = np.random.choice(edge_indices, num_to_remove, replace=False)
        
        # Create mask to keep points not in removed segments
        keep_mask = np.ones(n_points, dtype=bool)
        for seg_idx in segments_to_remove:
            keep_mask[mask == seg_idx] = False
        
        n_retained = np.sum(keep_mask)
        
        # Skip if too few points retained
        if n_retained < self.min_points:
            return data_dict, centers
        
        # Apply mask to all point-wise arrays
        data_dict["coord"] = coord[keep_mask]
        if "normal" in data_dict:
            data_dict["normal"] = data_dict["normal"][keep_mask]
        if "color" in data_dict:
            data_dict["color"] = data_dict["color"][keep_mask]
        
        # Update mask values - remap to consecutive indices
        old_mask = mask[keep_mask]
        
        # Get unique remaining segment indices in sorted order
        # Note: SphereCropMask assigns part_num+1 to leftover points (e.g., 65 when part_num=64)
        # We need to filter those out when selecting valid segment indices for centers
        all_unique = set(old_mask)
        valid_segment_indices = sorted(all_unique - set(segments_to_remove))
        
        # Separate valid segment indices (0 to len(centers)-1) from overflow indices
        max_valid_idx = len(centers) - 1
        remaining_valid_segments = [int(idx) for idx in valid_segment_indices if idx <= max_valid_idx]
        overflow_indices = [int(idx) for idx in valid_segment_indices if idx > max_valid_idx]
        
        # Create a mapping from old indices to new consecutive indices
        # Valid segments: old_idx -> new_idx (0, 1, 2, ...)
        # Overflow indices (like 65): map to a new overflow value at the end
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(remaining_valid_segments)}
        
        # Map overflow indices to values after the valid segment range
        overflow_new_idx = len(remaining_valid_segments)
        for overflow_idx in overflow_indices:
            old_to_new[overflow_idx] = overflow_new_idx
            overflow_new_idx += 1
        
        # Remap mask values to consecutive indices
        new_mask = np.array([old_to_new.get(int(m), int(m)) for m in old_mask], dtype=old_mask.dtype)
        data_dict["mask"] = new_mask
        
        # Update centers list - keep only non-removed centers in order (use integer indices)
        new_centers = [centers[i] for i in remaining_valid_segments]
        
        return data_dict, new_centers