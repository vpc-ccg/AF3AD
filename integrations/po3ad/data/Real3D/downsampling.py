import numpy as np
import open3d as o3d


def downsample_random_ratio(points, normals, downsample_ratio=0.5, random_seed=42):
    """Downsample by randomly selecting a subset of points (keeps ratio).

    Args:
        points: (N, 3) numpy array of point coordinates
        normals: (N, 3) numpy array of point normals (can be None)
        downsample_ratio: Ratio of points to keep (0.0-1.0)
        random_seed: Random seed for reproducibility

    Returns:
        downsampled_points: (M, 3) where M = N * downsample_ratio
        downsampled_normals: (M, 3) or None
    """
    rng = np.random.default_rng(random_seed)
    num_points = len(points)
    num_keep = int(num_points * downsample_ratio)
    indices = rng.choice(num_points, size=num_keep, replace=False)
    indices = np.sort(indices)  # Slight cache locality improvement
    return points[indices], normals[indices] if normals is not None else None


def downsample_voxel(points, normals=None, voxel_size=0.002, recompute_normals=True, max_nn=30):
    """Voxel-grid downsample using Open3D. Good for non-uniform scan density.

    Explanation:
        This function reduces the number of points in a point cloud by grouping 
        nearby points into small cubes (voxels) of a specified size.
        For each voxel, it keeps just one representative point, effectively
        "averaging out" dense regions and making the point cloud more
        uniform and manageable.

    Args:
        points: (N, 3) numpy array of point coordinates
        normals: (N, 3) numpy array of point normals (can be None)
        voxel_size: Size of voxel grid cells
        recompute_normals: Whether to recompute normals after downsampling
        max_nn: Maximum nearest neighbors for normal estimation

    Returns:
        downsampled_points: (M, 3)
        downsampled_normals: (M, 3) or None
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if normals is not None and len(normals) == len(points):
        pcd.normals = o3d.utility.Vector3dVector(normals)

    pcd_ds = pcd.voxel_down_sample(voxel_size)

    # Normals: better to recompute after voxel downsampling
    if recompute_normals:
        pcd_ds.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 3.0, max_nn=max_nn
            )
        )
        pcd_ds.normalize_normals()

    pts = np.asarray(pcd_ds.points)
    nrm = np.asarray(pcd_ds.normals) if pcd_ds.has_normals() else None
    return pts, nrm


def fps_indices(points, n_samples, random_seed=42):
    """
    Farthest Point Sampling (FPS) indices.
    Numpy implementation (O(N*n_samples)) — fine after voxel downsample.

    Args:
        points: (N, 3) numpy array of point coordinates
        n_samples: Number of points to sample
        random_seed: Random seed for reproducibility

    Returns:
        indices: (n_samples,) array of selected indices
    """
    rng = np.random.default_rng(random_seed)
    N = points.shape[0]
    if n_samples >= N:
        return np.arange(N, dtype=np.int64)

    idx = np.empty(n_samples, dtype=np.int64)
    idx[0] = rng.integers(0, N)

    dist = np.full(N, np.inf, dtype=np.float64)
    last = points[idx[0]]

    for i in range(1, n_samples):
        d = np.sum((points - last) ** 2, axis=1)
        dist = np.minimum(dist, d)
        idx[i] = int(np.argmax(dist))
        last = points[idx[i]]

    return idx


def downsample_fps(points, normals=None, target_num_points=20000, random_seed=42):
    """Downsample to a fixed number of points via FPS.

    Args:
        points: (N, 3) numpy array of point coordinates
        normals: (N, 3) numpy array of point normals (can be None)
        target_num_points: Target number of points after downsampling
        random_seed: Random seed for reproducibility

    Returns:
        downsampled_points: (target_num_points, 3)
        downsampled_normals: (target_num_points, 3) or None
    """
    idx = fps_indices(points, target_num_points, random_seed=random_seed)
    pts = points[idx]
    nrm = normals[idx] if normals is not None and len(
        normals) == len(points) else None
    return pts, nrm


def estimate_median_nn_distance(points, sample_size=20000, random_seed=42):
    """
    Estimate median nearest-neighbor distance on a random subset.
    Uses Open3D KD-tree for speed.

    Explanation:
        This method quickly estimates how close points are to each other in a large point cloud.
        It randomly picks a subset of points, finds the distance to each points nearest neighbor using a 
        fast KD-tree search, and then returns the median of those distances. This gives you a typical 
        spacing between points, which is useful for setting parameters in algorithms like downsampling 
        or clustering, without having to check every possible pair.

    Args:
        points: (N, 3) numpy array of point coordinates
        sample_size: Number of points to sample for estimation
        random_seed: Random seed for reproducibility

    Returns:
        median_distance: Median nearest-neighbor distance
    """
    rng = np.random.default_rng(random_seed)
    N = points.shape[0]
    if N == 0:
        return 0.0

    m = min(sample_size, N)
    idx = rng.choice(N, size=m, replace=False)
    pts = points[idx]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    dists = np.zeros(len(pts), dtype=np.float64)
    for i, p in enumerate(pts):
        # k=2 returns itself + nearest neighbor
        _, nn_idx, nn_dist2 = kdtree.search_knn_vector_3d(p, 2)
        if len(nn_dist2) >= 2:
            dists[i] = np.sqrt(nn_dist2[1])

    # Filter out zeros (failed searches)
    dists = dists[dists > 0]
    return float(np.median(dists)) if len(dists) else 0.0


def apply_downsampling(
    points,
    normals,
    downsample_mode="none",
    downsample_ratio=0.5,           # for random_ratio mode
    voxel_size=None,                # for voxel / voxel_fps mode
    voxel_size_multiplier=2.0,
    target_num_points=None,         # for fps / voxel_fps
    recompute_normals_after_downsample=True,
    random_seed=42,
):
    """
    Apply downsampling to point cloud with selectable mode.

    Args:
        points: (N, 3) numpy array of point coordinates
        normals: (N, 3) numpy array of point normals (can be None)
        downsample_mode: ['none', 'random_ratio', 'voxel', 'fps', 'voxel_fps']
        downsample_ratio: Ratio for random_ratio mode (0.0-1.0)
        voxel_size: Voxel size for voxel/voxel_fps (None for auto)
        voxel_size_multiplier: Multiplier for auto voxel size
        target_num_points: Target points for fps/voxel_fps
        recompute_normals_after_downsample: Whether to recompute normals
        random_seed: Random seed for reproducibility

    Returns:
        downsampled_points: (M, 3)
        downsampled_normals: (M, 3) or None

    Raises:
        ValueError: If invalid mode or parameters
    """
    mode = (downsample_mode or "none").lower()

    if mode == "none":
        return points, normals

    elif mode == "random_ratio":
        if downsample_ratio is None or not (0.0 < downsample_ratio <= 1.0):
            raise ValueError("downsample_ratio must be in (0, 1].")
        return downsample_random_ratio(
            points, normals, downsample_ratio=downsample_ratio, random_seed=random_seed
        )

    elif mode in ("voxel", "voxel_fps"):
        # 'voxel': Downsample by grouping points into voxels and keeping one per voxel for uniform density.
        # 'voxel_fps': Apply voxel downsampling, then use Farthest Point Sampling for maximal coverage and spread.

        # auto voxel size if not provided
        if voxel_size is None:
            dnn = estimate_median_nn_distance(points, random_seed=random_seed)
            if dnn <= 0:
                raise RuntimeError(
                    "Could not estimate median NN distance for auto voxel size.")
            voxel_size = voxel_size_multiplier * dnn

        points, normals = downsample_voxel(
            points,
            normals=normals,
            voxel_size=float(voxel_size),
            recompute_normals=recompute_normals_after_downsample,
        )

        if mode == "voxel_fps":
            if target_num_points is None:
                raise ValueError(
                    "target_num_points must be set for downsample_mode='voxel_fps'.")
            points, normals = downsample_fps(
                points, normals=normals, target_num_points=int(target_num_points), random_seed=random_seed
            )

        return points, normals

    elif mode == "fps":
        if target_num_points is None:
            raise ValueError(
                "target_num_points must be set for downsample_mode='fps'.")
        points, normals = downsample_fps(
            points, normals=normals, target_num_points=int(target_num_points), random_seed=random_seed
        )

        # optional: recompute normals after fps if requested
        if recompute_normals_after_downsample:
            # Build pcd and recompute normals based on local neighborhoods
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            # Heuristic radius: use median NN on fps points
            dnn = estimate_median_nn_distance(points, random_seed=random_seed)
            radius = max(3.0 * dnn, 1e-6)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius, max_nn=30)
            )
            pcd.normalize_normals()
            normals = np.asarray(pcd.normals)

        return points, normals

    else:
        raise ValueError(f"Unknown downsample_mode='{downsample_mode}'.")
