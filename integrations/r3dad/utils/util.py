import random
import numpy as np
import open3d as o3d
from itertools import repeat

from af3ad import PseudoAnomalySynthesizer

def random_rorate(pc):
    degree = np.random.uniform(-180, 180, 3)
    matrix = o3d.geometry.get_rotation_matrix_from_xyz(np.pi * degree / 180.0)
    pc_aug = np.matmul(pc, matrix)
    return pc_aug.astype(np.float32)

def split_pointcloud(pc, matrix):
    R = matrix[:3, :3]
    T = matrix[:3, 3]
    
    # The plane's normal vector is the z-axis of the transformed coordinate system
    normal_vector = R[:, 2]
    point_on_plane = T
    
    pc1 = []
    pc2 = []
    
    for p in pc:
        distance = np.dot(normal_vector, p - point_on_plane)
        if distance > 0:
            pc1.append(p)
        else:
            pc2.append(p)
    
    pc1 = np.array(pc1, np.float32)
    pc2 = np.array(pc2, np.float32)
    
    return pc1, pc2

def rotate_vector(vector, axis, angle_degrees):
    """
    Rotate a vector around a given axis by a specified angle.
    """
    angle_radians = np.radians(angle_degrees)
    axis = axis / np.linalg.norm(axis)
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    cross_matrix = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    rotation_matrix = cos_angle * np.eye(3) + sin_angle * cross_matrix + (1 - cos_angle) * np.outer(axis, axis)
    return np.dot(rotation_matrix, vector)

def trans_and_rotate_plane(matrix, distance, degree1, degree2):
    normal_vector = matrix[:3, 2]
    point_on_plane = matrix[:3, 3]
    
    # Move the plane along its normal direction
    point_on_plane += normal_vector * distance
    
    # Identify two orthogonal axes for rotation
    # For simplicity, assuming these axes are the x and y axes of the local plane coordinate system
    axis1 = matrix[:3, 0]  # Local x-axis
    axis2 = matrix[:3, 1]  # Local y-axis
    
    normal_vector = rotate_vector(normal_vector, axis1, degree1)
    normal_vector = rotate_vector(normal_vector, axis2, degree2)
    
    new_matrix = np.eye(4)
    new_matrix[:3, 2] = normal_vector  # Update the normal vector
    new_matrix[:3, 3] = point_on_plane  # Update the point on plane
    
    return new_matrix

def random_split(pc, matrix, distance=0.5, degree=5):
    distance = np.random.uniform(-distance, distance)
    degree1 = np.random.uniform(-degree, degree)
    degree2 = np.random.uniform(-degree, degree)
    new_matrix = trans_and_rotate_plane(matrix, distance, degree1, degree2)
    pc1, pc2 = split_pointcloud(pc, new_matrix)
    return pc1, pc2

def distance_square(p1, p2):
    tensor = p1 - p2
    val = tensor.mul(tensor).sum()
    return val

def normalize(pc):
    pc -= np.mean(pc, axis=0)
    pc /= np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    return pc.astype(np.float32)

def normalize_vector(vector):
    length = np.linalg.norm(vector)
    unit_vector = vector / length
    return unit_vector

def random_patch(points, patch_num, scale):
    patchP = np.zeros((patch_num, 3), dtype=np.float32)

    view = np.array([[2, -2, -2], [2, -2, 0], [2, -2, 2], [0, -2, -2], [0, -2, 0], [0, -2, 2],
                     [-2, -2, -2], [-2, -2, 0], [-2, -2, 2], [2, 0, -2], [2, 0, 0], [2, 0, 2],
                     [0, 0, -2], [0, 0, 2], [-2, 0, -2], [-2, 0, 0], [-2, 0, 2], [2, 2, -2],
                     [2, 2, 0], [2, 2, 2], [0, 2, -2], [0, 2, 0], [0, 2, 2], [-2, 2, -2],
                     [-2, 2, 0], [-2, 2, 2]], dtype=np.float32)
    
    index = view[random.randint(0, view.shape[0] - 1)][None, :]
    
    # Calculate distances from points to the selected view point
    distances = np.sum((points - index) ** 2, axis=1)
    distance_order = distances.argsort()
    
    for sp in range(patch_num):
        patchP[sp] = points[distance_order[sp]]
    
    # Calculate normal and apply transformations
    convex_normal = patchP-index
    convex_normal = np.apply_along_axis(normalize_vector, 1, convex_normal)
    convex_normal *= scale if random.random() < 0.5 else -scale

    # Generate a random distance for each point that follows a Gaussian distribution
    random_translation_distances =np.random.rand(patch_num)
    sort_random_translation_distances = np.sort(random_translation_distances)[::-1]

    # Generate random translation vector
    random_translation_vectors = convex_normal * sort_random_translation_distances[:, None]
    patchP += random_translation_vectors
    
    # Update original points with the transformed patch
    new_points = points.copy()
    new_points[distance_order[:patch_num]] = patchP
    
    mask = np.zeros(points.shape[0], np.float32)
    mask[distance_order[:patch_num]] = 1
    
    return new_points, mask

def af3ad_patch(points, synthesizer=None):
    """Generate pseudo anomaly using the AF3AD synthesizer.

    Returns (new_points, mask) with the same contract as ``random_patch``.
    """
    if synthesizer is None:
        synthesizer = PseudoAnomalySynthesizer()

    # Estimate normals via Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    normals = np.asarray(pcd.normals, dtype=np.float32)

    # Random anomaly centre
    center = points[np.random.randint(len(points))]

    # Random preset configuration
    presets = synthesizer.preset_factory.presets
    cfg = presets[np.random.randint(len(presets))]()

    new_points = synthesizer.generate(points, normals, center, cfg).astype(np.float32)

    # Binary mask: 1 where a point was displaced
    displacement = np.linalg.norm(new_points - points, axis=1)
    mask = (displacement > 1e-6).astype(np.float32)

    return new_points, mask


def random_translate(points, radius=0.08, translate=0.02, part=16):
    mask = np.zeros(points.shape[0], np.float32)

    for _ in range(part):
        random_point = points[random.randint(0, points.shape[0] - 1)]
        distances = np.linalg.norm(points - random_point, axis=1)
        fg = distances < radius

        point_fg = points[fg]
        n, dim= point_fg.shape
        t = list(repeat(translate, times=dim))

        ts = []
        for d in range(dim):
            ts.append(np.random.uniform(-abs(t[d]), abs(t[d]), n))
        
        points[fg] += np.stack(ts, axis=-1)
        mask[fg] = 1

    return points, mask
