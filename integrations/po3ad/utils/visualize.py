import os
import numpy as np
import plotly.offline as po
import plotly.graph_objects as go
import open3d as o3d
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import matplotlib.pyplot as plt


def save_pc_plotly_html(points_xyz, scores, out_html):
    # points_xyz: (N,3) numpy
    # scores: (N,) numpy, any scale
    s = scores.astype(float)
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)  # [0,1]

    # map to RGB (lightgray->yellow->red)
    # lightgray (0.83), yellow, red
    # simple lerp piecewise: 0-0.5 go gray->yellow, 0.5-1 yellow->red
    def lerp(a, b, t): return a*(1-t)+b*t
    N = points_xyz.shape[0]
    rgb = np.zeros((N, 3))
    for i, v in enumerate(s):
        if v < 0.5:
            t = v/0.5
            rgb[i] = [lerp(0.83, 1.0, t), lerp(0.83, 1.0, t),
                      lerp(0.83, 0.0, t)]  # gray→yellow
        else:
            t = (v-0.5)/0.5
            rgb[i] = [1.0, lerp(1.0, 0.0, t), 0.0]  # yellow→red

    fig = go.Figure(data=[go.Scatter3d(
        x=points_xyz[:, 0], y=points_xyz[:, 1], z=points_xyz[:, 2],
        mode='markers',
        marker=dict(size=2, opacity=0.9, color=['rgb({},{},{})'.format(
            int(255*r), int(255*g), int(255*b)) for r, g, b in rgb])
    )])
    fig.update_layout(scene=dict(aspectmode="data"),
                      margin=dict(l=0, r=0, t=0, b=0))
    po.plot(fig, filename=out_html, auto_open=False)


def save_anomalies_visualization_html(batch, pred_mask, gt_mask=None, output_dir="visualizations", file_prefix="sample", counter=0):

    os.makedirs(output_dir, exist_ok=True)
    points = batch['xyz_original'].numpy()
    tag = f"{file_prefix}_{counter}"

    save_pc_plotly_html(points, pred_mask, os.path.join(
        output_dir, f"{tag}_predicted.html"))
    if gt_mask is not None:
        save_pc_plotly_html(points, gt_mask.astype(float), os.path.join(
            output_dir, f"{tag}_ground_truth.html"))


def save_anomalies_visualization(
    batch,
    pred_mask,
    gt_mask=None,
    output_dir="visualizations",
    file_prefix="sample",
    counter=0,
):
    """
    Save point cloud anomalies visualization to a file.

    Args:
        batch (dict): Batch data containing the original point cloud.
        pred_mask (numpy.ndarray): Predicted anomaly scores for each point.
        gt_mask (numpy.ndarray, optional): Ground truth anomaly mask for each point.
        output_dir (str): Directory to save the visualizations.
        file_prefix (str): Prefix for the output file names.
    """
    os.makedirs(output_dir, exist_ok=True)

    points = batch['xyz_original'].numpy()

    # Normalize prediction
    eps = 1e-8
    pred_norm = (pred_mask - pred_mask.min()) / \
        (pred_mask.max() - pred_mask.min() + eps)

    def blue_yellow_red_colormap(v):
        colors = np.zeros((len(v), 3), dtype=np.float32)

        # Blue -> Yellow
        mask_low = v < 0.5
        t = v[mask_low] / 0.5
        colors[mask_low, 0] = t
        colors[mask_low, 1] = t
        colors[mask_low, 2] = 1.0 - t

        # Yellow -> Red
        mask_high = v >= 0.5
        t = (v[mask_high] - 0.5) / 0.5
        colors[mask_high, 0] = 1.0
        colors[mask_high, 1] = 1.0 - t
        colors[mask_high, 2] = 0.0

        return colors

    # Predicted visualization
    pred_colors = blue_yellow_red_colormap(pred_norm)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(pred_colors)

    file_prefix = f"{file_prefix}_{counter}"
    pred_path = os.path.join(output_dir, f"{file_prefix}_predicted.ply")
    o3d.io.write_point_cloud(pred_path, pcd)

    # Ground truth visualization (if provided)
    if gt_mask is not None:
        gt_norm = gt_mask.astype(np.float32)
        gt_colors = blue_yellow_red_colormap(gt_norm)

        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(points)
        gt_pcd.colors = o3d.utility.Vector3dVector(gt_colors)

        gt_path = os.path.join(output_dir, f"{file_prefix}_ground_truth.ply")
        o3d.io.write_point_cloud(gt_path, gt_pcd)
