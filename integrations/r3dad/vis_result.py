import argparse
import os
import numpy as np
import cv2
import torch

import open3d as o3d

cord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

def main(args):
    all_ref = np.load(os.path.join(args.work_path, 'ref.npy'))
    all_recons = np.load(os.path.join(args.work_path, 'out.npy'))
    all_masks = np.load(os.path.join(args.work_path, 'mask.npy'))

    for ref, recon, mask in zip(all_ref, all_recons, all_masks):

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(ref)
        source.paint_uniform_color([1, 0.706, 0])
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(recon)
        target.paint_uniform_color([0, 0.651, 0.929])
        gt = o3d.geometry.PointCloud()
        gt.points = o3d.utility.Vector3dVector(ref[mask==1])
        gt.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([source, target, gt])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work_path")
    args = parser.parse_args()

    main(args)