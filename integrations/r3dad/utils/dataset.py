import os
import random
from copy import copy
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm.auto import tqdm
import open3d as o3d

from integrations.r3dad.utils.util import (
    af3ad_patch,
    normalize,
    random_patch,
    random_rorate,
    random_translate,
)
from af3ad import PseudoAnomalySynthesizer

all_shapenetad_cates = ['ashtray0', 'bag0', 'bottle0', 'bottle1', 'bottle3', 'bowl0', 'bowl1', 'bowl2', 'bowl3', 'bowl4', 'bowl5', 'bucket0', 'bucket1', 'cap0', 'cap3', 'cap4', 'cap5', 'cup0', 'cup1', 'eraser0', 'headset0', 'headset1', 'helmet0', 'helmet1', 'helmet2', 'helmet3', 'jar0', 'microphone0', 'shelf0', 'tap0', 'tap1', 'vase0', 'vase1', 'vase2', 'vase3', 'vase4', 'vase5', 'vase7', 'vase8', 'vase9']

class ShapeNetAD(Dataset):
    
    def __init__(self, path, cates, split, scale_mode=None, num_points=2048, num_aug=4, transforms=list(), use_patch=False, patch_num=128, patch_scale=0.05, use_af3ad=False):
        super().__init__()
        assert isinstance(cates, list), '`cates` must be a list of cate names.'
        assert split in ('train', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.path = path
        if 'all' in cates:
            self.cates = all_shapenetad_cates
        else:
            self.cates = cates
        self.split = split
        self.scale_mode = scale_mode
        self.num_points = num_points
        self.num_aug = num_aug
        self.transforms = transforms
        self.use_patch = use_patch
        self.patch_num = patch_num
        self.patch_scale = patch_scale
        self.use_af3ad = use_af3ad
        if self.use_af3ad and self.use_patch:
            import warnings
            warnings.warn("Both use_af3ad and use_patch are True; use_af3ad takes precedence.")

        self.pointclouds = []
        self.stats = None

        # self.get_statistics()
        self._af3ad_synth = None
        self.load()

    def get_statistics(self):

        stats_dir = os.path.join(self.path, '../shapenet-ad' + '_stats/')
        os.makedirs(stats_dir, exist_ok=True)

        if len(self.cates) == len(all_shapenetad_cates):
            stats_save_path = os.path.join(stats_dir, 'stats_all.pt')
        else:
            stats_save_path = os.path.join(stats_dir, 'stats_' + '_'.join(self.cates) + '.pt')
        if os.path.exists(stats_save_path):
            self.stats = torch.load(stats_save_path)
            return self.stats

        pointclouds = []
        for cate in self.cates:
            for split in ('train', 'test'):
                local_path = os.path.join(self.path, cate, split)
                for f in os.listdir(local_path):
                    local_file = os.path.join(local_path, f)
                    pcd = o3d.io.read_point_cloud(local_file)
                    pointcloud = np.array(pcd.points, np.float32)
                    choice = np.random.choice(len(pointcloud), self.num_points, False)
                    pointcloud = torch.from_numpy(pointcloud[choice])
                    pointclouds.append(pointcloud)

        all_points = torch.stack(pointclouds, dim=0) # (B, N, 3)
        B, N, _ = all_points.size()
        mean = all_points.view(B*N, -1).mean(dim=0) # (1, 3)
        std = all_points.view(-1).std(dim=0)        # (1, )

        self.stats = {'mean': mean, 'std': std}
        torch.save(self.stats, stats_save_path)
        return self.stats

    def scale(self, pc):
        if self.scale_mode == 'global_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = self.stats['std'].reshape(1, 1)
        elif self.scale_mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif self.scale_mode == 'shape_half':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1) / (0.5)
        elif self.scale_mode == 'shape_34':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1) / (0.75)
        elif self.scale_mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
            pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        else:
            shift = torch.zeros([1, 3])
            scale = torch.ones([1, 1])

        pc = (pc - shift) / scale
        
        return pc, shift, scale

    def append(self, pc, cate, pc_id, mask, label):
        pc, shift, scale = self.scale(pc)
        self.pointclouds.append({
            'pointcloud': pc,
            'cate': cate,
            'id': pc_id,
            'shift': shift,
            'scale': scale,
            'mask': mask,
            'label': label,
        })
    
    def load(self):

        for cate in self.cates:
            if self.split == 'train':
                local_path = os.path.join(self.path, cate, 'train')
                tpls = []
                for f in os.listdir(local_path):
                    local_file = os.path.join(local_path, f)
                    pcd = o3d.io.read_point_cloud(local_file)
                    pointcloud = np.array(pcd.points, dtype=np.float32)
                    tpls.append(pointcloud)
                for pc_id in tqdm(range(self.num_aug), 'Augment'):
                    if self.num_aug == len(tpls):
                        pointcloud = tpls[pc_id]
                    else:
                        pointcloud = random.choice(tpls)
                    pointcloud = random_rorate(pointcloud)
                    choice = np.random.choice(len(pointcloud), self.num_points, False)
                    pointcloud = pointcloud[choice]
                    if self.use_af3ad:
                        if self._af3ad_synth is None:
                            self._af3ad_synth = PseudoAnomalySynthesizer()
                        pointcloud, mask_np = af3ad_patch(pointcloud, self._af3ad_synth)
                        mask = torch.from_numpy(mask_np)
                    elif self.use_patch:
                        pointcloud, mask_np = random_patch(pointcloud, self.patch_num, self.patch_scale)
                        mask = torch.from_numpy(mask_np)
                    else:
                        mask = torch.zeros(self.num_points)
                    pc = torch.from_numpy(pointcloud)
                    label = 0
                    self.append(pc, cate, pc_id, mask, label)

            elif self.split == 'test':
                local_path = os.path.join(self.path, cate, 'test')
                for pc_id, f in enumerate(os.listdir(local_path)):
                    if 'positive' in f:
                        local_file = os.path.join(local_path, f)
                        pcd = o3d.io.read_point_cloud(local_file)
                        pointcloud = np.array(pcd.points, dtype=np.float32)
                        choice = np.random.choice(len(pointcloud), self.num_points, False)
                        pc = torch.from_numpy(pointcloud[choice])
                        mask = torch.zeros(pc.shape[0])
                        label = 0
                    else:
                        local_file = os.path.join(local_path.replace('test', 'GT'), f.replace('pcd', 'txt'))
                        pointcloud_mask = np.genfromtxt(local_file, dtype=np.float32, delimiter=",")
                        choice = np.random.choice(len(pointcloud_mask), self.num_points, False)
                        pc = torch.from_numpy(pointcloud_mask[choice, :3])
                        mask = torch.from_numpy(pointcloud_mask[choice, 3])
                        label = 1
                    self.append(pc, cate, pc_id, mask, label)

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.shuffle(self.pointclouds)

    def save_as_ply(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        for idx, data in enumerate(tqdm(self.pointclouds, desc='Saving PLY')):
            pc = data['pointcloud'].numpy()
            cate = data['cate']
            pc_id = data['id']
            label = data['label']
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            filename = f'{cate}_{self.split}_{idx:04d}_id{pc_id}_label{label}.ply'
            o3d.io.write_point_cloud(os.path.join(save_dir, filename), pcd)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        for transform in self.transforms:
            data = transform(data)
        return data
