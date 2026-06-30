"""
From https://github.com/stevenygd/PointFlow/tree/master/metrics
"""
import warnings
import random
from tqdm.auto import tqdm
import numpy as np
from numpy.linalg import norm
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
import torch

from pointnet2_ops import pointnet2_utils
from .patchcore import NearestNeighbourScorer

def gps(xyz, num_group, group_size):
    '''
    input: B N 3
    ---------------------------
    neighborhood: B G M 3
    center : B G 3
    '''
    batch_size, num_points, _ = xyz.shape
    # fps the centers out
    fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group) 
    center = pointnet2_utils.gather_operation(xyz.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    # knn to get the neighborhood
    dist = torch.cdist(center, xyz)  # B G N
    _, idx = torch.topk(dist, k=group_size, largest=False, dim=2)  # B G M
    assert idx.size(1) == num_group
    assert idx.size(2) == group_size
    idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
    neighborhood = neighborhood.view(batch_size, num_group, group_size, 3).contiguous()
    # normalize
    neighborhood = neighborhood - center.unsqueeze(2)
    return neighborhood, center

def nn(xyz, k):
    
    # xyz (N, 3)
    N = xyz.shape[0]
    dist = torch.cdist(xyz.unsqueeze(0), xyz.unsqueeze(0)).squeeze(0)  # N, N
    _, idx = torch.topk(dist, k=k, largest=False, dim=1)  # N, k
    idx = idx.view(-1)
    neighborhood = xyz.view(N, -1)[idx, :]
    neighborhood = neighborhood.view(N, k, -1)
    neighborhood = neighborhood.view(N, -1).contiguous()

    return neighborhood

def ROC_AP(all_refs, all_recons, all_labels, all_masks):
    image_preds = []
    pixel_preds = []

    image_preds_nn = []
    pixel_preds_nn = []
    anomaly_scorer = NearestNeighbourScorer(n_nearest_neighbours=1)
    
    for patch, patch_lib in tqdm(zip(all_refs, all_recons), 'Calculate'):
        
        dist = torch.cdist(patch, patch_lib)
        min_val, min_idx = torch.min(dist, dim=1)
        s_idx = torch.argmax(min_val)
        s_star = torch.max(min_val)

        # reweighting
        m_test = patch[s_idx].unsqueeze(0)  # anomalous patch
        m_star = patch_lib[min_idx[s_idx]].unsqueeze(0)  # closest neighbour
        w_dist = torch.cdist(m_star, patch_lib)  # find knn to m_star pt.1
        _, nn_idx = torch.topk(w_dist, k=3, largest=False)  # pt.2
        # equation 7 from the paper
        m_star_knn = torch.linalg.norm(m_test - patch_lib[nn_idx[0, 1:]], dim=1)
        # Softmax normalization trick as in transformers.
        # As the patch vectors grow larger, their norm might differ a lot.
        # exp(norm) can give infinities.
        D = torch.sqrt(torch.tensor(patch.shape[1]))
        w = 1 - (torch.exp(s_star / D) / (torch.sum(torch.exp(m_star_knn / D))))
        s = w * s_star

        # segmentation map
        s_map = min_val

        image_preds.append(s.cpu().numpy())
        pixel_preds.append(s_map.cpu().numpy())
        
        patch_nn = nn(patch, 64)
        patch_lib_nn = nn(patch_lib, 64)
        anomaly_scorer.fit([patch_lib_nn.cpu().numpy()])
        patch_scores = anomaly_scorer.predict([patch_nn.cpu().numpy()])[0]

        image_preds_nn.append(np.max(patch_scores))
        pixel_preds_nn.append(patch_scores)


    image_preds = np.array(image_preds).flatten()
    image_preds = (image_preds - np.min(image_preds)) / (np.max(image_preds) - np.min(image_preds))
    pixel_preds = np.array(pixel_preds).flatten()
    pixel_preds = (pixel_preds - np.min(pixel_preds)) / (np.max(pixel_preds) - np.min(pixel_preds))
    
    image_labels = all_labels.cpu().numpy()
    pixel_labels = all_masks.flatten().cpu().numpy()

    image_rocauc = roc_auc_score(image_labels, image_preds)
    pixel_rocauc = roc_auc_score(pixel_labels, pixel_preds)
    image_aupr = average_precision_score(image_labels, image_preds)
    pixel_aupr = average_precision_score(pixel_labels, pixel_preds)

    image_preds_nn = np.array(image_preds_nn).flatten()
    image_preds_nn = (image_preds_nn - np.min(image_preds_nn)) / (np.max(image_preds_nn) - np.min(image_preds_nn))
    pixel_preds_nn = np.array(pixel_preds_nn).flatten()
    pixel_preds_nn = (pixel_preds_nn - np.min(pixel_preds_nn)) / (np.max(pixel_preds_nn) - np.min(pixel_preds_nn))

    image_rocauc_nn = roc_auc_score(image_labels, image_preds_nn)
    pixel_rocauc_nn = roc_auc_score(pixel_labels, pixel_preds_nn)
    image_aupr_nn = average_precision_score(image_labels, image_preds_nn)
    pixel_aupr_nn = average_precision_score(pixel_labels, pixel_preds_nn)

    results = {
        'ROC_i': image_rocauc,
        'ROC_p': pixel_rocauc,
        'AP_i': image_aupr,
        'AP_p': pixel_aupr,
        'ROC_i_nn': image_rocauc_nn,
        'ROC_p_nn': pixel_rocauc_nn,
        'AP_i_nn': image_aupr_nn,
        'AP_p_nn': pixel_aupr_nn,
    }
    return results

    