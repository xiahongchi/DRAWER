import cv2
import numpy as np
import torch
import torch.nn.functional as F


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def compute_kl_divergence(src_aff, tgt_aff):
    """
    Compute kl divergence of two affordance map.
    See https://github.com/Tushar-N/interaction-hotspots/blob/master/utils/evaluation.py
    """
    eps = 1e-12

    # normalize affordance map so that it sums to 1
    src_aff_norm = src_aff / (src_aff.sum(dim=-1).sum(dim=-1).unsqueeze(-1).unsqueeze(-1) + eps)
    tgt_aff_norm = tgt_aff / (tgt_aff.sum(dim=-1).sum(dim=-1).unsqueeze(-1).unsqueeze(-1) + eps)

    kld = F.kl_div(src_aff_norm.log(), tgt_aff_norm, reduction='none')
    kld = kld.sum(dim=-1).sum(dim=-1)

    # sometimes kld is inf
    kld = kld[~torch.isinf(kld)]

    return kld

def compute_sim(src_aff, tgt_aff):
    """
    Compute histogram intersection of two affordance map.
    See https://github.com/Tushar-N/interaction-hotspots/blob/master/utils/evaluation.py
    """
    eps = 1e-12

    # normalize affordance map so that it sums to 1
    src_aff_norm = src_aff / (src_aff.sum(dim=-1).sum(dim=-1).unsqueeze(-1).unsqueeze(-1) + eps)
    tgt_aff_norm = tgt_aff / (tgt_aff.sum(dim=-1).sum(dim=-1).unsqueeze(-1).unsqueeze(-1) + eps)

    intersection = torch.minimum(src_aff_norm, tgt_aff_norm)
    intersection = intersection.sum(dim=-1).sum(dim=-1)

    return intersection