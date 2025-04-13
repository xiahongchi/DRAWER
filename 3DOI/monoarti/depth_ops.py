import torch
import numpy as np
import random
import pdb


def recover_metric_depth(pred, gt):
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()
    gt = gt.squeeze()
    pred = pred.squeeze()
    mask = (gt > 1e-8) & (pred > 1e-8)

    gt_mask = gt[mask]
    pred_mask = pred[mask]
    a, b = np.polyfit(pred_mask, gt_mask, deg=1)
    #print("scale {}".format(a))
    #print("shift {}".format(b))
    pred_metric = a * pred + b
    return pred_metric


def recover_scale_shift(pred, gt):
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()
    gt = gt.squeeze()
    pred = pred.squeeze()
    mask = (gt > 1e-8) & (pred > 1e-8)

    gt_mask = gt[mask]
    pred_mask = pred[mask]
    a, b = np.polyfit(pred_mask, gt_mask, deg=1)
    #print("scale {}".format(a))
    #print("shift {}".format(b))
    pred_metric = a * pred + b
    return a, b


def normal_from_depth(depth, focal_length):
    """
    Compute surface normal from depth map
    """
    def normalization(data):
        mo_chang = np.sqrt(
            np.multiply(data[:, :, 0], data[:, :, 0])
            + np.multiply(data[:, :, 1], data[:, :, 1])
            + np.multiply(data[:, :, 2], data[:, :, 2])
        )
        mo_chang = np.dstack((mo_chang, mo_chang, mo_chang))
        return data/mo_chang

    width = 1024
    height = 768
    K = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1],
    ])
    x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x = x.reshape([-1])
    y = y.reshape([-1])
    xyz = np.vstack((x, y, np.ones_like(x)))
    pts_3d = np.dot(np.linalg.inv(K), xyz*depth.reshape([-1]))
    pts_3d_world = pts_3d.reshape((3, height, width))
    f = pts_3d_world[:, 1:height-1, 2:width] - pts_3d_world[:, 1:height-1, 1:width-1]
    t = pts_3d_world[:, 2:height, 1:width-1] - pts_3d_world[:, 1:height-1, 1:width-1]
    normal_map = np.cross(f, t, axisa=0, axisb=0)
    normal_map = normalization(normal_map)

    # pad it to 400x600
    normal_map = np.pad(normal_map, ((1, 1), (1, 1), (0, 0)), 'edge')

    return normal_map


def estimate_equation(points):
    # use linear system
    if points.shape[0] == points.shape[1]:
        normal = np.linalg.solve(points, np.ones(points.shape[0]))
    else:
        normal = np.linalg.lstsq(points, np.ones(points.shape[0]), rcond=None)[0]

    offset = -1 / np.linalg.norm(normal)
    normal /= np.linalg.norm(normal)
    if normal[2] > 0:  # make sure n_z is negative
        normal = -normal
    return normal, offset


def fit_plane(points, thres=0.01, debug=False):
    final_inliers = []
    final_equation = np.array([0, 0, 0])
    final_offset = 0.0

    for i in range(200):
        # Sample 3 points
        sample = np.array([random.choice(points) for _ in range(3)])

        # n_x * x + n_y * y + n_z * z = offset
        try:
            equation, offset = estimate_equation(sample)
        except np.linalg.LinAlgError:  # typically a singular matrix
            continue
        error = points @ equation - offset
        inliers = points[error < thres]

        if debug:
            print('n and inliers: {} {}'.format(equation, len(inliers)))
        if len(inliers) > len(final_inliers):
            final_inliers = inliers
            final_offset = offset
            final_equation = equation

    equation, offset = estimate_equation(final_inliers)

    if debug:
        print('Best Results:')
        print(final_equation)
        print(len(final_inliers))

        print('Final Fit:')
        print(equation)
        print(offset)

    return equation, offset
