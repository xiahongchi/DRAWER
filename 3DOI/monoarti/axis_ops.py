import torch
import numpy as np
import pdb


def ea_score(src_line, tgt_line):
    """
    Implement a differentiable EAScore of two 2D lines.

    Kai Zhao∗, Qi Han∗, Chang-Bin Zhang, Jun Xu, Ming-Ming Cheng. 
    Deep Hough Transform for Semantic Line Detection.
    TPAMI 2021.

    - src_line: tensor shape Nx4, XYXY format
    - tgt_line: tensor shape Nx4, XYXY format
    """
    
    # midpoint error
    src_line_mid_x = (src_line[:, 0] + src_line[:, 2]) / 2
    tgt_line_mid_x = (tgt_line[:, 0] + tgt_line[:, 2]) / 2
    src_line_mid_y = (src_line[:, 1] + src_line[:, 3]) / 2
    tgt_line_mid_y = (tgt_line[:, 1] + tgt_line[:, 3]) / 2
    line_se = 1 - torch.sqrt((src_line_mid_x - tgt_line_mid_x)**2 + (src_line_mid_y - tgt_line_mid_y)**2)
    line_se = line_se.clamp(min=0)

    # angle error
    src_line_angle = torch.atan((src_line[:, 1] - src_line[:, 3]) / (src_line[:, 0] - src_line[:, 2] + 1e-5))
    tgt_line_angle = torch.atan((tgt_line[:, 1] - tgt_line[:, 3]) / (tgt_line[:, 0] - tgt_line[:, 2] + 1e-5))
    d_angle = torch.abs(src_line_angle - tgt_line_angle)
    d_angle = torch.min(d_angle, torch.pi - d_angle)
    line_sa = 1 - d_angle / (torch.pi / 2)
    
    line_eascore = (line_se * line_sa) ** 2

    line_eascore[torch.isnan(line_eascore)] = 0.0

    return line_eascore, line_se, line_sa


def sine_to_angle(sin, cos, r, eps=1e-5, debug=False, scale=False):
    sin = sin + eps
    cos = cos + eps
    mag = torch.sqrt(sin ** 2 + cos ** 2)
    sin = sin / mag
    cos = cos / mag
    r = r / mag

    theta_cos = torch.atan2(sin, cos)
    if scale:
        # compress range from [-pi, pi] to [0, pi]
        # as theta and theta + pi represent the same line but r needs
        # to be inverted.
        r[theta_cos < 0] = - r[theta_cos < 0]
        theta_cos[theta_cos < 0] += torch.pi
    else:
        # change range from [-pi, pi] to [0, 2pi]
        theta_cos[theta_cos < 0] += torch.pi * 2

    # theta_cos = torch.acos(cos)
    # theta_sin = torch.asin(sin)

    # theta_cos[theta_sin < 0] = torch.pi * 2 - theta_cos[theta_sin < 0]
    # r[theta_sin < 0] = - r[theta_sin < 0]

    # if not debug:
    #     theta_cos[theta_sin < 0] = torch.pi * 2 - theta_cos[theta_sin < 0]
    # else:
    #     theta_cos = torch.atan(sin / cos)

    return theta_cos, r


def line_xyxy_to_angle(line_xyxy, center=[0.5, 0.5], debug=False):
    """
    Convert [X1, Y1, X2, Y2] representation of a 2D line to 
    [sin(theta), cos(theta), offset] representation.

    r = xcos(theta) + ysin(theta)

    For two points (x1, y1) and (x2, y2) within image plane [0, 1],
    - cos(theta) = y1 - y2
    - sin(theta) = x2 - x1
    - r = x2y1 - x1y2

    Shengyi Qian, Linyi Jin, Chris Rockwell, Siyi Chen, David Fouhey.
    Understanding 3D Object Articulation in Internet Videos.
    CVPR 2022.
    """
    eps = 1e-5
    device = line_xyxy.device

    if isinstance(center, list):
        center_w, center_h = center
        line_xyxy = line_xyxy - torch.as_tensor([center_w, center_h, center_w, center_h]).to(device)
    elif isinstance(center, torch.Tensor):
        line_xyxy = line_xyxy - center
    else:
        raise NotImplementedError

    #line_xyxy = line_xyxy.clamp(min=-0.5, max=0.5)
    line_xyxy = line_xyxy.clamp(min=-1.0, max=1.0)
    x1, y1, x2, y2 = line_xyxy[:,:1], line_xyxy[:,1:2], line_xyxy[:,2:3], line_xyxy[:,3:4]

    cos = y1 - y2
    sin = x2 - x1
    r = x2 * y1 - x1 * y2

    theta, r = sine_to_angle(sin, cos, r, debug=debug, scale=True)

    # if debug:
    #     pdb.set_trace()
    #     pass

    # normalize, and ensure
    # sin(theta) in [0, 1]
    # cos(theta) in [-1, 1]
    # r in (-sqrt(2) / 2, sqrt(2) / 2)
    theta *= 2
    sin = torch.sin(theta)
    cos = torch.cos(theta)
    
    # assert (r > (- np.sqrt(2) / 2)).all()
    # assert (r < (np.sqrt(2) / 2)).all()
    assert (r > (- np.sqrt(2))).all()
    assert (r < (np.sqrt(2))).all()
    assert (sin >= -1).all()
    assert (sin <= 1).all()
    assert (cos >= -1).all()
    assert (cos <= 1).all()

    return torch.cat((sin, cos, r), dim=1)


def line_angle_to_xyxy(line_angle, center=[0.5, 0.5], use_bins=False, debug=False):
    """
    Convert [sin(theta), cos(theta), offset] representation of a 2D line to 
    [X1, Y1, X2, Y2] representation.

    Shengyi Qian, Linyi Jin, Chris Rockwell, Siyi Chen, David Fouhey.
    Understanding 3D Object Articulation in Internet Videos.
    CVPR 2022.
    """
    eps = 1e-5
    device = line_angle.device

    if isinstance(center, list):
        center_w, center_h = center
    elif isinstance(center, torch.Tensor):
        center_w = center[:, 0:1]
        center_h = center[:, 1:2]
    else:
        raise NotImplementedError

    sin = line_angle[:, :1]
    cos = line_angle[:, 1:2]
    r = line_angle[:, 2:]

    # normalize sin and cos
    # make sure r is not out of boundary
    theta, r = sine_to_angle(sin, cos, r, debug=debug)
    theta /= 2
    sin = torch.sin(theta)
    cos = torch.cos(theta)
    r = r.clamp(min=(- np.sqrt(2) / 2), max=(np.sqrt(2) / 2))

    # intersect line with four boundaries
    y1 = (r - cos * (0.0 - center_w)) / sin + center_h
    y2 = (r - cos * (1.0 - center_w)) / sin + center_h
    x3 = (r - sin * (0.0 - center_h)) / cos + center_w
    x4 = (r - sin * (1.0 - center_h)) / cos + center_w

    

    line_xyxy = []
    
    for i in range(line_angle.shape[0]):
        line = []
        if y1[i] > - eps and y1[i] < (1.0 + eps):
            line.append([0.0, y1[i]])

        if y2[i] > - eps and y2[i] < (1.0 + eps):
            line.append([1.0, y2[i]])

        if len(line) < 2 and x3[i] > - eps and x3[i] < (1.0 + eps):
            line.append([x3[i], 0.0])

        if len(line) < 2 and x4[i] > - eps and x4[i] < (1.0 + eps):
            line.append([x4[i], 1.0])

        # Mathematically, we should only have two boundary points.
        # However, in training time, it is not guaranteed the represented
        # line is within the image plane. Even if r < sqrt(2)/2, it 
        # can be out of boundary. But it's rare and we want to ignore it.
        if len(line) != 2:
            line = [[0.0, y1[i]], [x3[i], 0.0]]

        line = torch.as_tensor(line, device=device)

        if torch.isnan(line.mean()):
            pdb.set_trace()
            pass

        # make sure it is sorted, so that model does not get confused.
        # flat [[x1, y1], [x2, y2]] to [x1, y1, x2, y2]
        sort_idx = torch.sort(line[:, 0], dim=0, descending=False)[1]
        line = line[sort_idx]
        line = line.flatten()
        line_xyxy.append(line)

    line_xyxy = torch.stack(line_xyxy)

    # if debug:
    #     pdb.set_trace()
    #     pass


    return line_xyxy




# def line_xyxy_to_angle(line_xyxy):
#     """
#     Convert [X1, Y1, X2, Y2] representation of a 2D line to 
#     [sin(theta), cos(theta), offset] representation.

#     xcos(theta) + ysin(theta) = r
    
#     For two points (x1, y1) and (x2, y2) within image plane [0, 1],
#     - cos(theta) = y1 - y2
#     - sin(theta) = x2 - x1
#     - r = x2y1 - x1y2

#     Shengyi Qian, Linyi Jin, Chris Rockwell, Siyi Chen, David Fouhey.
#     Understanding 3D Object Articulation in Internet Videos.
#     CVPR 2022.
#     """
#     eps = 1e-5
#     device = line_xyxy.device
#     line_xyxy = line_xyxy - torch.as_tensor([0.5, 0.5, 0.5, 0.5]).to(device)

#     line_xyxy = line_xyxy.clamp(min=-0.5, max=0.5)
#     x1, y1, x2, y2 = line_xyxy[:,:1], line_xyxy[:,1:2], line_xyxy[:,2:3], line_xyxy[:,3:4]

#     cos = y1 - y2
#     sin = x2 - x1
#     r = x2 * y1 - x1 * y2

#     # normalize, and ensure
#     # sin(theta) in [0, 1]
#     # cos(theta) in [-1, 1]
#     # r in (-sqrt(2) / 2, sqrt(2) / 2)
#     sign = torch.sign(r)
#     mag = torch.sqrt(sin ** 2 + cos ** 2) + eps
#     r = r / mag * sign
#     sin = sin / mag * sign
#     cos = cos / mag * sign

#     #assert (r > (- torch.sqrt(2) / 2)).all()
#     assert (r >= 0).all()
#     assert (r < (np.sqrt(2) / 2)).all()
#     assert (sin >= -1).all()
#     assert (sin <= 1).all()
#     assert (cos >= -1).all()
#     assert (cos <= 1).all()

#     return torch.cat((sin, cos, r), dim=1)


# def line_angle_to_xyxy(line_angle, use_bins=False):
#     """
#     Convert [sin(theta), cos(theta), offset] representation of a 2D line to 
#     [X1, Y1, X2, Y2] representation.

#     Shengyi Qian, Linyi Jin, Chris Rockwell, Siyi Chen, David Fouhey.
#     Understanding 3D Object Articulation in Internet Videos.
#     CVPR 2022.
#     """
#     eps = 1e-5
#     device = line_angle.device

#     sin = line_angle[:, :1] + eps
#     cos = line_angle[:, 1:2] + eps
#     r = line_angle[:, 2:]

#     # normalize sin and cos
#     # make sure r is not out of boundary
#     mag = torch.sqrt(sin ** 2 + cos ** 2)
#     sin = sin / mag
#     cos = cos / mag
#     r = r.clamp(min=0, max=(np.sqrt(2) / 2))

#     # intersect line with four boundaries
#     y1 = (r - cos * (- 0.5)) / sin + 0.5
#     y2 = (r - cos * 0.5) / sin + 0.5
#     x3 = (r - sin * (- 0.5)) / cos + 0.5
#     x4 = (r - sin * 0.5) / cos + 0.5

#     line_xyxy = []
    
#     for i in range(line_angle.shape[0]):
#         line = []
#         if y1[i] > - eps and y1[i] < (1.0 + eps):
#             line.append([0.0, y1[i]])

#         if y2[i] > - eps and y2[i] < (1.0 + eps):
#             line.append([1.0, y2[i]])

#         if len(line) < 2 and x3[i] > - eps and x3[i] < (1.0 + eps):
#             line.append([x3[i], 0.0])

#         if len(line) < 2 and x4[i] > - eps and x4[i] < (1.0 + eps):
#             line.append([x4[i], 1.0])

#         # Mathematically, we should only have two boundary points.
#         # However, in training time, it is not guaranteed the represented
#         # line is within the image plane. Even if r < sqrt(2)/2, it 
#         # can be out of boundary. But it's rare and we want to ignore it.
#         if len(line) != 2:
#             line = [[0.0, y1[i]], [x3[i], 0.0]]

#         line = torch.as_tensor(line, device=device)

#         if torch.isnan(line.mean()):
#             pdb.set_trace()
#             pass

#         # make sure it is sorted, so that model does not get confused.
#         # flat [[x1, y1], [x2, y2]] to [x1, y1, x2, y2]
#         sort_idx = torch.sort(line[:, 0], dim=0, descending=False)[1]
#         line = line[sort_idx]
#         line = line.flatten()
#         line_xyxy.append(line)

#     line_xyxy = torch.stack(line_xyxy)

#     return line_xyxy

