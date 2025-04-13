import torch
import torch.nn.functional as F
import os
import numpy as np
import hydra
import logging
import submitit
import pickle
import collections
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
import pdb
import cv2
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

# get_monoarti_datasets
from monoarti.dataset import get_interaction_datasets
from monoarti.stats import Stats
from monoarti.visualizer import Visualizer
from monoarti.detr import box_ops
from monoarti.model import build_model
from monoarti import axis_ops, depth_ops
from monoarti.detr.misc import interpolate
from monoarti.vis_utils import draw_properties, draw_affordance, draw_localization

# 3d modules
# comment them if you do not need export_video
import pytorch3d
from pytorch3d.io import save_obj
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.structures import Meshes#, Textures
from pytorch3d.renderer.mesh import TexturesVertex, TexturesUV, Textures
from pytorch3d.utils import ico_sphere
from monoarti.mesh_utils import triangulate_pcd, get_pcd, fit_homography, get_axis_mesh, save_obj_articulation


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
logger = logging.getLogger(__name__)



def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = b
        cmap[i, 1] = g
        cmap[i, 2] = r
    return cmap


colors_256 = labelcolormap(256)

colors = np.array([[255, 0, 0],
                   [0, 255, 0],
                   [0, 0, 255],
                   [80, 128, 255],
                   [255, 230, 180],
                   [255, 0, 255],
                   [0, 255, 255],
                   [100, 0, 0],
                   [0, 100, 0],
                   [255, 255, 0],
                   [50, 150, 0],
                   [200, 255, 255],
                   [255, 200, 255],
                   [128, 128, 80],
                   [0, 50, 128],
                   [0, 100, 100],
                   [0, 255, 128],
                   [0, 128, 255],
                   [255, 0, 128],
                   [128, 0, 255],
                   [255, 128, 0],
                   [128, 255, 0],
                   [0, 0, 0]
                   ])


Tensor_to_Image = transforms.Compose([
    transforms.Normalize([0.0, 0.0, 0.0], [1.0/0.229, 1.0/0.224, 1.0/0.225]),
    transforms.Normalize([-0.485, -0.456, -0.406], [1.0, 1.0, 1.0]),
    transforms.ToPILImage()
])


def tensor_to_image(image):
    image = Tensor_to_Image(image)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


movable_imap = {
    0: 'one_hand',
    1: 'two_hands',
    2: 'fixture',
    -100: 'n/a',
}

rigid_imap = {
    1: 'yes',
    0: 'no',
    2: 'bad',
    -100: 'n/a',
}

kinematic_imap = {
    0: 'freeform',
    1: 'rotation',
    2: 'translation',
    -100: 'n/a'
}

action_imap = {
    0: 'free',
    1: 'pull',
    2: 'push',
    -100: 'n/a',
}


def export_imgs(cfg, val_dataloader, model, stats, export_dir, device):
    """
    Draw qualitative results in the paper.
    """
    batch_size = cfg.train.batch_size
    shortlist = []

    for iteration, batch in tqdm(enumerate(val_dataloader)):
        img_names = batch['img_name']
        image = batch['image']
        image_size = (image.shape[2], image.shape[3])
        bbox = batch['bbox']
        valid = batch['valid']
        keypoints = batch['keypoints']
        movable_gts = batch['movable']
        rigid_gts = batch['rigid']
        kinematic_gts = batch['kinematic']
        action_gts = batch['action']
        axis_gts = batch['axis']
        depth_gts = batch['depth']
        affordance_gts = batch['affordance_map']
        affordance_gts = interpolate(affordance_gts, size=image_size, mode='bilinear', align_corners=False)
        mask_gts = batch['masks']

        bbox_gt_scaled = box_ops.rescale_bboxes(bbox.cpu(), cfg.data.image_size).long()

        # inference
        with torch.no_grad():
           out = model(**batch)

        mask_preds = out['pred_masks']
        mask_preds = interpolate(mask_preds, size=image_size, mode='bilinear', align_corners=False)
        mask_preds = mask_preds.sigmoid() > 0.5
        movable_preds = out['pred_movable'].argmax(dim=-1)
        rigid_preds = out['pred_rigid'].argmax(dim=-1)
        kinematic_preds = out['pred_kinematic'].argmax(dim=-1)
        action_preds = out['pred_action'].argmax(dim=-1)
        axis_preds = out['pred_axis']
        depth_preds = out['pred_depth']
        affordance_preds = out['pred_affordance']
        affordance_preds = interpolate(affordance_preds, size=image_size, mode='bilinear', align_corners=False)
        if depth_preds is not None:
            depth_preds = interpolate(depth_preds, size=depth_gts.shape[-2:], mode='bilinear', align_corners=False)

        for i in range(image.shape[0]):
            img_name = img_names[i]
            img_idx = iteration * batch_size + i

            if len(shortlist) > 0 and img_name not in shortlist:
               continue

            rgb = tensor_to_image(image.cpu()[i])
            rgb = rgb[:, :, ::-1]

            # regression
            axis_center = box_ops.box_xyxy_to_cxcywh(bbox[i]).clone()
            axis_center[:, 2:] = axis_center[:, :2]
            axis_pred = axis_preds[i]
            axis_pred_norm = F.normalize(axis_pred[:, :2])
            axis_pred = torch.cat((axis_pred_norm, axis_pred[:, 2:]), dim=-1)
            src_axis_xyxys = axis_ops.line_angle_to_xyxy(axis_pred, center=axis_center)
            
            pred_entry1 = {
                'keypoint': None,
                'bbox': None,
                'mask': mask_gts[i, 0].cpu().numpy(),
                'affordance': None,
                'move': None,
                'rigid': None,
                'kinematic': None,
                'pull_or_push': None,
                'axis': [-1, -1, -1, -1],
            }
            pred_entry2 = {
                'keypoint': None,
                'bbox': None,
                'mask': mask_preds[i, 2].cpu().numpy(),
                'affordance': None,
                'move': None,
                'rigid': None,
                'kinematic': None,
                'pull_or_push': None,
                'axis': [-1, -1, -1, -1],
            }
            instances = [pred_entry2, pred_entry1]
            output_path = os.path.join(export_dir, 'vis.png')
            vis = Visualizer(rgb)
            colors_teaser = np.array([
                [31, 73, 125], # blue
                [192, 80, 77], # red
            ]) / 255.0
            vis.overlay_instances(instances, assigned_colors=colors_teaser, alpha=0.6)

            instances = []
            for j in range(15):
                if not valid[i, j]:
                    break

                # original image + keypoint
                vis = rgb.copy()
                kp = keypoints[i, j].cpu().numpy()
                vis = cv2.circle(vis, kp, 24, (255, 255, 255), -1)
                vis = cv2.circle(vis, kp, 20, (31, 73, 125), -1)
                img_path = os.path.join(export_dir, '{}_kp_{:0>2}_01_img.png'.format(img_name, j))
                Image.fromarray(vis).save(img_path)

                # physical properties
                movable_pred = movable_preds[i, j].item()
                rigid_pred = rigid_preds[i, j].item()
                kinematic_pred = kinematic_preds[i, j].item()
                action_pred = action_preds[i, j].item()
                output_path = os.path.join(export_dir, '{}_kp_{:0>2}_02_phy.png'.format(img_name, j))
                draw_properties(output_path, movable_pred, rigid_pred, kinematic_pred, action_pred)

                # box mask axis
                axis_pred = src_axis_xyxys[j]
                if kinematic_imap[kinematic_pred] != 'rotation':
                    axis_pred = [-1, -1, -1, -1]
                img_path = os.path.join(export_dir, '{}_kp_{:0>2}_03_loc.png'.format(img_name, j))
                draw_localization(
                    rgb, 
                    img_path, 
                    None,
                    mask_preds[i, j].cpu().numpy(),
                    axis_pred,
                    colors=None,
                    alpha=0.6,    
                )

                # affordance
                affordance_pred = affordance_preds[i, j].sigmoid()
                affordance_pred = affordance_pred.cpu().numpy() #[:, :, np.newaxis]
                aff_path = os.path.join(export_dir, '{}_kp_{:0>2}_04_affordance.png'.format(img_name, j))
                draw_affordance(rgb, aff_path, affordance_pred)

            axis_gt = axis_gts[i]
            valid_axis = axis_gt[:, 0] > 0
            if valid_axis.sum() > 0:
                axis_angle = axis_ops.line_xyxy_to_angle(axis_gt[valid_axis])
                axis_xyxy = axis_ops.line_angle_to_xyxy(axis_angle)
                axis_gt[valid_axis] = axis_xyxy

            gt_instances = []
            for j in range(10):
                if not valid[i, j]:
                    break

                # physical properties
                movable_pred = movable_gts[i, j].item()
                rigid_pred = rigid_gts[i, j].item()
                kinematic_pred = kinematic_gts[i, j].item()
                action_pred = action_gts[i, j].item()
                output_path = os.path.join(export_dir, '{}_kp_{:0>2}_02_phy_gt.png'.format(img_name, j))
                draw_properties(output_path, movable_pred, rigid_pred, kinematic_pred, action_pred)

                # box mask axis
                img_path = os.path.join(export_dir, '{}_kp_{:0>2}_03_loc_gt.png'.format(img_name, j))
                draw_localization(
                    rgb, 
                    img_path, 
                    None,
                    mask_gts[i, j].cpu().numpy(),
                    axis_gt[j],
                    colors=None,
                    alpha=0.6,    
                )

                # affordance
                affordance_gt = affordance_gts[i, j] #.sigmoid()
                affordance_gt = affordance_gt.cpu().numpy() #[:, :, np.newaxis]
                aff_path = os.path.join(export_dir, '{}_kp_{:0>2}_04_affordance_gt.png'.format(img_name, j))
                draw_affordance(rgb, aff_path, affordance_gt)

            # depth
            if depth_preds is not None:
                depth_pred = depth_preds[i]
                gt_depth = depth_gts[i]
                if (gt_depth > -0.5).any():
                    depth_pred_metric = depth_ops.recover_metric_depth(depth_pred[0], gt_depth)
                else:
                    depth_pred_metric = depth_pred[0] * 0.945 + 0.658
                    depth_pred_metric = depth_pred_metric.cpu().numpy()

                for j in range(15):
                    if not valid[i, j]:
                        break

                    fig = plt.figure()
                    plt.imshow(depth_pred_metric, cmap=mpl.colormaps['plasma'])
                    plt.axis('off')
                    depth_path = os.path.join(export_dir, '{}_kp_{:0>2}_05_depth.png'.format(img_name, j))
                    plt.savefig(depth_path, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    fig = plt.figure()
                    plt.imshow(gt_depth.cpu().numpy(), cmap=mpl.colormaps['plasma'])
                    plt.axis('off')
                    depth_path = os.path.join(export_dir, '{}_kp_{:0>2}_05_depth_gt.png'.format(img_name, j))
                    plt.savefig(depth_path, bbox_inches='tight', pad_inches=0)
                    plt.close(fig)


def export_video(cfg, val_dataloader, model, stats, export_dir, device):
    batch_size = cfg.train.batch_size
    assert batch_size == 1

    shortlist_ids = [
        'taskonomy_springerville_point_774_view_4_domain_rgb.png',  # teaser
        'taskonomy_everton_point_669_view_13_domain_rgb.png',
        'taskonomy_haxtun_point_338_view_5_domain_rgb.png',
        'taskonomy_helix_point_125_view_2_domain_rgb.png',
        'taskonomy_ihlen_point_1418_view_3_domain_rgb.png', # drawer
        'AR_AbWI265V0rI_4_540_55.jpg', # drawer
        'AR_Bh0qJ76_f_Y_0_7470_85.jpg', # refrigerator drawer
        'AR_DW3IFMnWSko_12_270_35.jpg', # a set of drawers
        'AR_M_7yDZyaPZY_60_0_55.jpg', # a set of drawers
        'AR_U0p5YHBWgjs_29_540_5.jpg', # washing machine
        'AR_bTBnBBCi6VI_5_11430_5.jpg', # refrigerator drawer
        'EK_0043_P30_101_frame_0000015392.jpg', # refrigerator, challenging
    ]

    for iteration, batch in tqdm(enumerate(val_dataloader)):
        img_name = batch['img_name']
        if len(shortlist_ids) > 0 and img_name[0] not in shortlist_ids:
            continue

        image = batch['image']
        image_size = (image.shape[2], image.shape[3])
        bbox = batch['bbox']
        valid = batch['valid']
        keypoints = batch['keypoints']
        movable_gts = batch['movable']
        rigid_gts = batch['rigid']
        kinematic_gts = batch['kinematic']
        action_gts = batch['action']
        axis_gts = batch['axis']
        depth_gts = batch['depth']
        fov = batch['fov']

        bbox_gt_scaled = box_ops.rescale_bboxes(bbox.cpu(), cfg.data.image_size).long()

        # inference
        with torch.no_grad():
           out = model(**batch)

        bbox_preds = out['pred_boxes']
        bbox_scaled = box_ops.rescale_bboxes(out['pred_boxes'].cpu(), cfg.data.image_size).long()
        movable_preds = out['pred_movable'].argmax(dim=-1)
        rigid_preds = out['pred_rigid'].argmax(dim=-1)
        kinematic_preds = out['pred_kinematic'].argmax(dim=-1)
        action_preds = out['pred_action'].argmax(dim=-1)
        axis_preds = out['pred_axis']
        depth_preds = out['pred_depth']
        depth_preds = interpolate(depth_preds, size=depth_gts.shape[-2:], mode='bilinear', align_corners=False)
        mask_preds = out['pred_masks']
        mask_preds = interpolate(mask_preds, size=image_size, mode='bilinear', align_corners=False)
        mask_preds = mask_preds.sigmoid() > 0.5
        image_size = (image.shape[2], image.shape[3])

        for i in range(image.shape[0]):
            rgb = tensor_to_image(image.cpu()[i])
            rgb = rgb[:, :, ::-1]

            axis_center = box_ops.box_xyxy_to_cxcywh(bbox[i]).clone()
            axis_center[:, 2:] = axis_center[:, :2]
            axis_pred = axis_preds[i]
            axis_pred_norm = F.normalize(axis_pred[:, :2])
            axis_pred = torch.cat((axis_pred_norm, axis_pred[:, 2:]), dim=-1)
            src_axis_xyxys = axis_ops.line_angle_to_xyxy(axis_pred, center=axis_center)

            instances = []
            for j in range(15):
                if not valid[i, j]:
                    break

                movable_pred = movable_preds[i, j].item()
                rigid_pred = rigid_preds[i, j].item()
                kinematic_pred = kinematic_preds[i, j].item()
                action_pred = action_preds[i, j].item()

                axis_pred = src_axis_xyxys[j]
                if kinematic_imap[kinematic_pred] != 'rotation':
                    axis_pred = [-1, -1, -1, -1]

                pred_entry = {
                    'keypoint': keypoints[i, j].cpu().numpy().tolist(),
                    'bbox': bbox_scaled[i, j].numpy().tolist(),
                    'mask': mask_preds[i, j].cpu().numpy(),
                    'affordance': None,
                    'move': movable_imap[movable_pred],
                    'rigid': rigid_imap[rigid_pred],
                    'kinematic': kinematic_imap[kinematic_pred],
                    'pull_or_push': action_imap[action_pred],
                    'axis': axis_pred,
                }

                instances.append(pred_entry)

            vis = Visualizer(rgb)
            vis.overlay_instances(instances)
            img_path = os.path.join(export_dir, '{:0>6}_det.png'.format(iteration * batch_size + i))
            vis.output.save(img_path)

            # depth
            depth_pred = depth_preds[i]
            gt_depth = depth_gts[i]
            if gt_depth[0, 0] <= -0.5:
                depth_pred_metric = depth_pred * 6.69 + 1.10 # omnidata average scale and shift
                depth_pred_metric = depth_pred_metric.cpu()[0]
            else:
                #depth_pred_metric = depth_pred * 6.69 + 1.10 # omnidata average scale and shift
                depth_pred_metric = depth_ops.recover_metric_depth(depth_pred[0], gt_depth)
                depth_pred_metric = torch.as_tensor(depth_pred_metric)

            # focal length
            if fov[i] < 0: # no gt fov
                fov_proj = torch.as_tensor(1.0).to(fov[i].device)
            else:
                fov_proj = fov[i]

            focal_length = (image_size[1] / 2 / torch.tan(fov_proj / 2)).item()

            for obj_idx, inst in enumerate(instances):
                if inst['kinematic'] == 'freeform':
                    continue

                obj_mask = mask_preds[i, obj_idx]
                if obj_mask.sum() < 20:
                    continue

                # unproject depth to point cloud
                camera = FoVPerspectiveCameras(fov=fov_proj, degrees=False)
                xys = depth_pred_metric.nonzero().float()
                xys[:, 0] = - (xys[:, 0] / image_size[0] - 0.5) * 2
                xys[:, 1] = - (xys[:, 1] / image_size[1] - 0.5) * 2
                xys = xys.flip(1)
                depth_pred_flat = depth_pred_metric.reshape(-1, 1)
                xy_depth = torch.cat([xys, depth_pred_flat], dim=1)
                points = camera.unproject_points(xy_depth, world_coordinates=False)
                points = points.reshape((*image_size, 3))

                verts = points.reshape(-1, 3)
                rgb_tensor = torch.as_tensor(rgb.copy()) / 255.0
                # rgb_mask = torch.ones_like(rgb_tensor)[:, :, 0]
                # faces = triangulate_pcd(rgb_mask, rgb_mask, image_size, 0)
                # verts_uvs = depth_pred_metric.nonzero().float()
                # verts_uvs[:, 0] = - verts_uvs[:, 0] / image_size[0]
                # verts_uvs[:, 1] = - (1 - verts_uvs[:, 1] / image_size[1])
                # verts_uvs = verts_uvs.flip(1)
                # obj_path = os.path.join(export_dir, '{:0>6}_pred.obj'.format(iteration * batch_size + i))
                # save_obj(obj_path, verts, faces, verts_uvs=verts_uvs, faces_uvs=faces, texture_map=rgb_tensor)

                # estimation plane normal and offset for the object
                obj_points = points[obj_mask]
                plane_normal, plane_offset = depth_ops.fit_plane(obj_points.numpy(), thres=0.25)
                plane_normal = torch.as_tensor(plane_normal)
                plane_offset = torch.as_tensor(plane_offset)

                num_steps = 5

                if inst['kinematic'] == 'rotation':
                    axis_pts = instances[obj_idx]['axis'].reshape(-1, 2)
                    axis_pts[:, 0] = axis_pts[:, 0] * 1024
                    axis_pts[:, 1] = axis_pts[:, 1] * 768
                    axis_pts[:, 0] = 1024 - axis_pts[:, 0]
                    axis_pts = axis_pts.long()
                    axis_points = get_pcd(axis_pts.cpu(), plane_normal, plane_offset, h=image_size[0], w=image_size[1], focal_length=focal_length)

                    # create visualization for rot axis
                    # axis_mesh = get_axis_mesh(0.05, axis_points[0], axis_points[1])
                    # axis_mesh.textures = Textures(verts_uvs=torch.ones_like(axis_mesh.verts_list()[0][:,:2])[None], faces_uvs=axis_mesh.faces_list(), maps=torch.zeros((1,5,5,3)))
                    # basename = 'axis'
                    # uv_maps_list = [rgb_tensor]
                    # save_obj_articulation(export_dir, basename+'_pred', axis_mesh, decimal_places=10, uv_maps=uv_maps_list)
                    # pdb.set_trace()

                    axis_scale = pytorch3d.transforms.Scale(0.1).cuda()
                    axis_pt1_t = pytorch3d.transforms.Transform3d().translate(axis_points[0][0], axis_points[0][1], axis_points[0][2])
                    axis_pt1_t = axis_pt1_t.cuda()
                    axis_pt1 = ico_sphere(0).cuda()
                    axis_pt1.verts_list()[0] = axis_scale.transform_points(axis_pt1.verts_list()[0])
                    axis_pt1.verts_list()[0] = axis_pt1_t.transform_points(axis_pt1.verts_list()[0])
                    axis_pt2_t = pytorch3d.transforms.Transform3d().translate(axis_points[1][0], axis_points[1][1], axis_points[1][2])
                    axis_pt2_t = axis_pt2_t.cuda()
                    axis_pt2 = ico_sphere(0).cuda()
                    axis_pt2.verts_list()[0] = axis_scale.transform_points(axis_pt2.verts_list()[0])
                    axis_pt2.verts_list()[0] = axis_pt2_t.transform_points(axis_pt2.verts_list()[0])
                    axis_verts_rgb = torch.ones_like(verts)[None].cuda()  # (1, V, 3)
                    # axis_textures = Textures(verts_uvs=axis_verts_rgb, faces_uvs=axis_pt1.faces_list(), maps=torch.zeros((1,5,5,3)).cuda())
                    # axis_pt1.textures = axis_textures
                    # axis_pt2.textures = axis_textures

                    # transform objects
                    axis_dir = 'r'
                    pcd = obj_points
                    obj_faces = triangulate_pcd(obj_mask.cpu(), obj_mask.cpu(), image_size, 0)
                    dir_vec = axis_points[1] - axis_points[0]
                    dir_vec = dir_vec / np.linalg.norm(dir_vec)
                    t1 = pytorch3d.transforms.Transform3d().translate(axis_points[0][0], axis_points[0][1], axis_points[0][2])
                    t1 = t1.cuda()
                    angles = torch.FloatTensor(np.linspace(0.0, 1.8, num_steps)[:, np.newaxis])
                    axis_angles = angles * dir_vec
                    rot_mats = pytorch3d.transforms.axis_angle_to_matrix(axis_angles)
                    t2 = pytorch3d.transforms.Rotate(rot_mats)
                    t2 = t2.cuda()
                    t3 = t1.inverse()
                    pcd_trans = t3.transform_points(pcd.cuda())
                    pcd_trans = t2.transform_points(pcd_trans)
                    pcd_trans = t1.transform_points(pcd_trans)
                    pcd_trans = pcd_trans.cpu()

                    meshes = [axis_pt1, axis_pt2]
                    uv_maps_list = []
                    meshes = pytorch3d.structures.join_meshes_as_batch(meshes)
                    meshes = meshes.cpu()

                    
                elif inst['kinematic'] == 'translation':
                    angles = torch.linspace(0, 0.40, num_steps).unsqueeze(1)
                    dir_vec = plane_normal
                    trans_vectors = angles * dir_vec
                    t_combined = pytorch3d.transforms.Transform3d().translate(trans_vectors)
                    t_combined = t_combined.cuda()
                    pcd = obj_points
                    pcd_trans = t_combined.transform_points(pcd.cuda())
                    pcd_trans = pcd_trans.cpu()
                else:
                    raise ValueError
                
                # project back
                pts_reproj = camera.transform_points(pcd_trans)
                pts_reproj = pts_reproj[:, :, :2] # drop depth
                pts_reproj = pts_reproj.flip(2)
                pts_reproj = (- pts_reproj / 2) + 0.5
                pts_reproj[:, :, 0] = pts_reproj[:, :, 0] * image_size[0]
                pts_reproj[:, :, 1] = pts_reproj[:, :, 1] * image_size[1]

                
                for warp_idx in range(num_steps):
                    src_pts = obj_mask.nonzero().cpu()
                    H = fit_homography(src_pts.flip(1), pts_reproj[warp_idx].flip(1))
                    obj_rgb = rgb.copy()
                    obj_rgb[~obj_mask.cpu()] = 0
                    warp_rgb = cv2.warpPerspective(obj_rgb, H, (1024, 768))
                    warp_mask = ~(warp_rgb.sum(axis=2) == 0)
                    
                    vis_rgb = rgb.copy()
                    vis_rgb[obj_mask.cpu()] = 0
                    vis_rgb[warp_mask] = warp_rgb[warp_mask]

                    output_path = os.path.join(export_dir, '{:0>6}_kp_{}_rot_{}.png'.format(iteration * batch_size + i, obj_idx, warp_idx))
                    Image.fromarray(vis_rgb).save(output_path)
                
                # obj_path = os.path.join(export_dir, '{:0>6}_rot.obj'.format(iteration * batch_size + i))
                # save_obj(obj_path, pcd_trans[1], obj_faces, verts_uvs=verts_uvs, faces_uvs=obj_faces, texture_map=rgb_tensor)

                # obj_path = os.path.join(export_dir, '{:0>6}_pred_axis_0.obj'.format(iteration * batch_size + i))
                # save_obj(obj_path, meshes.verts_list()[0], meshes.faces_list()[0], verts_uvs=verts_uvs, faces_uvs=faces, texture_map=rgb_tensor)

                # obj_path = os.path.join(export_dir, '{:0>6}_pred_axis_1.obj'.format(iteration * batch_size + i))
                # save_obj(obj_path, meshes.verts_list()[1], meshes.faces_list()[1], verts_uvs=verts_uvs, faces_uvs=faces, texture_map=rgb_tensor)


@hydra.main(config_path=CONFIG_DIR, config_name="defaults", version_base='1.2')
def main(cfg: DictConfig):
    # Set the relevant seeds for reproducibility.
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if len(cfg.output_dir) == 0:
        raise ValueError("output_dir must be set for test!")

    logger.critical("launching experiment {}".format(cfg.experiment_name))

    ddp_scaler = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_scaler])
    device = accelerator.device
    
    # Init model according to the config.
    model = build_model(cfg)

    # Init stats to None before loading.
    stats = None
    optimizer_state_dict = None
    start_epoch = 0
    
    checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_path)
    if cfg.resume and os.path.isfile(checkpoint_path):
        logger.info(f"Resuming from checkpoint {checkpoint_path}.")
        if not accelerator.is_local_main_process:
            map_location = {'cuda:0': 'cuda:%d' % accelerator.local_process_index}
        else:
            # Running locally
            map_location = "cuda:0"
        
        loaded_data = torch.load(checkpoint_path, map_location=map_location)

        state_dict = loaded_data["model"]

        model.load_state_dict(state_dict, strict=False)

        # continue training: load optimizer and stats
        stats = pickle.loads(loaded_data["stats"])
        logger.info(f"   => resuming from epoch {stats.epoch}.")
        optimizer_state_dict = loaded_data["optimizer"]
        start_epoch = stats.epoch + 1
    else:
        logger.warning("Start from scratch.")

    
    print("data split: {}".format(cfg.test.split))

    # Init the stats object.
    if stats is None:
        stats = Stats(
            ["triplet_loss", "sec/it"],
        )

    batch_size = cfg.train.batch_size


    train_dataset, val_dataset, test_dataset = get_interaction_datasets(
        train_dataset_names=cfg.data.train_dataset_names,
        val_dataset_names=cfg.data.val_dataset_names,
        test_dataset_names=cfg.data.test_dataset_names,
        image_size=cfg.data.image_size,
        output_size=cfg.data.output_size,
        num_views=cfg.train.num_views,
        load_depth=cfg.train.depth_on,
        affordance_radius=cfg.data.affordance_radius,
        num_queries=cfg.data.num_queries,
        bbox_to_mask=cfg.data.bbox_to_mask,
    )

    logger.info("train has {} examples".format(len(train_dataset)))
    logger.info("val has {} examples".format(len(val_dataset)))  

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        #collate_fn=collate_fn,
    )

    print("data split: {}".format(cfg.test.split))
    if cfg.test.split == 'train':
        val_dataset = train_dataset
    elif cfg.test.split == 'val':
        pass
    elif cfg.test.split == 'test':
        val_dataset = test_dataset
    else:
        raise NotImplementedError

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )

    # Prepare the model for accelerate and move to the relevant device
    model, train_dataloader, val_dataloader = accelerator.prepare(
        model, train_dataloader, val_dataloader
    )

    # Set the model to the training mode.
    model.eval()

    # random seed again to ensure we have the same batch of images
    # for different models
    torch.manual_seed(cfg.seed)

    export_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.output_dir)
    os.makedirs(export_dir, exist_ok=True)

    if cfg.test.mode == 'export_imgs':
        export_imgs(cfg, val_dataloader, model, stats, export_dir, device)
    elif cfg.test.mode == "export_video":
        export_video(cfg, val_dataloader, model, stats, export_dir, device)
    else:
        raise NotImplementedError


if __name__=="__main__":
    main()
