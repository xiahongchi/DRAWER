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

import json
import torchvision
import subprocess

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

def sample_negative(mask_pixel_grid, region):
    min_h, min_w, max_h, max_w = region

    negative_pts = []
    delta = 0.2
    ex_delta = 0.1

    b_min_h = max(int(min_h - delta * (max_h - min_h)), 0)
    b_max_h = min(int(max_h + delta * (max_h - min_h)), mask_pixel_grid.shape[0] - 1)
    b_min_w = max(int(min_w - delta * (max_w - min_w)), 0)
    b_max_w = min(int(max_w + delta * (max_w - min_w)), mask_pixel_grid.shape[1] - 1)

    ex_min_h = max(int(min_h - ex_delta * (max_h - min_h)), 0)
    ex_max_h = min(int(max_h + ex_delta * (max_h - min_h)), mask_pixel_grid.shape[0] - 1)
    ex_min_w = max(int(min_w - ex_delta * (max_w - min_w)), 0)
    ex_max_w = min(int(max_w + ex_delta * (max_w - min_w)), mask_pixel_grid.shape[1] - 1)

    heights = []
    widths = []

    if b_min_h != ex_min_h:
        heights.append((b_min_h + ex_min_h) / 2)
    if b_max_h != ex_max_h:
        heights.append((b_max_h + ex_max_h) / 2)
    if b_min_w != ex_min_w:
        widths.append((b_min_w + b_min_w) / 2)
    if b_max_w != ex_max_w:
        widths.append((b_max_w + ex_max_w) / 2)

    for select_h in heights:
        for select_w in widths:
            negative_pts.append([int(select_h), int(select_w)])

    for select_h in heights:
        negative_pts.append([int(select_h), int((max_w + min_w) / 2)])
        negative_pts.append([int(select_h), int(b_max_w * 0.75 + b_min_w * 0.25)])
        negative_pts.append([int(select_h), int(b_max_w * 0.25 + b_min_w * 0.75)])

    for select_w in widths:
        negative_pts.append([int((max_h + min_h) / 2), int(select_w)])
        negative_pts.append([int(b_max_h * 0.75 + b_min_h * 0.25), int(select_w)])
        negative_pts.append([int(b_max_h * 0.25 + b_min_h * 0.75), int(select_w)])

    return negative_pts

def load_scannetpp_img(data_root, mask_dir):

    batch = {}
    src_names_json = os.path.join(mask_dir, "all.json")
    with open(src_names_json, 'r') as f:
        src_names = json.load(f)
    
    im_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    images = []
    keypoints_p = []
    keypoints_n = []
    img_names = []
    masks = []
    marks = []
    scale_factor_list = []
    original_size_list = []

    for name in src_names:
        idxs = src_names[name]
        stem = os.path.splitext(os.path.basename(name))[0]
        image_path = os.path.join(data_root, stem+".JPG")
        if not os.path.exists(image_path):
            image_path = os.path.join(data_root, stem+".png")
        resized_image = Image.open(image_path).convert("RGB")
        W, H = resized_image.size
        MAX_SIZE = 1024
        scale_factor = 1.0
        original_size = (H, W)
        if max(W, H) > MAX_SIZE:
            scale_factor = MAX_SIZE / max(W, H)
            assert scale_factor < 1.0
            resized_W = int(W * scale_factor)
            resized_H = int(H * scale_factor)
            resized_image = resized_image.resize((resized_W, resized_H))
        image = np.array(resized_image, dtype="uint8")
        H, W, _ = image.shape

        pixel_grid = np.stack(np.meshgrid(
            np.linspace(0, H - 1, H).astype(np.int32),
            np.linspace(0, W - 1, W).astype(np.int32),
            indexing='ij'
        ), axis=-1)
        
        with open(os.path.join(mask_dir, "mask", stem+".json"), 'r') as f:
            mask_src_json = json.load(f)
        with open(os.path.join(mask_dir, "mask_data", stem+".pkl"), 'rb') as f:
            mask_src_data = pickle.load(f)["mask"]
        for mask_dict in mask_src_json:
            value = mask_dict["value"]
            if value not in idxs:
                continue
            if value > 0:
                idx = value - 1
                if scale_factor < 1.0:
                    binary_mask = mask_src_data[idx]
                    binary_mask = torchvision.transforms.functional.resize(binary_mask.unsqueeze(0), (H, W), interpolation=torchvision.transforms.InterpolationMode.NEAREST).reshape(H, W).cpu().numpy()
                else:
                    binary_mask = mask_src_data[idx].reshape(H, W).cpu().numpy()
                mask_pixel_grid = pixel_grid.copy()
                coords = np.stack(np.nonzero(binary_mask), axis=1)  # [:, (1, 0)]
                min_h, min_w = np.min(coords, 0)
                max_h, max_w = np.max(coords, 0)
                negative_pts = sample_negative(mask_pixel_grid, (min_h, min_w, max_h, max_w))

                negative_pts = [torch.LongTensor([
                    int(pt[1]),
                    int(pt[0]),
                ]) for pt in negative_pts]

                delta = 0.2
                positive_pts = []
                possible_positive_pts = [
                    (int(min_h * 0.5 + max_h * 0.5), int(min_w * 0.5 + max_w * 0.5)),
                    (int(min_h * (0.5 - delta) + max_h * (0.5 + delta)), int(min_w * (0.5 - delta) + max_w * (0.5 + delta))),
                    (int(min_h * (0.5 - delta) + max_h * (0.5 + delta)), int(min_w * (0.5 + delta) + max_w * (0.5 - delta))),
                    (int(min_h * (0.5 + delta) + max_h * (0.5 - delta)), int(min_w * (0.5 - delta) + max_w * (0.5 + delta))),
                    (int(min_h * (0.5 + delta) + max_h * (0.5 - delta)), int(min_w * (0.5 + delta) + max_w * (0.5 - delta))),
                ]
                for pti, pt in enumerate(possible_positive_pts):
                    if binary_mask[pt[0], pt[1]]:
                        positive_pts.append(torch.LongTensor([
                            int(pt[1]),
                            int(pt[0]),
                        ]))
                        if pti == 0:
                            break

                assert len(positive_pts) > 0

                images.append(im_transforms(image))
                keypoints_p.append(torch.stack(positive_pts, dim=0).cuda())
                keypoints_n.append(torch.stack(negative_pts, dim=0).cuda())
                img_names.append(stem+".JPG")
                masks.append(torch.from_numpy(binary_mask))
                marks.append({
                    'stem': stem,
                    'idx': idx
                })
                scale_factor_list.append(scale_factor)
                original_size_list.append(original_size)
    
    images = torch.stack(images, dim=0)

    batch['img_name'] = img_names
    batch['image'] = images
    batch['keypoints_p'] = keypoints_p
    batch['keypoints_n'] = keypoints_n
    batch['masks'] = masks
    batch['marks'] = marks
    batch['scale_factor'] = scale_factor_list
    batch['original_size'] = original_size_list

    return batch

def export_imgs(cfg, batch, model, stats, export_dir, device):
    """
    Draw qualitative results in the paper.
    """

    img_names = batch['img_name']
    image = batch['image']
    image_size = (image.shape[2], image.shape[3])
    keypoints_p = batch['keypoints_p']
    keypoints_n = batch['keypoints_n']
    masks = batch['masks']
    marks = batch['marks']
    scale_factor_list = batch['scale_factor']
    original_size_list = batch['original_size']

    # inference
    with torch.no_grad():
        out = model.predict(image, keypoints_p, keypoints_n, masks)

    mask_preds = out['pred_masks']
    mask_preds = interpolate(mask_preds, size=image_size, mode='bilinear', align_corners=False)
    mask_preds = mask_preds.sigmoid() > 0.5
    movable_preds = out['pred_movable'].argmax(dim=-1)
    rigid_preds = out['pred_rigid'].argmax(dim=-1)
    kinematic_preds = out['pred_kinematic'].argmax(dim=-1)
    action_preds = out['pred_action'].argmax(dim=-1)
    axis_preds = out['pred_axis']
    bbox_preds = out['pred_boxes']
    affordance_preds = out['pred_affordance']
    affordance_preds = interpolate(affordance_preds, size=image_size, mode='bilinear', align_corners=False)

    interact_info = {

    }

    for i in range(image.shape[0]):
        img_name = img_names[i]

        scale_factor = scale_factor_list[i]
        H, W = original_size_list[i]

        rgb = image.cpu()[i]

        if scale_factor < 1.0:
            # image = torchvision.transforms.functional.resize(image, (H, W), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
            rgb = torchvision.transforms.functional.resize(rgb, (H, W), interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

        rgb = tensor_to_image(rgb)
        rgb = rgb[:, :, ::-1]

        # regression
        axis_center = box_ops.box_xyxy_to_cxcywh(bbox_preds[i]).clone()
        axis_center[:, 2:] = axis_center[:, :2]
        axis_pred = axis_preds[i]
        axis_pred_norm = F.normalize(axis_pred[:, :2])
        axis_pred = torch.cat((axis_pred_norm, axis_pred[:, 2:]), dim=-1)
        src_axis_xyxys = axis_ops.line_angle_to_xyxy(axis_pred, center=axis_center)

        mask_pred = mask_preds[i, 0]
        if scale_factor < 1.0:
            mask_pred = torchvision.transforms.functional.resize(mask_pred.unsqueeze(0), (H, W), interpolation=torchvision.transforms.InterpolationMode.NEAREST).reshape(H, W)
        
        pred_entry2 = {
            'keypoint': None,
            'bbox': None,
            'mask': mask_pred.cpu().numpy(),
            'affordance': None,
            'move': None,
            'rigid': None,
            'kinematic': None,
            'pull_or_push': None,
            'axis': [-1, -1, -1, -1],
        }
        instances = [pred_entry2]
        output_path = os.path.join(export_dir, 'vis.png')
        vis = Visualizer(rgb)
        colors_teaser = np.array([
            [31, 73, 125], # blue
            [192, 80, 77], # red
        ]) / 255.0
        vis.overlay_instances(instances, assigned_colors=colors_teaser, alpha=0.6)

        instances = []

        mark = marks[i]
        mark = f"{mark['stem']}_{mark['idx']}"
        interact_info[mark] = {}
        # for j in range(len(keypoints_p)):
        j = 0
        # original image + keypoint
        vis = rgb.copy()
        kp = keypoints_p[i][j].cpu().numpy()
        if scale_factor < 1.0:
            kp = [int(elem * (1 / scale_factor)) for elem in kp]
        vis = cv2.circle(vis, kp, 4, (255, 255, 255), -1)
        vis = cv2.circle(vis, kp, 2, (31, 73, 125), -1)
        img_path = os.path.join(export_dir, '{}_{:0>2}_kp_{:0>2}_01_img.png'.format(img_name, i, j))
        Image.fromarray(vis).save(img_path)

        # physical properties
        movable_pred = movable_preds[i, j].item()
        rigid_pred = rigid_preds[i, j].item()
        kinematic_pred = kinematic_preds[i, j].item()
        action_pred = action_preds[i, j].item()
        output_path = os.path.join(export_dir, '{}_{:0>2}_kp_{:0>2}_02_phy.png'.format(img_name, i, j))
        draw_properties(output_path, movable_pred, rigid_pred, kinematic_pred, action_pred)

        interact_info[mark]['phy'] = {
            'movable': movable_pred,
            'rigid': rigid_pred,
            'kinematic': kinematic_pred,
            'action': action_pred,
        }

        # box mask axis
        axis_pred = src_axis_xyxys[j]
        if kinematic_imap[kinematic_pred] != 'rotation':
            axis_pred = [-1, -1, -1, -1]
        img_path = os.path.join(export_dir, '{}_{:0>2}_kp_{:0>2}_03_loc.png'.format(img_name, i, j))
        draw_localization(
            rgb, 
            img_path, 
            None,
            mask_pred.cpu().numpy(),
            axis_pred,
            colors=None,
            alpha=0.6,    
        )
        interact_info[mark]['axis'] = axis_pred

        # affordance
        affordance_pred = affordance_preds[i, j].sigmoid()
        if scale_factor < 1.0:
            affordance_pred = torchvision.transforms.functional.resize(affordance_pred.unsqueeze(0), (H, W), interpolation=torchvision.transforms.InterpolationMode.BILINEAR).reshape(H, W)

        affordance_pred = affordance_pred.cpu().numpy() #[:, :, np.newaxis]
        aff_path = os.path.join(export_dir, '{}_{:0>2}_kp_{:0>2}_04_affordance.png'.format(img_name, i, j))
        draw_affordance(rgb, aff_path, affordance_pred)

        interact_info[mark]['affordance'] = affordance_pred
        
    with open(os.path.join(export_dir, 'interact_info.pkl'), 'wb') as f:
        pickle.dump(interact_info, f)


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
    
    checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_path)
    if cfg.resume and os.path.isfile(checkpoint_path):
        logger.info(f"Resuming from checkpoint {checkpoint_path}.")
        if not accelerator.is_local_main_process:
            map_location = {'cuda:0': 'cuda:%d' % accelerator.local_process_index}
        else:
            # Running locally
            map_location = "cuda:0"
        
        loaded_data = torch.load(checkpoint_path, map_location='cpu')

        state_dict = loaded_data["model"]

        model.load_state_dict(state_dict, strict=False)
        model = model.cuda()

        # continue training: load optimizer and stats
        stats = pickle.loads(loaded_data["stats"])
        logger.info(f"   => resuming from epoch {stats.epoch}.")
    else:
        logger.warning("Start from scratch.")


    # Prepare the model for accelerate and move to the relevant device
    model = accelerator.prepare(
        model
    )

    # Set the model to the training mode.
    model.eval()

    # random seed again to ensure we have the same batch of images
    # for different models
    torch.manual_seed(cfg.seed)

    export_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.output_dir)
    os.makedirs(export_dir, exist_ok=True)
    
    subprocess.run(f"rm -rf {export_dir}/*", shell=True)
    
    data_root = cfg.image_dir
    mask_dir = os.path.join(cfg.data_dir, "grounded_sam")
    
    batch = load_scannetpp_img(data_root, mask_dir)

    export_imgs(cfg, batch, model, stats, export_dir, device)


if __name__=="__main__":
    main()
