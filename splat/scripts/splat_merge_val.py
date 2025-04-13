from __future__ import annotations
import os

import cv2

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType


import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
import pickle
import transformations
import trimesh
import imageio

from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion, quaternion_invert, quaternion_multiply
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.structures import Meshes
from nerfstudio.engine.optimizers import Optimizers
from torch.cuda.amp.grad_scaler import GradScaler
import wandb
from tqdm import tqdm
from datetime import datetime
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.renderer import TexturesUV, TexturesVertex, RasterizationSettings, MeshRenderer, MeshRendererWithFragments, MeshRasterizer, HardPhongShader, SoftSilhouetteShader, PointLights, AmbientLights, FoVPerspectiveCameras, look_at_view_transform

from torchmetrics.image import PeakSignalNoiseRatio
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial.transform import Slerp, Rotation

import sys
sys.path.append("./scripts")
# from guidance import Guidance, guidance_config
import torchvision.transforms.functional as F_V
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
import networkx as nx

import torch._dynamo
torch._dynamo.config.suppress_errors = True
import nvdiffrast.torch as dr

import argparse

def point_transform(point, transform):
    point_shape = point.shape
    point_pad = np.pad(point.reshape(-1, 3), ((0, 0), (0, 1)), constant_values=(0, 1))
    point_transformed_pad = point_pad @ transform.T
    point_transformed = point_transformed_pad[:, :3] / point_transformed_pad[:, 3:]
    return point_transformed.reshape(point_shape)

glctx = dr.RasterizeGLContext(output_db=False)

parser = argparse.ArgumentParser()
parser.add_argument("--splat_dir", type=str, required=True)
parser.add_argument("--sdf_dir", type=str, required=True)
parser.add_argument("--interior_dir", type=str, required=True)
parser.add_argument("--save_note", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
args = parser.parse_args()

save_note = args.save_note
bakedsdf_dir = args.sdf_dir
load_config = os.path.join(args.splat_dir, "config.yml")
save_dir = args.save_dir

os.makedirs(save_dir, exist_ok=True)
config, pipeline, checkpoint_path, _ = eval_setup(Path(load_config))

model = pipeline.model
model = model.eval()

load_dir = os.path.join(os.path.dirname(load_config), save_note)
load_dict = torch.load(os.path.join(load_dir, "som.pt"))
print("load articulation gsplats from ", os.path.join(load_dir, "som.pt"))
model.load_splatfacto_on_mesh(load_dict)

with open(os.path.join(load_dir, "trace.pkl"), 'rb') as f:
    trace_dict = pickle.load(f)
prim_means_indices = trace_dict["prim_means_indices"]
append_indices_prims = trace_dict["append_indices_prims"]
append_indices_prims_door = trace_dict["append_indices_prims_door"]
trace_data = trace_dict["trace_data"]
interact_type_list = trace_dict["interact_type_list"]
total_drawers_num = len(interact_type_list)
total_frames = len(trace_data["00"])
prim_means_indices = [torch.from_numpy(elem).cuda() for elem in prim_means_indices]
append_indices_prims = [torch.from_numpy(elem).cuda() for elem in append_indices_prims]
append_indices_prims_door = [torch.from_numpy(elem).cuda() for elem in append_indices_prims_door]
N_Gaussians = model.means.shape[0]


max_move_frame_list = []
for prim_i in range(total_drawers_num):
    trace_data_prim = trace_data[f"{prim_i:02d}"]
    interact_type = interact_type_list[prim_i]
    if interact_type in ["2", "3.3"]:
        init_pos = np.array(trace_data_prim[0]["position"]).reshape(3)
        final_pos = np.array(trace_data_prim[total_frames-1]["position"]).reshape(3)

        move_diff_abs = np.abs(final_pos - init_pos)
        max_move_axis = np.argmax(move_diff_abs)

        pos_move_axis = np.array([trace_data_prim[pos_move_i]["position"][max_move_axis] for pos_move_i in range(total_frames)]).reshape(-1)
        max_move_frame = np.min(np.arange(total_frames)[np.abs(pos_move_axis - pos_move_axis[0]) / move_diff_abs[max_move_axis] > 0.99])
        # print("max_move_frame: ", max_move_frame)
        max_move_frame_list.append(max_move_frame)
    else:
        delta_frame = 10
        rot_all = np.stack(
            [np.array(
            transformations.quaternion_matrix(trace_data_prim[rot_i]["orientation"]))[:3, :3]
                   for rot_i in range(total_frames)], axis=0)
        rot_diff = np.matmul(rot_all[delta_frame:], np.linalg.inv(rot_all[:-delta_frame]))
        # print("rot_diff: ", rot_diff.shape, rot_diff)
        max_move_frame = np.min(np.arange(total_frames)[delta_frame:][np.linalg.norm((rot_diff - np.eye(3).reshape(1, 3, 3)).reshape(-1, 9), axis=-1) < 1e-5])
        # print("max_move_frame: ", max_move_frame)
        max_move_frame_list.append(max_move_frame)
    
image_size = 512
target_camera = pipeline.datamanager.eval_dataset.cameras[0]
fx = float(target_camera.fx) * 0.5
fy = float(target_camera.fy) * 0.5
fx = fy = (fx + fy) / 2
cx = image_size * 0.5  # float(target_camera.cx)
cy = image_size * 0.5  # float(target_camera.cy)
# H, W = int(target_camera.height), int(target_camera.width)
H, W = image_size, image_size
resolution = [H, W]

num_frames = 48

for prim_i in range(total_drawers_num):
    with open(os.path.join(bakedsdf_dir, "drawers", "results", f"drawer_{prim_i}.pkl"), 'rb') as f:
        prim_info = pickle.load(f)
        prim_transform = prim_info["transform"]
        interact_type = prim_info["interact"]
    
    scale, _, angles, trans, _ = transformations.decompose_matrix(prim_transform)
    prim_rotation = transformations.euler_matrix(axes='sxyz', *angles).reshape(4, 4)
    prim_translation = np.eye(4)
    prim_translation[:3, 3] = trans
    prim_rot_trans_original = prim_translation @ prim_rotation
    
    max_move_frame = max_move_frame_list[prim_i]
    
    trace_data_prim = trace_data[f"{prim_i:02d}"]
    
    if interact_type == "1.1":
        interior_depth = scale[1]*1.5
        r_yz = ((scale[1]) ** 2 + (scale[2]) ** 2) ** 0.5
        radius = 1.5 * r_yz
        _theta = -15 + 90
        theta = 90 - _theta
        
        target_y = 0.0
        target_z = 0.0
        camera_z = 0.0
        target_x = np.random.uniform(low=-interior_depth, high=0.2 * interior_depth)
        end_point_original = np.array([target_x, target_y, target_z]).reshape(3)
        
        camera_x = radius * np.cos(theta * np.pi / 180.0)
        camera_y = radius * np.sin(theta * np.pi / 180.0)  # - 0.5 * scale[1]
        cam_pos = np.array([camera_x, camera_y, camera_z]).reshape(3)
    
    elif interact_type == "1.2":
        interior_depth = scale[1]*1.5
        r_yz = ((scale[1]) ** 2 + (scale[2]) ** 2) ** 0.5
        radius = 1.5 * r_yz
        _theta = 15 + 90
        theta = 90 - _theta
        
        target_y = 0.0
        target_z = 0.0
        camera_z = 0.0
        target_x = np.random.uniform(low=-interior_depth, high=0.2 * interior_depth)
        end_point_original = np.array([target_x, target_y, target_z]).reshape(3)
        
        camera_x = radius * np.cos(theta * np.pi / 180.0)
        camera_y = radius * np.sin(theta * np.pi / 180.0)  # - 0.5 * scale[1]
        cam_pos = np.array([camera_x, camera_y, camera_z]).reshape(3)
        
        end_point_original[..., :2] *= -1
        cam_pos[..., :2] *= -1
    
    elif interact_type in ["2", "3.3"]:
        interior_depth = scale[1]*1.25
        
        yz_r = ((scale[1] * 0.5) ** 2 + (scale[2] * 0.5) ** 2) ** 0.5
        target_y = 0.0
        target_z = 0.0
        end_point_original = np.array([0, target_y, target_z]).reshape(3)
        
        camera_x = 2.0 * interior_depth
        camera_y = 0.0
        camera_z = 1.5 * scale[2]

        cam_pos = np.array([camera_x, camera_y, camera_z]).reshape(3)
    
    else:
        assert False, "not implemented"
        
    max_move_frame = max_move_frame_list[prim_i]

    end_point = point_transform(end_point_original, prim_rot_trans_original)
    cam_pos = point_transform(cam_pos, prim_rot_trans_original)
    
    R, tvec = look_at_view_transform(eye=cam_pos.reshape(1, 3), at=end_point.reshape(1, 3), up=((0, 0, 1),))
    znear = 0.01
    zfar = 1e10
    aspect_ratio = W / H
    fov_x = 2 * np.arctan(W / (2 * fx)) * (180.0 / np.pi)  # in degrees
    fov_y = 2 * np.arctan(H / (2 * fy)) * (180.0 / np.pi)  # in degrees
    fov = (fov_x + fov_y) / 2  # average the fovs
    camera_p3d = FoVPerspectiveCameras(
        znear=znear,
        zfar=zfar,
        aspect_ratio=aspect_ratio,
        fov=fov,
        degrees=True,
        R=R,
        T=tvec,
        device='cpu'
    )
    w2c = camera_p3d.get_world_to_view_transform().get_matrix().reshape(4, 4).T
    c2w = torch.inverse(w2c)
    c2w[:3, 0:2] *= -1
    c2w_gl = c2w.clone()
    c2w_gl[..., :3, 1:3] *= -1
        
    target_camera = Cameras(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        # distortion_params=distortion_params,
        height=H,
        width=W,
        camera_to_worlds=c2w_gl[..., :3, :4].float().reshape(1, 3, 4),
        camera_type=CameraType.PERSPECTIVE,
    ).to("cuda")
    
    rgb_list = []
    
    for image_i, _frame_i in enumerate(list(range(num_frames)) + list(reversed(list(range(num_frames))))):
        transform_indices_list = []
        transform_matrix_means_list = []
        transform_matrix_quats_list = []
        
        frame_i = _frame_i / num_frames * (1.5 * max_move_frame)
        frame_i = int(frame_i)
        frame_i = min(frame_i, total_frames - 1)
        
        trace_data_prim_transform_original = trace_data_prim[0]
        original_translation_matrix = np.eye(4)
        original_translation = trace_data_prim_transform_original["position"]
        original_translation_matrix[:3, 3] = np.array(original_translation).reshape(3)
        original_rotation_matrix = np.eye(4)
        original_rotation_matrix[:3, :3] = transformations.quaternion_matrix(
            trace_data_prim_transform_original["orientation"])[:3, :3]
        original_matrix = original_translation_matrix @ original_rotation_matrix
        original_matrix = torch.from_numpy(original_matrix).float().cuda()
        
        prim_frame_selected = frame_i
        prim_frame_selected_lower = int(prim_frame_selected)
        prim_frame_selected_upper = prim_frame_selected_lower + 1

        trace_data_prim_transform_lower_current = trace_data_prim[prim_frame_selected_lower]
        lower_current_translation_matrix = np.eye(4)
        lower_current_translation = trace_data_prim_transform_lower_current["position"]
        lower_current_translation_matrix[:3, 3] = np.array(lower_current_translation).reshape(3)
        lower_current_rotation_matrix = np.eye(4)
        lower_current_rotation_matrix[:3, :3] = transformations.quaternion_matrix(
            trace_data_prim_transform_lower_current["orientation"])[:3, :3]
        lower_current_matrix = lower_current_translation_matrix @ lower_current_rotation_matrix
        lower_current_matrix = torch.from_numpy(lower_current_matrix).float().cuda()

        trace_data_prim_transform_higher_current = trace_data_prim[prim_frame_selected_upper]
        higher_current_translation_matrix = np.eye(4)
        higher_current_translation = trace_data_prim_transform_higher_current["position"]
        higher_current_translation_matrix[:3, 3] = np.array(higher_current_translation).reshape(3)
        higher_current_rotation_matrix = np.eye(4)
        higher_current_rotation_matrix[:3, :3] = transformations.quaternion_matrix(
            trace_data_prim_transform_higher_current["orientation"])[:3, :3]
        higher_current_matrix = higher_current_translation_matrix @ higher_current_rotation_matrix
        higher_current_matrix = torch.from_numpy(higher_current_matrix).float().cuda()

        interp_ratio = prim_frame_selected - prim_frame_selected_lower

        # interpolate between lower_current_translation_matrix and higher_current_translation_matrix
        current_translation_matrix = lower_current_translation_matrix * (1 - interp_ratio) + higher_current_translation_matrix * interp_ratio

        # interpolate between lower_current_rotation_matrix and higher_current_rotation_matrix, by slerp
        rots = Rotation.from_matrix(np.stack([lower_current_rotation_matrix[:3, :3], higher_current_rotation_matrix[:3, :3]]))
        slerp = Slerp([0, 1], rots)
        current_rotation_matrix = np.eye(4)
        current_rotation_matrix[:3, :3] = slerp(interp_ratio).as_matrix()
        current_matrix = current_translation_matrix @ current_rotation_matrix

        current_matrix = torch.from_numpy(current_matrix).float().cuda()
        
        prim_means_indices_prim_i = torch.zeros((N_Gaussians), dtype=torch.bool)
        prim_means_indices_prim_i[prim_means_indices[prim_i]] = True
        if interact_type in ["2", "3.3"]:
            prim_means_indices_prim_i[
                append_indices_prims[prim_i].to(prim_means_indices_prim_i.device)] = True
        prim_means_indices_prim_i[
            append_indices_prims_door[prim_i].to(prim_means_indices_prim_i.device)] = True
        transform_indices_list.append(prim_means_indices_prim_i)
        transform_matrix_means_list.append((current_matrix @ torch.inverse(original_matrix)).clone())
        transform_matrix_quats_list.append((torch.from_numpy(
            current_rotation_matrix[:3, :3]).float().cuda() @ torch.inverse(
            torch.from_numpy(original_rotation_matrix[:3, :3]).float().cuda())).clone())

        model.articulate_transform = {
            "transform_indices_list": transform_indices_list,
            "transform_matrix_means_list": transform_matrix_means_list,
            "transform_matrix_quats_list": transform_matrix_quats_list,
        }
        output = model.get_outputs(target_camera.to("cuda"))
        rgb = np.clip(output["rgb"].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
        
        rgb_list.append(rgb)
        
    imageio.mimwrite(os.path.join(save_dir, f"drawer_{prim_i}.mp4"), rgb_list, fps=24)


