from __future__ import annotations
import os
os.environ['HF_HOME'] = "/home/hongchix/scratch/"

from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup

from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType

import torch
import numpy as np
import pickle
import transformations

import imageio
from scipy.spatial.transform import Slerp, Rotation
from tqdm import tqdm

from PIL import Image

import sys
sys.path.append("./scripts")

def slerp_cameras(c2w_0, c2w_1, intrinsic, image_size, ratio):
    rots = Rotation.from_matrix(np.stack([c2w_0[:3, :3], c2w_1[:3, :3]]))
    slerp = Slerp([0, 1], rots)
    # ratio = random.random()
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = slerp(ratio).as_matrix()
    c2w[:3, 3] = (1 - ratio) * c2w_0[:3, 3] + ratio * c2w_1[:3, 3]
    c2w = torch.from_numpy(c2w).float().unsqueeze(0)
    fx, fy, cx, cy = intrinsic
    width, height = image_size
    return Cameras(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        height=height,
        width=width,
        camera_to_worlds=c2w[:, :3, :4],
        camera_type=CameraType.PERSPECTIVE,
    )

load_config = "outputs/meshgauss/241005_cs_kitchen_splatfacto_on_mesh_bakedsdf_sdfstudio_normal_mono_depth_mono/splatfacto_on_mesh_uc/2024-11-08_234546/config.yml"
save_note = "default_grid_v3"
source_dir = "/u/hongchix/data/cs_kitchen_n/images_2"

# save_dir = "./vis/cs_kitchen/open/"
# os.makedirs(save_dir, exist_ok=True)

# os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

config, pipeline, checkpoint_path, _ = eval_setup(Path(load_config))
model = pipeline.model
model = model.eval()

source_image_names = [os.path.splitext(os.path.basename(str(name)))[0] for name in sorted(os.listdir(source_dir))]

image_filenames = pipeline.datamanager.train_dataset._dataparser_outputs.image_filenames
train_image_filenames = [os.path.splitext(os.path.basename(str(elem)))[0] for elem in image_filenames]

image_filenames = pipeline.datamanager.eval_dataset._dataparser_outputs.image_filenames
eval_image_filenames = [os.path.splitext(os.path.basename(str(elem)))[0] for elem in image_filenames]


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

frame_seq = list(range(total_frames))

open_snippet_frame = [
    {"frame": 2509, "drawer_idx": 4},
    {"frame": 2509, "drawer_idx": 3},
    {"frame": 2509, "drawer_idx": 2},
    {"frame": 2509, "drawer_idx": 1},
    {"frame": 2509, "drawer_idx": 0},
    {"frame": 2509, "drawer_idx": 6},
    {"frame": 2520, "drawer_idx": 8},
    {"frame": 2520, "drawer_idx": 7},
    {"frame": 2520, "drawer_idx": 12},
    {"frame": 2520, "drawer_idx": 11},
    {"frame": 2572, "drawer_idx": 9},
    {"frame": 2572, "drawer_idx": 10},
    {"frame": 2587, "drawer_idx": 15},
    {"frame": 2599, "drawer_idx": 5},
]


with torch.no_grad():
    for _drawer_i in range(len(open_snippet_frame)):

        # if _drawer_i not in [0, 1, 2, 3,4,5]:
        #     continue

        rgb_list = []
        drawer_dict = open_snippet_frame[_drawer_i]
        source_image_name = f"frame_{drawer_dict['frame']:0>5d}"
        vis_drawer_i = drawer_dict['drawer_idx']

        save_dir = f"./vis/cs_kitchen/open_videos_grid/drawer_{_drawer_i:0>2d}/"
        os.makedirs(save_dir, exist_ok=True)

        target_image = source_image_name
        if target_image in train_image_filenames:
            target_dataset = pipeline.datamanager.train_dataset
            target_idx = train_image_filenames.index(target_image)
            print("locate in train dataset with index ", target_idx)
        elif target_image in eval_image_filenames:
            target_dataset = pipeline.datamanager.eval_dataset
            target_idx = eval_image_filenames.index(target_image)
            print("locate in eval dataset with index ", target_idx)
        else:
            assert False

        target_cam = target_dataset.cameras[target_idx:target_idx + 1].to("cuda")

        target_cam.cx *= (1.2)
        target_cam.cy *= (1.2)
        target_cam.width = (target_cam.width * (1.2)).int()
        target_cam.height = (target_cam.height * (1.2)).int()

        target_cam.fx *= 1.0
        target_cam.fy *= 1.0

        for prim_frame in tqdm(frame_seq):

            transform_indices_list = []
            transform_matrix_means_list = []
            transform_matrix_quats_list = []
            prim_i = vis_drawer_i
            prim_mark = f"{prim_i:0>2d}"
            trace_data_prim = trace_data[prim_mark]
            interact_type = interact_type_list[prim_i]

            trace_data_prim_transform_original = trace_data_prim[0]
            original_translation_matrix = np.eye(4)
            original_translation = trace_data_prim_transform_original["position"]
            original_translation_matrix[:3, 3] = np.array(original_translation).reshape(3)
            original_rotation_matrix = np.eye(4)
            original_rotation_matrix[:3, :3] = transformations.quaternion_matrix(
                trace_data_prim_transform_original["orientation"])[:3, :3]
            original_matrix = original_translation_matrix @ original_rotation_matrix
            original_matrix = torch.from_numpy(original_matrix).float().cuda()

            trace_data_prim_transform_current = trace_data_prim[prim_frame]
            current_translation_matrix = np.eye(4)
            current_translation = trace_data_prim_transform_current["position"]
            current_translation_matrix[:3, 3] = np.array(current_translation).reshape(3)
            current_rotation_matrix = np.eye(4)
            current_rotation_matrix[:3, :3] = transformations.quaternion_matrix(
                trace_data_prim_transform_current["orientation"])[:3, :3]
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
            output = model.get_outputs(target_cam)
            rgb = np.clip(output["rgb"].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
            Image.fromarray(rgb).save(os.path.join(save_dir, f"open_{prim_frame:0>4d}.png"))
            # rgb_list.append(rgb)
        # imageio.mimwrite(output_video_path, rgb_list, fps=24)

