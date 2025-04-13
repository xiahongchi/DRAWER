from __future__ import annotations
import os

import cv2

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tyro

from nerfstudio.utils.eval_utils import eval_setup, resume_setup_gs_uc
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

glctx = dr.RasterizeGLContext(output_db=False)
psnr = PeakSignalNoiseRatio(data_range=1.0).to("cuda")

def filter_mesh_from_vertices(keep, mesh_points, faces):
    filter_mapping = np.arange(keep.shape[0])[keep]
    filter_unmapping = -np.ones((keep.shape[0]))
    filter_unmapping[filter_mapping] = np.arange(filter_mapping.shape[0])
    mesh_points = mesh_points[keep]
    keep_0 = keep[faces[:, 0]]
    keep_1 = keep[faces[:, 1]]
    keep_2 = keep[faces[:, 2]]
    keep_faces = np.logical_and(keep_0, keep_1)
    keep_faces = np.logical_and(keep_faces, keep_2)
    faces = faces[keep_faces]
    face_mapping = np.arange(keep_faces.shape[0])[keep_faces]
    faces[:, 0] = filter_unmapping[faces[:, 0]]
    faces[:, 1] = filter_unmapping[faces[:, 1]]
    faces[:, 2] = filter_unmapping[faces[:, 2]]
    return mesh_points, faces, face_mapping

def filter_faces_from_verts(keep, faces):

    keep_0 = keep[faces[:, 0]]
    keep_1 = keep[faces[:, 1]]
    keep_2 = keep[faces[:, 2]]
    keep_faces = torch.logical_and(keep_0, keep_1)
    keep_faces = torch.logical_and(keep_faces, keep_2)

    return keep_faces

def filter_faces_from_verts_strict(keep, faces):

    keep_0 = keep[faces[:, 0]]
    keep_1 = keep[faces[:, 1]]
    keep_2 = keep[faces[:, 2]]
    keep_faces = torch.logical_or(keep_0, keep_1)
    keep_faces = torch.logical_or(keep_faces, keep_2)

    return keep_faces


def point_transform(point, transform):
    point_shape = point.shape
    point_pad = np.pad(point.reshape(-1, 3), ((0, 0), (0, 1)), constant_values=(0, 1))
    point_transformed_pad = point_pad @ transform.T
    point_transformed = point_transformed_pad[:, :3] / point_transformed_pad[:, 3:]
    return point_transformed.reshape(point_shape)

def get_relative_depth_map(depth_buf, pad_value=10):
    absolute_depth = depth_buf  # B, H, W
    no_depth = torch.logical_or(depth_buf > 1e4, depth_buf < 0)
    valid_depth = torch.logical_not(no_depth)
    if not torch.any(valid_depth):
        return None
    depth_min, depth_max = absolute_depth[valid_depth].min(), absolute_depth[
        valid_depth].max()
    target_min, target_max = 50, 255

    depth_value = absolute_depth[valid_depth]
    depth_value = depth_max - depth_value  # reverse values

    depth_value /= (depth_max - depth_min)
    depth_value = depth_value * (target_max - target_min) + target_min

    relative_depth = absolute_depth.clone()
    relative_depth[valid_depth] = depth_value
    relative_depth[no_depth] = pad_value  # not completely black

    return relative_depth

def depth_blend(renderer, mesh_list, resolution):
    H, W = resolution
    pix_to_face_list = []
    bary_coords_list = []
    for mesh_i, mesh in enumerate(mesh_list):
        image_i, frags = renderer(mesh)
        image_i = image_i[..., :3].reshape(H, W, 3)
        depth_i = frags.zbuf.reshape(H, W)
        depth_i[depth_i < 0] = 1e5 + mesh_i
        pix_to_face_i = frags.pix_to_face.reshape(H, W).clone()
        bary_coords_i = frags.bary_coords.reshape(H, W, 3).clone()
        pix_to_face_list.append(pix_to_face_i)
        bary_coords_list.append(bary_coords_i)
        if mesh_i == 0:
            depth_buf = depth_i
            image = image_i
            idx_buf = torch.zeros((H, W), dtype=torch.int64).to(depth_i.device)
            idx_buf[depth_i >= 0] = 1

        else:
            image = torch.where(depth_i.unsqueeze(-1).expand(-1, -1, 3) < depth_buf.unsqueeze(-1).expand(-1, -1, 3), image_i, image)
            idx_buf = torch.where(depth_i < depth_buf, torch.ones((H, W), dtype=torch.int64).to(depth_i.device) * (mesh_i + 1), idx_buf)

            depth_buf = torch.where(depth_i < depth_buf, depth_i, depth_buf)
    return {
        "rgb": image,
        "depth": depth_buf,
        "idx": idx_buf,
        "pix_to_face": pix_to_face_list,
        "bary_coords": bary_coords_list,
        "depth_map": get_relative_depth_map(depth_buf),
    }


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    height, width = image.shape[:2]
    # print("height: ", height)
    # print("width: ", width)

    scaling_factor = 1.0 / d
    # weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    # return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)
    height_new = int(torch.floor(0.5 + (torch.tensor([float(height)]) * scaling_factor)).to(torch.int64))
    width_new = int(torch.floor(0.5 + (torch.tensor([float(width)]) * scaling_factor)).to(torch.int64))

    # print("height_new: ", height_new)
    # print("width_new: ", width_new)

    image_resized = F_V.resize(image.permute(2, 0, 1), [height_new, width_new])
    image_resized = image_resized.permute(1, 2, 0)

    return image_resized

def get_cc_verts(mesh):
    adjacency = mesh.edges

    # Create a graph from the adjacency list
    G = nx.Graph()
    G.add_edges_from(adjacency)

    # Find all connected components in the graph
    connected_components = nx.connected_components(G)
    connected_components = [list(cc) for cc in connected_components]
    connected_components = [cc for cc in connected_components if len(cc) > 100]

    return connected_components

def find_connected_component_verts(connected_components, vert_index):

    for comp in connected_components:
        if vert_index in list(comp):
            return np.array(list(comp), dtype=np.int32)
    return None

def get_centroids(mesh):
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    centroids = verts[faces.reshape(-1)].reshape(-1, 3, 3).mean(dim=1)
    return centroids

parser = argparse.ArgumentParser()
parser.add_argument("--splat_dir", type=str, required=True)
parser.add_argument("--sdf_dir", type=str, required=True)
parser.add_argument("--interior_dir", type=str, required=True)
parser.add_argument("--save_note", type=str, required=True)
parser.add_argument("--total_iterations", type=int, default=15000)
parser.add_argument("--training_with_segmented_objects", action="store_true")
args = parser.parse_args()

load_config = os.path.join(args.splat_dir, "config.yml")
bakedsdf_dir = args.sdf_dir
matfuse_dir = args.interior_dir
save_note = args.save_note
total_iterations = args.total_iterations
training_with_segmented_objects = args.training_with_segmented_objects

# only use this when you don't want that door to be considered
skip_door_indices = []

# universal hyperparameters
door_side_ratio = 0.5
mask_dilate_n1 = 0
mask_dilate_n2 = 0
door_back_distance_means = 0.05
data_scale = 0.00

if training_with_segmented_objects:
    separate_mesh_dir = os.path.join(bakedsdf_dir, "separate/texture_mesh/")
    objects_faces_list_path = os.path.join(separate_mesh_dir, "combined.pkl")

    with open(objects_faces_list_path, 'rb') as f:
        objects_faces_list = pickle.load(f)
    for objects_faces in objects_faces_list:
        print("objects_faces: ", objects_faces.shape)

    object_cnt = len(objects_faces_list) - 1
    object_i_tex_mesh_list = []
    for object_i in range(object_cnt):
        object_i_mesh_path = os.path.join(separate_mesh_dir, f"mesh-box_{object_i}_connected.obj")
        object_i_tex_mesh = load_objs_as_meshes([object_i_mesh_path], device="cuda")
        object_i_tex_mesh_list.append(object_i_tex_mesh)

    main_mesh_path = os.path.join(separate_mesh_dir, "lama_inpaint/main_tex_mesh_inpainted.obj")
    if not os.path.exists(main_mesh_path):
        main_mesh_path = os.path.join(separate_mesh_dir, f"mesh_inpainted.obj")
    print("main_mesh_path: ", main_mesh_path)
    main_tex_mesh = load_objs_as_meshes([main_mesh_path], device="cuda")

    config, pipeline, checkpoint_path, _, optimizer = resume_setup_gs_uc(Path(load_config))
    model = pipeline.model

    model_mesh_verts = model.mesh_verts.cpu().numpy()
    model_mesh_faces = model.mesh_faces.cpu().numpy()
    model_mesh = trimesh.Trimesh(model_mesh_verts, model_mesh_faces, process=False)
    model_mesh_cc = get_cc_verts(model_mesh)
    gaussians_to_mesh_indices = model.gaussians_to_mesh_indices.long()
    gaussian_in_object_indices_list = []
    gaussian_all_mask = torch.ones(gaussians_to_mesh_indices.shape[0], device="cuda") > 0
    faces_all_mask = torch.ones(model_mesh_faces.shape[0], device="cuda") > 0
    # the last one is main mesh
    object_i_sub_mesh_list = []
    for object_i, objects_faces in enumerate(objects_faces_list[:-1]):
        object_verts = find_connected_component_verts(model_mesh_cc, objects_faces[0, 0])
        keep_verts = np.zeros(model_mesh_verts.shape[0]) > 0
        keep_verts[object_verts] = True
        object_verts, object_faces, face_mapping = filter_mesh_from_vertices(keep_verts, model_mesh_verts, model_mesh_faces)
        # trimesh.exchange.export.export_mesh(
        #     trimesh.Trimesh(
        #         object_verts,
        #         object_faces,
        #     ),
        #     os.path.join(separate_mesh_dir, f"object_{object_i}_sub_mesh.ply")
        # )

        # object_faces_centroids = object_verts[object_faces.reshape(-1)].reshape(-1, 3, 3).mean(axis=1)
        object_i_sub_mesh_list.append(
            Meshes(
                [torch.from_numpy(object_verts).cuda()],
                [torch.from_numpy(object_faces).cuda()],
            )
        )
        object_faces = face_mapping
        object_faces = torch.from_numpy(object_faces).long().cuda()
        faces_all_mask[object_faces] = False
        gaussian_in_object = torch.isin(gaussians_to_mesh_indices, object_faces)
        # print("gaussian_in_object: ", torch.count_nonzero(gaussian_in_object), gaussian_in_object.shape)
        gaussian_all_mask[gaussian_in_object] = False
        gaussian_in_object_indices_list.append(torch.arange(gaussian_in_object.shape[0], device="cuda")[gaussian_in_object])
    # print("gaussian_all_mask: ", torch.count_nonzero(gaussian_all_mask), gaussian_all_mask.shape)
    gaussian_main_indices = torch.arange(gaussian_all_mask.shape[0], device="cuda")[gaussian_all_mask]
    # print("gaussian_main_indices: ", gaussian_main_indices.shape)

    main_sub_mesh_faces_indices = torch.arange(faces_all_mask.shape[0], device="cuda")[faces_all_mask]
    main_sub_mesh = Meshes(
        [model.mesh_verts.cuda()],
        [model.mesh_faces.cuda()],
    ).submeshes([[main_sub_mesh_faces_indices]])
    # trimesh.exchange.export.export_mesh(
    #     trimesh.Trimesh(
    #         main_sub_mesh.verts_packed().cpu().numpy(),
    #         main_sub_mesh.faces_packed().cpu().numpy(),
    #     ),
    #     os.path.join(separate_mesh_dir, f"main_sub_mesh.ply")
    # )
    # object_i_tex_mesh_list
    # main_tex_mesh

    # object_i_sub_mesh_list
    # main_sub_mesh

    # gaussian_in_object_indices_list
    # gaussian_main_indices

    # find overlapping faces idx
    close_dist = 0.006

    main_tex_mesh_centroids = get_centroids(main_tex_mesh).cpu().numpy()
    knn_main = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(main_tex_mesh_centroids)
    main_tex_mesh_overlap_faces = []
    object_tex_mesh_overlap_faces = []
    for object_i, object_i_tex_mesh in enumerate(object_i_tex_mesh_list):
        object_i_tex_mesh_centroids = get_centroids(object_i_tex_mesh).cpu().numpy()
        dists, indices = knn_main.kneighbors(object_i_tex_mesh_centroids)
        dists = dists.reshape(-1)
        indices = indices.reshape(-1)
        object_overlap_faces = torch.from_numpy(np.arange(indices.shape[0])[dists < close_dist]).long().cuda()
        object_tex_mesh_overlap_faces.append(object_overlap_faces)

        # object_i_tex_mesh_mask_colors = torch.ones(object_i_tex_mesh_centroids.shape[0], 3, device="cuda")
        # object_i_tex_mesh_mask_colors[object_overlap_faces] = 0.
        # trimesh.exchange.export.export_mesh(
        #     trimesh.Trimesh(
        #         object_i_tex_mesh.verts_packed().cpu().numpy(),
        #         object_i_tex_mesh.faces_packed().cpu().numpy(),
        #         face_colors=object_i_tex_mesh_mask_colors.cpu().numpy(),
        #     ),
        #     os.path.join(separate_mesh_dir, f"object_{object_i}_tex_mesh_mask.ply")
        # )

        knn_object = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(object_i_tex_mesh_centroids)
        dists, indices = knn_object.kneighbors(main_tex_mesh_centroids)
        dists = dists.reshape(-1)
        indices = indices.reshape(-1)
        main_overlap_faces = torch.from_numpy(np.arange(indices.shape[0])[dists < close_dist]).long().cuda()
        main_tex_mesh_overlap_faces.append(main_overlap_faces)
    main_tex_mesh_overlap_faces = torch.cat(main_tex_mesh_overlap_faces, dim=0)

    # main_tex_mesh_mask_colors = torch.ones(main_tex_mesh_centroids.shape[0], 3, device="cuda")
    # main_tex_mesh_mask_colors[main_tex_mesh_overlap_faces] = 0.
    # trimesh.exchange.export.export_mesh(
    #     trimesh.Trimesh(
    #         main_tex_mesh.verts_packed().cpu().numpy(),
    #         main_tex_mesh.faces_packed().cpu().numpy(),
    #         face_colors=main_tex_mesh_mask_colors.cpu().numpy(),
    #     ),
    #     os.path.join(separate_mesh_dir, f"main_tex_mesh_mask.ply")
    # )

    main_sub_mesh_centroids = get_centroids(main_sub_mesh).cpu().numpy()
    knn_main = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(main_sub_mesh_centroids)
    main_sub_mesh_overlap_faces = []
    object_sub_mesh_overlap_faces = []
    for object_i, object_i_sub_mesh in enumerate(object_i_sub_mesh_list):
        object_i_sub_mesh_centroids = get_centroids(object_i_sub_mesh).cpu().numpy()
        dists, indices = knn_main.kneighbors(object_i_sub_mesh_centroids)
        dists = dists.reshape(-1)
        indices = indices.reshape(-1)
        object_overlap_faces = torch.from_numpy(np.arange(indices.shape[0])[dists < close_dist]).long().cuda()
        object_sub_mesh_overlap_faces.append(object_overlap_faces)

        # object_i_sub_mesh_mask_colors = torch.ones(object_i_sub_mesh_centroids.shape[0], 3, device="cuda")
        # object_i_sub_mesh_mask_colors[object_overlap_faces] = 0.
        # trimesh.exchange.export.export_mesh(
        #     trimesh.Trimesh(
        #         object_i_sub_mesh.verts_packed().cpu().numpy(),
        #         object_i_sub_mesh.faces_packed().cpu().numpy(),
        #         face_colors=object_i_sub_mesh_mask_colors.cpu().numpy()
        #     ),
        #     os.path.join(separate_mesh_dir, f"object_{object_i}_sub_mesh_mask.ply")
        # )

        knn_object = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(object_i_sub_mesh_centroids)
        dists, indices = knn_object.kneighbors(main_sub_mesh_centroids)
        dists = dists.reshape(-1)
        indices = indices.reshape(-1)
        main_overlap_faces = torch.from_numpy(np.arange(indices.shape[0])[dists < close_dist]).long().cuda()
        main_sub_mesh_overlap_faces.append(main_overlap_faces)
    main_sub_mesh_overlap_faces = torch.cat(main_sub_mesh_overlap_faces, dim=0)

    # main_sub_mesh_mask_colors = torch.ones(main_sub_mesh_centroids.shape[0], 3, device="cuda")
    # main_sub_mesh_mask_colors[main_sub_mesh_overlap_faces] = 0.
    # trimesh.exchange.export.export_mesh(
    #     trimesh.Trimesh(
    #         main_sub_mesh.verts_packed().cpu().numpy(),
    #         main_sub_mesh.faces_packed().cpu().numpy(),
    #         face_colors=main_sub_mesh_mask_colors.cpu().numpy()
    #     ),
    #     os.path.join(separate_mesh_dir, f"main_sub_mesh_mask.ply")
    # )



    door_depth = 0.2
    # tmp_save_dir = "./vis/cs_kitchen/gs_internal/"
    # os.makedirs(tmp_save_dir, exist_ok=True)
    save_dir = os.path.join(os.path.dirname(load_config), save_note)
    os.makedirs(save_dir, exist_ok=True)

    save_dir_doors = os.path.join(save_dir, "doors")
    save_dir_boxes = os.path.join(save_dir, "boxes")
    save_dir_remaining = os.path.join(save_dir, "remaining")
    os.makedirs(save_dir_doors, exist_ok=True)
    os.makedirs(save_dir_boxes, exist_ok=True)
    os.makedirs(save_dir_remaining, exist_ok=True)

    textured_mesh_path = os.path.join(bakedsdf_dir, "separate/texture_mesh/lama_inpaint/main_tex_mesh_inpainted.obj")
    # texture_image_path = os.path.join(bakedsdf_dir, "separate/texture_mesh/lama_inpaint/main_tex_mesh_inpainted.png")
    if not os.path.exists(textured_mesh_path):
        textured_mesh_path = os.path.join(bakedsdf_dir, 'texture_mesh/mesh-simplify.obj')
        # texture_image_path = os.path.join(bakedsdf_dir, 'texture_mesh/mesh-simplify.png')

    model = model.train()
    N_Gaussians_original = model.means.shape[0]

    image_filenames = pipeline.datamanager.train_dataset._dataparser_outputs.image_filenames
    train_image_filenames = [os.path.splitext(os.path.basename(str(elem)))[0] for elem in image_filenames]

    image_filenames = pipeline.datamanager.eval_dataset._dataparser_outputs.image_filenames
    eval_image_filenames = [os.path.splitext(os.path.basename(str(elem)))[0] for elem in image_filenames]

    # target_image = "DSC04672"
    # if target_image in train_image_filenames:
    #     target_dataset = pipeline.datamanager.train_dataset
    #     target_idx = train_image_filenames.index(target_image)
    #     print("locate in train dataset with index ", target_idx)
    # elif target_image in eval_image_filenames:
    #     target_dataset = pipeline.datamanager.eval_dataset
    #     target_idx = eval_image_filenames.index(target_image)
    #     print("locate in eval dataset with index ", target_idx)
    # else:
    #     assert False

    # target_camera = target_dataset.cameras[target_idx:target_idx+1].to("cuda")
    # output = model.get_outputs()
    # rgb = np.clip(output["rgb"].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
    # Image.fromarray(rgb).save("./vis/rgb.png")

    with open(os.path.join(bakedsdf_dir, "drawers", "trace.json"), 'r') as f:
        trace_data = json.load(f)

    drawer_files = [name for name in os.listdir(os.path.join(bakedsdf_dir, "drawers", "results")) if name.endswith(".ply")]
    total_drawers_num = len(drawer_files)

    total_frames = len(trace_data["00"])

    # means_pad = np.pad(model.means.detach().cpu().numpy(), ((0, 0), (0, 1)), constant_values=(0, 1))
    # prim_means_indices = []
    # interact_type_list = []
    # for prim_i in range(total_drawers_num):
    #     with open(os.path.join(bakedsdf_dir, "drawers", "results", f"drawer_{prim_i}.pkl"), 'rb') as f:
    #         prim_info = pickle.load(f)
    #         prim_transform = prim_info["transform"]
    #         interact_type = prim_info["interact"]
    #     interact_type_list.append(interact_type)
    #     means_pad_transformed = means_pad @ np.linalg.inv(prim_transform).T
    #     means_transformed = means_pad_transformed[:, :3] / means_pad_transformed[:, 3:]
    #     scale_limit = np.array([1e4 * door_depth, 1, 1]).reshape(1, 3)
    #     means_transformed = means_transformed / scale_limit
    #
    #     # inside_yz = (np.abs(means_transformed[:, 1:]) < 0.5)
    #     # if interact_type in ["1.1", "2", "3.3"]:
    #     #     inside_x = np.logical_and(means_transformed[:, 0:1] > -door_depth * 0.2,
    #     #                               means_transformed[:, 0:1] < door_depth)
    #     # else:
    #     #     inside_x = np.logical_and(means_transformed[:, 0:1] < door_depth * 0.2,
    #     #                               means_transformed[:, 0:1] > -door_depth)
    #     print("means interact_type: ", interact_type)
    #
    #     inside_y = (np.abs(means_transformed[:, 1:2]) < 0.5)
    #     # inside_z = (np.abs(means_transformed[:, 2:]) < 0.5)
    #     inside_z = np.logical_and(
    #         means_transformed[:, 2:] < 0.5,
    #         means_transformed[:, 2:] > -0.5
    #     )
    #     if interact_type in ["1.1"]:
    #         inside_x = np.logical_and(
    #             means_transformed[:, 0:1] > -door_back_distance_means,
    #             means_transformed[:, 0:1] < 0.3)
    #         # inside_y = np.logical_and(
    #         #     means_transformed[:, 1:2] < 0.5,
    #         #     means_transformed[:, 1:2] > -0.4)
    #     elif interact_type in ["1.2"]:
    #         inside_x = np.logical_and(
    #             means_transformed[:, 0:1] < door_back_distance_means,
    #             means_transformed[:, 0:1] > -0.3)
    #         # inside_y = np.logical_and(
    #         #     means_transformed[:, 1:2] < 0.4,
    #         #     means_transformed[:, 1:2] > -0.5)
    #     else:
    #         inside_x = np.logical_and(
    #             means_transformed[:, 0:1] > -door_back_distance_means,
    #             means_transformed[:, 0:1] < 0.3)
    #         # inside_y = np.logical_and(
    #         #     means_transformed[:, 1:2] < 0.5,
    #         #     means_transformed[:, 1:2] > -0.4)
    #
    #     inside = np.concatenate([inside_x, inside_y, inside_z], axis=-1)
    #     prim_i_means_indices = np.all(inside, axis=1).reshape(-1)

        # given indices, find all gs points in the same face, then add them all

    faces_means = torch.mean(model.mesh_faces_verts, dim=1).reshape(-1, 3).cpu().numpy()
    means_pad = np.pad(faces_means, ((0, 0), (0, 1)), constant_values=(0, 1))
    prim_means_indices = []
    interact_type_list = []
    for prim_i in range(total_drawers_num):
        with open(os.path.join(bakedsdf_dir, "drawers", "results", f"drawer_{prim_i}.pkl"), 'rb') as f:
            prim_info = pickle.load(f)
            prim_transform = prim_info["transform"]
            interact_type = prim_info["interact"]
        interact_type_list.append(interact_type)

        means_pad_transformed = means_pad @ np.linalg.inv(prim_transform).T
        means_transformed = means_pad_transformed[:, :3] / means_pad_transformed[:, 3:]
        scale_limit = np.array([1e4 * door_depth, 1, 1]).reshape(1, 3)
        means_transformed = means_transformed / scale_limit

        # inside_yz = (np.abs(means_transformed[:, 1:]) < 0.5)
        # if interact_type in ["1.1", "2", "3.3"]:
        #     inside_x = np.logical_and(means_transformed[:, 0:1] > -door_depth * 0.2,
        #                               means_transformed[:, 0:1] < door_depth)
        # else:
        #     inside_x = np.logical_and(means_transformed[:, 0:1] < door_depth * 0.2,
        #                               means_transformed[:, 0:1] > -door_depth)
        print("means interact_type: ", interact_type)

        inside_y = (np.abs(means_transformed[:, 1:2]) < 0.5)
        # inside_z = (np.abs(means_transformed[:, 2:]) < 0.5)
        inside_z = np.logical_and(
            means_transformed[:, 2:] < 0.5,
            means_transformed[:, 2:] > -0.5
        )
        if interact_type in ["1.1"]:
            inside_x = np.logical_and(
                means_transformed[:, 0:1] > -door_back_distance_means,
                means_transformed[:, 0:1] < 0.3)

        elif interact_type in ["1.2"]:
            inside_x = np.logical_and(
                means_transformed[:, 0:1] < door_back_distance_means,
                means_transformed[:, 0:1] > -0.3)

        else:
            inside_x = np.logical_and(
                means_transformed[:, 0:1] > -door_back_distance_means,
                means_transformed[:, 0:1] < 0.3)
        inside = np.concatenate([inside_x, inside_y, inside_z], axis=-1)
        prim_i_faces_indices = np.all(inside, axis=1).reshape(-1)

        prim_i_faces_indices = np.logical_or(prim_i_faces_indices, np.all(np.abs(means_transformed) < 0.3, axis=-1).reshape(-1))

        prim_i_faces_indices = np.logical_and(prim_i_faces_indices, np.logical_not(
            np.all(np.concatenate([
                np.logical_and(np.abs(means_transformed[:, 0:1]) > door_back_distance_means,
                               np.abs(means_transformed[:, 0:1]) < 0.3) if interact_type in ["1.1", "2", "3.3"] else
                np.logical_and(np.abs(means_transformed[:, 0:1]) < -door_back_distance_means,
                               np.abs(means_transformed[:, 0:1]) > -0.3),
                np.abs(means_transformed[:, 1:2]) < 0.5,
                np.abs(means_transformed[:, 2:3]) > 0.45
            ], axis=-1).reshape(-1, 3), axis=-1)
        ))


        prim_i_faces_indices = torch.nonzero(torch.from_numpy(prim_i_faces_indices).cuda()).reshape(-1)
        prim_i_means_indices = torch.isin(model.gaussians_to_mesh_indices, prim_i_faces_indices)


        for prim_j in range(prim_i):
            prim_j_means_indices = prim_means_indices[prim_j]
            prim_i_means_indices = torch.logical_and(prim_i_means_indices, torch.logical_not(prim_j_means_indices))
        print(f"prim_i_means_indices {prim_i}: {torch.count_nonzero(prim_i_means_indices)}")
        prim_means_indices.append(prim_i_means_indices)


    prim_means_indices = [torch.nonzero(elem) for elem in prim_means_indices]
    print("total_frames: ", total_frames)



    textured_mesh = load_objs_as_meshes([textured_mesh_path], device='cuda')

    textured_mesh_verts = textured_mesh.verts_packed()
    textured_mesh_faces = textured_mesh.faces_packed()
    full_mesh_verts_pad = np.pad(textured_mesh_verts.cpu().numpy(), ((0, 0), (0, 1)), constant_values=(0, 1))

    mesh_door_indices = []
    for prim_i in range(total_drawers_num):
        with open(os.path.join(bakedsdf_dir, "drawers", "results", f"drawer_{prim_i}.pkl"), 'rb') as f:
            prim_transform = pickle.load(f)
            interact_type = prim_transform["interact"]
            prim_transform = prim_transform["transform"]
        means_pad_transformed = full_mesh_verts_pad @ np.linalg.inv(prim_transform).T
        means_transformed = means_pad_transformed[:, :3] / means_pad_transformed[:, 3:]
        scale_limit = np.array([1e4 * door_depth, 1, 1]).reshape(1, 3)
        means_transformed = means_transformed / scale_limit

        # inside_yz = (np.abs(means_transformed[:, 1:]) < 0.5)
        # if interact_type in ["1.1", "2", "3.3"]:
        #     inside_x = np.logical_and(means_transformed[:, 0:1] > -door_depth * 0.2,
        #                               means_transformed[:, 0:1] < door_depth)
        # else:
        #     inside_x = np.logical_and(means_transformed[:, 0:1] < door_depth * 0.2,
        #                               means_transformed[:, 0:1] > -door_depth)
        print("door 0 interact_type: ", interact_type)
        inside_y = (np.abs(means_transformed[:, 1:2]) < door_side_ratio)
        inside_z = np.logical_and(
            means_transformed[:, 2:] < door_side_ratio,
            means_transformed[:, 2:] > -door_side_ratio
        )
        if interact_type in ["1.1"]:
            inside_x = np.logical_and(
                means_transformed[:, 0:1] > -0.05,
                means_transformed[:, 0:1] < 0.3)
            # inside_y = np.logical_and(
            #     means_transformed[:, 1:2] < 0.5,
            #     means_transformed[:, 1:2] > -0.4)
        elif interact_type in ["1.2"]:
            inside_x = np.logical_and(
                means_transformed[:, 0:1] < 0.05,
                means_transformed[:, 0:1] > -0.3)
            # inside_y = np.logical_and(
            #     means_transformed[:, 1:2] < 0.4,
            #     means_transformed[:, 1:2] > -0.5)
        else:
            inside_x = np.logical_and(
                means_transformed[:, 0:1] > -0.05,
                means_transformed[:, 0:1] < 0.3)
            # inside_y = np.logical_and(
            #     means_transformed[:, 1:2] < 0.5,
            #     means_transformed[:, 1:2] > -0.4)

        inside = np.concatenate([inside_x, inside_y, inside_z], axis=-1)
        prim_i_means_indices = np.all(inside, axis=1).reshape(-1)

        prim_i_means_indices = torch.from_numpy(prim_i_means_indices).cuda()
        for prim_j in range(prim_i):
            prim_j_means_indices = mesh_door_indices[prim_j]
            # prim_i_means_indices = torch.logical_and(prim_i_means_indices, torch.logical_not(prim_j_means_indices))
        print(f"prim_i_means_indices {prim_i}: {torch.count_nonzero(prim_i_means_indices)}")
        mesh_door_indices.append(prim_i_means_indices)

    # for the base
    mesh_door_indices_base = []
    for prim_i in range(total_drawers_num):
        with open(os.path.join(bakedsdf_dir, "drawers", "results", f"drawer_{prim_i}.pkl"), 'rb') as f:
            prim_transform = pickle.load(f)
            interact_type = prim_transform["interact"]
            prim_transform = prim_transform["transform"]
        means_pad_transformed = full_mesh_verts_pad @ np.linalg.inv(prim_transform).T
        means_transformed = means_pad_transformed[:, :3] / means_pad_transformed[:, 3:]
        scale_limit = np.array([1e4 * door_depth, 1, 1]).reshape(1, 3)
        means_transformed = means_transformed / scale_limit

        # inside_yz = (np.abs(means_transformed[:, 1:]) < 0.5)
        # if interact_type in ["1.1", "2", "3.3"]:
        #     inside_x = np.logical_and(means_transformed[:, 0:1] > -door_depth * 0.2,
        #                               means_transformed[:, 0:1] < door_depth)
        # else:
        #     inside_x = np.logical_and(means_transformed[:, 0:1] < door_depth * 0.2,
        #                               means_transformed[:, 0:1] > -door_depth)
        print("door 0 interact_type: ", interact_type)

        inside_y = (np.abs(means_transformed[:, 1:2]) < 0.5)
        inside_z = np.logical_and(
            means_transformed[:, 2:] < 0.5,
            means_transformed[:, 2:] > -0.5
        )
        if interact_type in ["1.1"]:
            inside_x = np.logical_and(
                means_transformed[:, 0:1] > -0.05,
                means_transformed[:, 0:1] < 0.3)
            # inside_y = np.logical_and(
            #     means_transformed[:, 1:2] < 0.5,
            #     means_transformed[:, 1:2] > -0.4)
        elif interact_type in ["1.2"]:
            inside_x = np.logical_and(
                means_transformed[:, 0:1] < 0.05,
                means_transformed[:, 0:1] > -0.3)
            # inside_y = np.logical_and(
            #     means_transformed[:, 1:2] < 0.4,
            #     means_transformed[:, 1:2] > -0.5)
        else:
            inside_x = np.logical_and(
                means_transformed[:, 0:1] > -0.05,
                means_transformed[:, 0:1] < 0.3)
            # inside_y = np.logical_and(
            #     means_transformed[:, 1:2] < 0.5,
            #     means_transformed[:, 1:2] > -0.4)

        inside = np.concatenate([inside_x, inside_y, inside_z], axis=-1)
        prim_i_means_indices = np.all(inside, axis=1).reshape(-1)

        prim_i_means_indices = torch.from_numpy(prim_i_means_indices).cuda()
        # for prim_j in range(prim_i):
        #     prim_j_means_indices = mesh_door_indices_base[prim_j]
        # prim_i_means_indices = torch.logical_and(prim_i_means_indices, torch.logical_not(prim_j_means_indices))
        print(f"prim_i_means_indices {prim_i}: {torch.count_nonzero(prim_i_means_indices)}")
        mesh_door_indices_base.append(prim_i_means_indices)

    F_cnt = textured_mesh_faces.shape[0]
    remaining_face_indices = torch.ones(F_cnt) > 0
    for prim_i in range(total_drawers_num):
        door_keep_faces = filter_faces_from_verts(mesh_door_indices[prim_i].cpu(), textured_mesh_faces.cpu())
        remaining_face_indices[door_keep_faces] = False
    remaining_face_indices = torch.arange(F_cnt)[remaining_face_indices]
    remaining_mesh = textured_mesh.submeshes([[remaining_face_indices]])

    trimesh.exchange.export.export_mesh(
        trimesh.Trimesh(
            remaining_mesh.verts_packed().cpu().numpy(),
            remaining_mesh.faces_packed().cpu().numpy(),
            process=False
        ),
        os.path.join(save_dir_remaining, f"remaining.ply")
    )

    print("loading drawer meshes")
    append_indices_prims = []
    drawer_interior_mesh_p3d_list = []
    for prim_i in tqdm(range(total_drawers_num)):
        gs_drawer_mesh_path = f"/projects/illinois/eng/cs/shenlong/personals/hongchix/codes/ns_revival/vis/{matfuse_dir_name}/{prim_i:0>2d}/mesh_{prim_i:0>2d}.obj"
        drawer_interior_mesh_p3d = load_objs_as_meshes([gs_drawer_mesh_path], device='cuda')
        drawer_interior_mesh_p3d_list.append(drawer_interior_mesh_p3d)
        append_indices, append_gaussian_indices = model.append_from_mesh(gs_drawer_mesh_path)
        append_indices_prims.append(append_gaussian_indices)

    print("loading drawer door meshes")
    append_indices_prims_door = []
    drawer_interior_mesh_p3d_list_door = []
    for prim_i in tqdm(range(total_drawers_num)):
        gs_drawer_door_mesh_path = f"/projects/illinois/eng/cs/shenlong/personals/hongchix/codes/ns_revival/vis/{matfuse_dir_name}/{prim_i:0>2d}/mesh_door_{prim_i:0>2d}.obj"
        drawer_interior_mesh_door_p3d = load_objs_as_meshes([gs_drawer_door_mesh_path], device='cuda')
        drawer_interior_mesh_p3d_list_door.append(drawer_interior_mesh_door_p3d)
        append_indices_door, append_gaussian_indices_door = model.append_from_mesh(gs_drawer_door_mesh_path)
        append_indices_prims_door.append(append_gaussian_indices_door)


    # data gen
    znear = 0.01
    zfar = 1e10
    render_H = render_W = 512
    resolution = [render_H, render_W]
    aspect_ratio = 1.0
    fov = 80.0
    fx = fy = render_W / (2 * np.tan(fov * np.pi / 180. / 2))
    cx = render_H / 2
    cy = render_W / 2
    # for the main mesh
    # main_tex_mesh_overlap_faces
    # main_tex_mesh
    os.makedirs("./vis/cs_kitchen/gs_internal_uc_objs", exist_ok=True)
    with torch.no_grad():
        # for every object
        n_samples = 100
        objects_sample_data = []
        print("sampling data for objects...")
        for object_i, object_i_tex_mesh in enumerate(object_i_tex_mesh_list):
            object_i_tex_mesh_verts = object_i_tex_mesh.verts_packed()
            object_i_tex_mesh_faces = object_i_tex_mesh.faces_packed()

            overlap_faces_indices = object_tex_mesh_overlap_faces[object_i]
            gaussian_in_object_indices = gaussian_in_object_indices_list[object_i]

            verts_xyz_min = torch.min(object_i_tex_mesh_verts, dim=0)[0].reshape(1, 3)
            verts_xyz_max = torch.max(object_i_tex_mesh_verts, dim=0)[0].reshape(1, 3)

            # print("verts_xyz_min: ", verts_xyz_min)
            # print("verts_xyz_max: ", verts_xyz_max)

            verts_xyz_center = (verts_xyz_min + verts_xyz_max) * 0.5
            verts_xyz_scale = float(torch.max(verts_xyz_max - verts_xyz_min))

            # object_i_tex_mesh_verts = (object_i_tex_mesh_verts - verts_xyz_center) / verts_xyz_scale
            # object_i_tex_mesh_originated = object_i_tex_mesh.clone()
            # object_i_tex_mesh_originated.update_padded(object_i_tex_mesh_verts.unsqueeze(0))

            sample_i = 0
            sample_data = []
            while sample_i < n_samples:
                radius = 0.8
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                cam_pos = torch.from_numpy(np.array([
                    np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)
                ]) * radius).reshape(1, 3).to("cuda").float() * verts_xyz_scale + verts_xyz_center
                R, tvec = look_at_view_transform(eye=cam_pos.reshape(1, 3), at=verts_xyz_center.reshape(1, 3),
                                                 up=((0, 0, 1),))

                camera_p3d = FoVPerspectiveCameras(
                    znear=znear,
                    zfar=zfar,
                    aspect_ratio=aspect_ratio,
                    fov=fov,
                    degrees=True,
                    R=R,
                    T=tvec,
                    device='cuda'
                )

                raster_settings = RasterizationSettings(
                    image_size=resolution,
                    blur_radius=1e-10,
                    faces_per_pixel=1,
                    perspective_correct=False,
                    cull_backfaces=False
                )

                renderer = MeshRendererWithFragments(
                    rasterizer=MeshRasterizer(
                        cameras=camera_p3d,
                        raster_settings=raster_settings
                    ),
                    shader=HardPhongShader(device='cuda', cameras=camera_p3d, lights=AmbientLights(device='cuda'))
                )

                image, frag = renderer(object_i_tex_mesh)
                pix_to_face = frag.pix_to_face.reshape(render_H, render_W)
                valid = pix_to_face >= 0
                pix_to_face_valid = pix_to_face[valid]

                pix_to_face_overlap = torch.isin(pix_to_face_valid, overlap_faces_indices)

                inside_faces_cnt = torch.unique(pix_to_face_valid[pix_to_face_overlap]).shape[0]
                # if inside_faces_cnt >= 0.5 * overlap_faces_indices.shape[0]:
                if inside_faces_cnt >= 0.2 * overlap_faces_indices.shape[0]:
                    sample_i += 1
                    mask = torch.zeros(render_H, render_W, device="cuda")
                    mask_valid = mask[valid]
                    mask_valid[pix_to_face_overlap] = 1.0
                    mask[valid] = mask_valid

                    object_mask = valid

                    depth = frag.zbuf.reshape(render_H, render_W)
                    depth[torch.logical_not(object_mask > 0.)] = 0.

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
                        height=render_H,
                        width=render_W,
                        camera_to_worlds=c2w_gl[..., :3, :4].float().reshape(1, 3, 4),
                        camera_type=CameraType.PERSPECTIVE,
                    ).to("cpu")

                    sample_data.append({
                        "mesh_camera": camera_p3d.to("cpu"),
                        "rgb": image[..., :3].reshape(render_H, render_W, 3).cpu(),
                        "mask": (mask > 0.).cpu(),
                        "depth": depth.cpu(),
                        "gs_camera": target_camera,
                        "object_mask": object_mask.cpu()
                    })

                    # image_pil = Image.fromarray(np.clip(image[..., :3].reshape(render_H, render_W, 3).cpu().numpy() * 255., 0, 255).astype(np.uint8))
                    # image_pil.save(f"./vis/cs_kitchen/gs_internal_uc_objs/object_{object_i}_sample_{sample_i:0>2d}.png")
                    #
                    # mask_pil = Image.fromarray(np.clip(mask.cpu().numpy() * 255., 0, 255).astype(np.uint8))
                    # mask_pil.save(f"./vis/cs_kitchen/gs_internal_uc_objs/object_{object_i}_sample_{sample_i:0>2d}_mask.png")
                    #
                    # model.visible_gs_indices = gaussian_in_object_indices
                    # output = model.get_outputs(target_camera.to("cuda"))
                    # model.visible_gs_indices = None
                    #
                    # render_pil = Image.fromarray(np.clip(output["rgb"].reshape(render_H, render_W, 3).cpu().numpy() * 255., 0, 255).astype(np.uint8))
                    # render_pil.save(f"./vis/cs_kitchen/gs_internal_uc_objs/object_{object_i}_sample_{sample_i:0>2d}_gs.png")

                    # assert False

            print("finish sampling data for object", object_i)
            objects_sample_data.append(sample_data)

        fov = 40.0
        fx = fy = render_W / (2 * np.tan(fov * np.pi / 180. / 2))
        print("sampling data for main mesh...")
        main_mesh_sample_data = []
        train_cameras = pipeline.datamanager.train_dataset.cameras
        for camera in train_cameras:
            cam_pos = camera.camera_to_worlds[:3, 3].reshape(1, 3)
            dir = -camera.camera_to_worlds[:3, 2].reshape(1, 3)
            up_vec = camera.camera_to_worlds[:3, 1].reshape(1, 3)
            dist = 1.0
            end_point = cam_pos + dir * dist
            R, tvec = look_at_view_transform(eye=cam_pos, at=end_point, up=up_vec)

            camera_p3d = FoVPerspectiveCameras(
                znear=znear,
                zfar=zfar,
                aspect_ratio=aspect_ratio,
                fov=fov,
                degrees=True,
                R=R,
                T=tvec,
                device='cuda'
            )

            raster_settings = RasterizationSettings(
                image_size=resolution,
                blur_radius=1e-10,
                faces_per_pixel=1,
                perspective_correct=False,
                cull_backfaces=False
            )

            renderer = MeshRendererWithFragments(
                rasterizer=MeshRasterizer(
                    cameras=camera_p3d,
                    raster_settings=raster_settings
                ),
                shader=HardPhongShader(device='cuda', cameras=camera_p3d, lights=AmbientLights(device='cuda'))
            )

            image, frag = renderer(main_tex_mesh)

            pix_to_face = frag.pix_to_face.reshape(render_H, render_W)
            valid = pix_to_face >= 0
            pix_to_face_valid = pix_to_face[valid]

            pix_to_face_overlap = torch.isin(pix_to_face_valid, main_tex_mesh_overlap_faces)

            inside_faces_cnt = torch.unique(pix_to_face_valid[pix_to_face_overlap]).shape[0]
            if inside_faces_cnt >= 0.2 * main_tex_mesh_overlap_faces.shape[0]:
                mask = torch.zeros(render_H, render_W, device="cuda")
                mask_valid = mask[valid]
                mask_valid[pix_to_face_overlap] = 1.0
                mask[valid] = mask_valid

                if mask_dilate_n1 > 0:
                    mask = torch.from_numpy(binary_dilation(mask.cpu().numpy(), iterations=mask_dilate_n1)).cuda()
                else:
                    mask = torch.from_numpy((mask.cpu().numpy())).cuda()

                depth = frag.zbuf.reshape(render_H, render_W)
                depth[torch.logical_not(mask > 0.)] = 0.

                sample_i = len(main_mesh_sample_data)

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
                    height=render_H,
                    width=render_W,
                    camera_to_worlds=c2w_gl[..., :3, :4].float().reshape(1, 3, 4),
                    camera_type=CameraType.PERSPECTIVE,
                ).to("cpu")


                main_mesh_sample_data.append({
                    "mesh_camera": camera_p3d.to("cpu"),
                    "rgb": image[..., :3].reshape(render_H, render_W, 3).cpu(),
                    "mask": (mask > 0.).cpu(),
                    "depth": depth.cpu(),
                    "gs_camera": target_camera,
                })

                # image_pil = Image.fromarray(np.clip(image[..., :3].reshape(render_H, render_W, 3).cpu().numpy() * 255., 0, 255).astype(np.uint8))
                # image_pil.save(f"./vis/cs_kitchen/gs_internal_uc_objs/main_sample_{sample_i:0>2d}.png")
                #
                # mask_pil = Image.fromarray(np.clip(mask.cpu().numpy() * 255., 0, 255).astype(np.uint8))
                # mask_pil.save(f"./vis/cs_kitchen/gs_internal_uc_objs/main_sample_{sample_i:0>2d}_mask.png")

                # model.visible_gs_indices = gaussian_main_indices
                # output = model.get_outputs(target_camera)
                # model.visible_gs_indices = None

                # render_pil = Image.fromarray(np.clip(output["rgb"].reshape(render_H, render_W, 3).cpu().numpy() * 255., 0, 255).astype(np.uint8))
                # render_pil.save(f"./vis/cs_kitchen/gs_internal_uc_objs/main_sample_{sample_i:0>2d}_gs.png")
                #
                # assert False
else:
    config, pipeline, checkpoint_path, _, optimizer = resume_setup_gs_uc(Path(load_config))
    model = pipeline.model

    model_mesh_verts = model.mesh_verts.cpu().numpy()
    model_mesh_faces = model.mesh_faces.cpu().numpy()
    model_mesh = trimesh.Trimesh(model_mesh_verts, model_mesh_faces, process=False)

    door_depth = 0.2
    # tmp_save_dir = "./vis/cs_kitchen/gs_internal/"
    # os.makedirs(tmp_save_dir, exist_ok=True)
    save_dir = os.path.join(os.path.dirname(load_config), save_note)
    save_dir_doors = os.path.join(save_dir, "doors")
    save_dir_boxes = os.path.join(save_dir, "boxes")
    save_dir_remaining = os.path.join(save_dir, "remaining")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_doors, exist_ok=True)
    os.makedirs(save_dir_boxes, exist_ok=True)
    os.makedirs(save_dir_remaining, exist_ok=True)

    textured_mesh_path = os.path.join(bakedsdf_dir, 'texture_mesh/mesh-simplify.obj')
    # texture_image_path = os.path.join(bakedsdf_dir, 'texture_mesh/mesh-simplify.png')

    high_res_mesh_path = os.path.join(bakedsdf_dir, 'mesh.ply')
    high_res_mesh_processed_path = os.path.join(bakedsdf_dir, 'mesh-high-processed.ply')

    if os.path.exists(high_res_mesh_processed_path):
        high_res_mesh_trimesh = trimesh.exchange.load.load_mesh(high_res_mesh_processed_path)
    else:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(high_res_mesh_path)
        target_faces_num = 5000000
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces_num)
        m = ms.current_mesh()
        v_np = m.vertex_matrix()
        f_np = m.face_matrix()
        high_res_mesh_trimesh = trimesh.Trimesh(v_np, f_np, process=False)
        trimesh.exchange.export.export_mesh(high_res_mesh_trimesh, high_res_mesh_processed_path)


    model = model.train()
    N_Gaussians_original = model.means.shape[0]

    image_filenames = pipeline.datamanager.train_dataset._dataparser_outputs.image_filenames
    train_image_filenames = [os.path.splitext(os.path.basename(str(elem)))[0] for elem in image_filenames]

    image_filenames = pipeline.datamanager.eval_dataset._dataparser_outputs.image_filenames
    eval_image_filenames = [os.path.splitext(os.path.basename(str(elem)))[0] for elem in image_filenames]

    with open(os.path.join(bakedsdf_dir, "drawers", "trace.json"), 'r') as f:
        trace_data = json.load(f)

    drawer_files = [name for name in os.listdir(os.path.join(bakedsdf_dir, "drawers", "results")) if name.endswith(".ply")]
    total_drawers_num = len(drawer_files)

    total_frames = len(trace_data["00"])

    faces_means = torch.mean(model.mesh_faces_verts, dim=1).reshape(-1, 3).cpu().numpy()
    means_pad = np.pad(faces_means, ((0, 0), (0, 1)), constant_values=(0, 1))
    prim_means_indices = []
    interact_type_list = []
    for prim_i in range(total_drawers_num):
        with open(os.path.join(bakedsdf_dir, "drawers", "results", f"drawer_{prim_i}.pkl"), 'rb') as f:
            prim_info = pickle.load(f)
            prim_transform = prim_info["transform"]
            interact_type = prim_info["interact"]
        interact_type_list.append(interact_type)

        means_pad_transformed = means_pad @ np.linalg.inv(prim_transform).T
        means_transformed = means_pad_transformed[:, :3] / means_pad_transformed[:, 3:]
        scale_limit = np.array([1e4 * door_depth, 1, 1]).reshape(1, 3)
        means_transformed = means_transformed / scale_limit

        # inside_yz = (np.abs(means_transformed[:, 1:]) < 0.5)
        # if interact_type in ["1.1", "2", "3.3"]:
        #     inside_x = np.logical_and(means_transformed[:, 0:1] > -door_depth * 0.2,
        #                               means_transformed[:, 0:1] < door_depth)
        # else:
        #     inside_x = np.logical_and(means_transformed[:, 0:1] < door_depth * 0.2,
        #                               means_transformed[:, 0:1] > -door_depth)
        # print("means interact_type: ", interact_type)

        inside_y = (np.abs(means_transformed[:, 1:2]) < 0.5)
        # inside_z = (np.abs(means_transformed[:, 2:]) < 0.5)
        inside_z = np.logical_and(
            means_transformed[:, 2:] < 0.5,
            means_transformed[:, 2:] > -0.5
        )
        if interact_type in ["1.1"]:
            inside_x = np.logical_and(
                means_transformed[:, 0:1] > -door_back_distance_means,
                means_transformed[:, 0:1] < 0.3)

        elif interact_type in ["1.2"]:
            inside_x = np.logical_and(
                means_transformed[:, 0:1] < door_back_distance_means,
                means_transformed[:, 0:1] > -0.3)

        else:
            inside_x = np.logical_and(
                means_transformed[:, 0:1] > -door_back_distance_means,
                means_transformed[:, 0:1] < 0.3)
        inside = np.concatenate([inside_x, inside_y, inside_z], axis=-1)

        prim_i_faces_indices = np.all(inside, axis=1).reshape(-1)

        prim_i_faces_indices = np.logical_or(prim_i_faces_indices, np.all(np.abs(means_transformed) < 0.3, axis=-1).reshape(-1))

        prim_i_faces_indices = np.logical_and(prim_i_faces_indices, np.logical_not(
            np.all(np.concatenate([
                np.logical_and(np.abs(means_transformed[:, 0:1]) > door_back_distance_means,
                               np.abs(means_transformed[:, 0:1]) < 0.3) if interact_type in ["1.1", "2", "3.3"] else
                np.logical_and(np.abs(means_transformed[:, 0:1]) < -door_back_distance_means,
                               np.abs(means_transformed[:, 0:1]) > -0.3),
                np.abs(means_transformed[:, 1:2]) < 0.5,
                np.abs(means_transformed[:, 2:3]) > 0.45
            ], axis=-1).reshape(-1, 3), axis=-1)
        ))

        prim_i_faces_indices = torch.nonzero(torch.from_numpy(prim_i_faces_indices).cuda()).reshape(-1)
        prim_i_means_indices = torch.isin(model.gaussians_to_mesh_indices, prim_i_faces_indices)

        for prim_j in range(prim_i):
            prim_j_means_indices = prim_means_indices[prim_j]
            prim_i_means_indices = torch.logical_and(prim_i_means_indices, torch.logical_not(prim_j_means_indices))
        # print(f"prim_i_means_indices {prim_i}: {torch.count_nonzero(prim_i_means_indices)}")
        prim_means_indices.append(prim_i_means_indices)


    prim_means_indices = [torch.nonzero(elem) for elem in prim_means_indices]
    # print("total_frames: ", total_frames)

    # textured_mesh = load_objs_as_meshes([textured_mesh_path], device='cuda')
    textured_mesh = Meshes(
        torch.from_numpy(high_res_mesh_trimesh.vertices).unsqueeze(0).cuda().float(),
        torch.from_numpy(high_res_mesh_trimesh.faces).unsqueeze(0).cuda().int(),
        textures=TexturesVertex(torch.zeros(high_res_mesh_trimesh.vertices.shape[0], 3).unsqueeze(0).cuda().float())
    )

    textured_mesh_verts = textured_mesh.verts_packed()
    textured_mesh_faces = textured_mesh.faces_packed()
    full_mesh_verts_pad = np.pad(textured_mesh_verts.cpu().numpy(), ((0, 0), (0, 1)), constant_values=(0, 1))

    mesh_door_indices = []
    for prim_i in range(total_drawers_num):
        with open(os.path.join(bakedsdf_dir, "drawers", "results", f"drawer_{prim_i}.pkl"), 'rb') as f:
            prim_transform = pickle.load(f)
            interact_type = prim_transform["interact"]
            prim_transform = prim_transform["transform"]
        means_pad_transformed = full_mesh_verts_pad @ np.linalg.inv(prim_transform).T
        means_transformed = means_pad_transformed[:, :3] / means_pad_transformed[:, 3:]
        scale_limit = np.array([1e4 * door_depth, 1, 1]).reshape(1, 3)
        means_transformed = means_transformed / scale_limit

        # inside_yz = (np.abs(means_transformed[:, 1:]) < 0.5)
        # if interact_type in ["1.1", "2", "3.3"]:
        #     inside_x = np.logical_and(means_transformed[:, 0:1] > -door_depth * 0.2,
        #                               means_transformed[:, 0:1] < door_depth)
        # else:
        #     inside_x = np.logical_and(means_transformed[:, 0:1] < door_depth * 0.2,
        #                               means_transformed[:, 0:1] > -door_depth)
        # print("door 0 interact_type: ", interact_type)

        inside_y = (np.abs(means_transformed[:, 1:2]) < door_side_ratio)
        inside_z = np.logical_and(
            means_transformed[:, 2:] < door_side_ratio,
            means_transformed[:, 2:] > -door_side_ratio
        )
        if interact_type in ["1.1"]:
            inside_x = np.logical_and(
                means_transformed[:, 0:1] > -0.05,
                means_transformed[:, 0:1] < 0.3)
            # inside_y = np.logical_and(
            #     means_transformed[:, 1:2] < 0.5,
            #     means_transformed[:, 1:2] > -0.4)
        elif interact_type in ["1.2"]:
            inside_x = np.logical_and(
                means_transformed[:, 0:1] < 0.05,
                means_transformed[:, 0:1] > -0.3)
            # inside_y = np.logical_and(
            #     means_transformed[:, 1:2] < 0.4,
            #     means_transformed[:, 1:2] > -0.5)
        else:
            inside_x = np.logical_and(
                means_transformed[:, 0:1] > -0.05,
                means_transformed[:, 0:1] < 0.3)
            # inside_y = np.logical_and(
            #     means_transformed[:, 1:2] < 0.5,
            #     means_transformed[:, 1:2] > -0.4)

        inside = np.concatenate([inside_x, inside_y, inside_z], axis=-1)
        prim_i_means_indices = np.all(inside, axis=1).reshape(-1)

        prim_i_means_indices = torch.from_numpy(prim_i_means_indices).cuda()
        for prim_j in range(prim_i):
            prim_j_means_indices = mesh_door_indices[prim_j]
            # prim_i_means_indices = torch.logical_and(prim_i_means_indices, torch.logical_not(prim_j_means_indices))
        # print(f"prim_i_means_indices {prim_i}: {torch.count_nonzero(prim_i_means_indices)}")
        mesh_door_indices.append(prim_i_means_indices)

    # for the base
    mesh_door_indices_base = []
    for prim_i in range(total_drawers_num):
        with open(os.path.join(bakedsdf_dir, "drawers", "results", f"drawer_{prim_i}.pkl"), 'rb') as f:
            prim_transform = pickle.load(f)
            interact_type = prim_transform["interact"]
            prim_transform = prim_transform["transform"]
        means_pad_transformed = full_mesh_verts_pad @ np.linalg.inv(prim_transform).T
        means_transformed = means_pad_transformed[:, :3] / means_pad_transformed[:, 3:]
        scale_limit = np.array([1e4 * door_depth, 1, 1]).reshape(1, 3)
        means_transformed = means_transformed / scale_limit

        # inside_yz = (np.abs(means_transformed[:, 1:]) < 0.5)
        # if interact_type in ["1.1", "2", "3.3"]:
        #     inside_x = np.logical_and(means_transformed[:, 0:1] > -door_depth * 0.2,
        #                               means_transformed[:, 0:1] < door_depth)
        # else:
        #     inside_x = np.logical_and(means_transformed[:, 0:1] < door_depth * 0.2,
        #                               means_transformed[:, 0:1] > -door_depth)
        # print("door 0 interact_type: ", interact_type)

        inside_y = (np.abs(means_transformed[:, 1:2]) < 0.5)
        inside_z = np.logical_and(
            means_transformed[:, 2:] < 0.5,
            means_transformed[:, 2:] > -0.5
        )
        if interact_type in ["1.1"]:
            inside_x = np.logical_and(
                means_transformed[:, 0:1] > -0.05,
                means_transformed[:, 0:1] < 0.3)
            # inside_y = np.logical_and(
            #     means_transformed[:, 1:2] < 0.5,
            #     means_transformed[:, 1:2] > -0.4)
        elif interact_type in ["1.2"]:
            inside_x = np.logical_and(
                means_transformed[:, 0:1] < 0.05,
                means_transformed[:, 0:1] > -0.3)
            # inside_y = np.logical_and(
            #     means_transformed[:, 1:2] < 0.4,
            #     means_transformed[:, 1:2] > -0.5)
        else:
            inside_x = np.logical_and(
                means_transformed[:, 0:1] > -0.05,
                means_transformed[:, 0:1] < 0.3)
            # inside_y = np.logical_and(
            #     means_transformed[:, 1:2] < 0.5,
            #     means_transformed[:, 1:2] > -0.4)

        inside = np.concatenate([inside_x, inside_y, inside_z], axis=-1)
        prim_i_means_indices = np.all(inside, axis=1).reshape(-1)

        prim_i_means_indices = torch.from_numpy(prim_i_means_indices).cuda()
        # for prim_j in range(prim_i):
        #     prim_j_means_indices = mesh_door_indices_base[prim_j]
            # prim_i_means_indices = torch.logical_and(prim_i_means_indices, torch.logical_not(prim_j_means_indices))
        # print(f"prim_i_means_indices {prim_i}: {torch.count_nonzero(prim_i_means_indices)}")
        mesh_door_indices_base.append(prim_i_means_indices)

    F_cnt = textured_mesh_faces.shape[0]
    remaining_face_indices = torch.ones(F_cnt) > 0
    for prim_i in range(total_drawers_num):
        door_keep_faces = filter_faces_from_verts(mesh_door_indices[prim_i].cpu(), textured_mesh_faces.cpu())
        remaining_face_indices[door_keep_faces] = False
    remaining_face_indices = torch.arange(F_cnt)[remaining_face_indices]
    remaining_mesh = textured_mesh.submeshes([[remaining_face_indices]])

    trimesh.exchange.export.export_mesh(
        trimesh.Trimesh(
            remaining_mesh.verts_packed().cpu().numpy(),
            remaining_mesh.faces_packed().cpu().numpy(),
            process=False
        ),
        os.path.join(save_dir_remaining, f"remaining.ply")
    )

    print("loading drawer meshes")
    append_indices_prims = []
    drawer_interior_mesh_p3d_list = []
    for prim_i in tqdm(range(total_drawers_num)):
        gs_drawer_mesh_path = os.path.join(matfuse_dir, f"{prim_i:0>2d}/mesh_{prim_i:0>2d}.obj")
        drawer_interior_mesh_p3d = load_objs_as_meshes([gs_drawer_mesh_path], device='cuda')
        drawer_interior_mesh_p3d_list.append(drawer_interior_mesh_p3d)
        append_indices, append_gaussian_indices = model.append_from_mesh(gs_drawer_mesh_path)
        append_indices_prims.append(append_gaussian_indices)

    print("loading drawer door meshes")
    append_indices_prims_door = []
    drawer_interior_mesh_p3d_list_door = []
    for prim_i in tqdm(range(total_drawers_num)):
        gs_drawer_door_mesh_path = os.path.join(matfuse_dir, f"{prim_i:0>2d}/mesh_door_{prim_i:0>2d}.obj")
        drawer_interior_mesh_door_p3d = load_objs_as_meshes([gs_drawer_door_mesh_path], device='cuda')
        drawer_interior_mesh_p3d_list_door.append(drawer_interior_mesh_door_p3d)
        append_indices_door, append_gaussian_indices_door = model.append_from_mesh(gs_drawer_door_mesh_path)
        append_indices_prims_door.append(append_gaussian_indices_door)



N_Gaussians = model.means.shape[0]
N_Gaussians_added = N_Gaussians - N_Gaussians_original
optimizer_config = config.optimizers.copy()
param_groups = {
    name: [model.gauss_params[name]]
    for name in ["means_2d", "normal_elevates", "scales", "quats", "features_dc", "features_rest", "opacities"]
}
for item in optimizer_config:
    if item in ["means2d", "normal_elevates"]:
        optimizer_config[item]["scheduler"] = None
        optimizer_config[item]["optimizer"] = AdamOptimizerConfig(lr=1.6e-6, eps=1e-15)

optimizer = Optimizers(optimizer_config, param_groups)

# for param_group_name, params in param_groups.items():
#     optimizer.parameters[param_group_name] = params

prim_mesh_dict_list = []
wandb.init(project="splat_merge", name=datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

print("preparing training images...")
door_mapping = (np.ones((textured_mesh_faces.shape[0])) * -1).astype(np.int32)

for prim_i in tqdm(range(total_drawers_num)):

    prim_mark = f"{prim_i:0>2d}"
    trace_data_prim = trace_data[prim_mark]

    finetune_iterations = 0

    drawer_dir = os.path.join(bakedsdf_dir, "drawers", "results")

    drawer_path = os.path.join(drawer_dir, f"drawer_{prim_i}.pkl")
    with open(drawer_path, 'rb') as f:
        prim_info = pickle.load(f)
        prim_transform = prim_info["transform"]
        interact_type = prim_info["interact"]
    # print("prim_i: ", prim_i)
    # print("interact_type: ", interact_type)
    # remove the original mesh door from the scene mesh
    full_mesh_verts_pad_transformed = full_mesh_verts_pad @ np.linalg.inv(prim_transform).T
    full_mesh_verts_transformed = full_mesh_verts_pad_transformed[:, :3] / full_mesh_verts_pad_transformed[:, 3:]
    scale_limit = np.array([1e4 * door_depth, 1, 1]).reshape(1, 3)
    full_mesh_verts_transformed = full_mesh_verts_transformed / scale_limit
    # inside_yz = (np.abs(full_mesh_verts_transformed[:, 1:]) < 0.5)
    # if interact_type in ["1.1", "2", "3.3"]:
    #     inside_x = np.logical_and(full_mesh_verts_transformed[:, 0:1] > -door_depth * 0.2, full_mesh_verts_transformed[:, 0:1] < door_depth)
    # else:
    #     inside_x = np.logical_and(full_mesh_verts_transformed[:, 0:1] < door_depth * 0.2, full_mesh_verts_transformed[:, 0:1] > -door_depth)
    # print("mesh door 2 interact_type: ", interact_type)
    means_transformed = full_mesh_verts_transformed
    inside_y = (np.abs(means_transformed[:, 1:2]) < 0.5)
    inside_z = np.logical_and(
        means_transformed[:, 2:] < 0.48,
        means_transformed[:, 2:] > -0.5
    )
    if interact_type in ["1.1"]:
        inside_x = np.logical_and(
            means_transformed[:, 0:1] > -0.05,
            means_transformed[:, 0:1] < 0.3)
        # inside_y = np.logical_and(
        #     means_transformed[:, 1:2] < 0.5,
        #     means_transformed[:, 1:2] > -0.4)
    elif interact_type in ["1.2"]:
        inside_x = np.logical_and(
            means_transformed[:, 0:1] < 0.05,
            means_transformed[:, 0:1] > -0.3)
        # inside_y = np.logical_and(
        #     means_transformed[:, 1:2] < 0.4,
        #     means_transformed[:, 1:2] > -0.5)
    else:
        inside_x = np.logical_and(
            means_transformed[:, 0:1] > -0.05,
            means_transformed[:, 0:1] < 0.3)
        # inside_y = np.logical_and(
        #     means_transformed[:, 1:2] < 0.5,
        #     means_transformed[:, 1:2] > -0.4)

    inside = np.concatenate([inside_x, inside_y, inside_z], axis=-1)

    drawer_sub_verts_local = np.all(inside, axis=1).reshape(-1)


    door_verts, door_faces, door_face_mapping_i = filter_mesh_from_vertices(drawer_sub_verts_local,
                                                          textured_mesh_verts.cpu().numpy(),
                                                          textured_mesh_faces.cpu().numpy())
    door_mapping[door_face_mapping_i] = int(prim_i)


    drawer_sub_verts_local = np.logical_not(drawer_sub_verts_local)
    no_drawer_mesh_verts, no_drawer_mesh_faces, _ = filter_mesh_from_vertices(drawer_sub_verts_local,
                                                                              textured_mesh_verts.cpu().numpy(),
                                                                              textured_mesh_faces.cpu().numpy())
    # get the door transform
    scale, _, angles, trans, _ = transformations.decompose_matrix(prim_transform)
    prim_rotation = transformations.euler_matrix(axes='sxyz', *angles).reshape(4, 4)
    prim_translation = np.eye(4)
    prim_translation[:3, 3] = trans
    prim_rot_trans_original = prim_translation @ prim_rotation

    # load internal mesh and transform it
    drawer_internal_dir = os.path.join(bakedsdf_dir, "drawers", "internal")
    drawer_prim_internal_original = trimesh.exchange.load.load_mesh(
        os.path.join(drawer_internal_dir, f"drawer_internal_{prim_i}.ply"))
    drawer_prim_internal_original_verts = np.array(drawer_prim_internal_original.vertices)
    drawer_prim_internal_boundary = np.stack(
        [drawer_prim_internal_original_verts.min(axis=0), drawer_prim_internal_original_verts.max(axis=0)], axis=0).reshape(2, 3)

    prim_rot_trans_original_inverse = np.linalg.inv(prim_rot_trans_original)
    full_mesh_verts_pad_prim_internal_transformed = full_mesh_verts_pad @ prim_rot_trans_original_inverse.T
    full_mesh_verts_prim_internal_transformed = full_mesh_verts_pad_prim_internal_transformed[:,
                                                :3] / full_mesh_verts_pad_prim_internal_transformed[:, 3:]

    full_mesh_verts_prim_internal_inside = np.logical_and(
        np.all(full_mesh_verts_prim_internal_transformed > drawer_prim_internal_boundary[0:1], axis=-1),
        np.all(full_mesh_verts_prim_internal_transformed < drawer_prim_internal_boundary[1:2], axis=-1)
    )
    #
    # drawer_prim_internal_verts_pad = np.pad(drawer_prim_internal_verts, ((0, 0), (0, 1)),
    #                                         constant_values=(0, 1))
    # drawer_prim_internal_verts_transformed_pad = drawer_prim_internal_verts_pad @ prim_rot_trans_original.T
    # drawer_prim_internal_verts_transformed = drawer_prim_internal_verts_transformed_pad[:,
    #                                          :3] / drawer_prim_internal_verts_transformed_pad[:, 3:]
    # drawer_prim_internal.vertices = drawer_prim_internal_verts_transformed

    drawer_interior_mesh_p3d = drawer_interior_mesh_p3d_list[prim_i]
    drawer_prim_internal = trimesh.Trimesh(vertices=drawer_interior_mesh_p3d.verts_packed().cpu().numpy(), faces=drawer_interior_mesh_p3d.faces_packed().cpu().numpy())
    drawer_prim_internal_verts = np.array(drawer_prim_internal.vertices)

    internal_verts = np.array(drawer_prim_internal.vertices)
    internal_faces = np.array(drawer_prim_internal.faces).astype(np.int32)

    internal_verts_torch = torch.from_numpy(internal_verts).float().cuda()
    internal_faces_torch = torch.from_numpy(internal_faces).float().cuda()
    internal_mesh_p3d = Meshes(
        verts=[internal_verts_torch],
        faces=[internal_faces_torch],
    )

    # textured mesh composition
    textured_mesh_p3d = textured_mesh
    F_cnt = textured_mesh_faces.shape[0]
    door_keep_faces = filter_faces_from_verts(mesh_door_indices[prim_i].cpu(), textured_mesh_faces.cpu())

    door_keep_faces_base = filter_faces_from_verts_strict(mesh_door_indices_base[prim_i].cpu(), textured_mesh_faces.cpu())
    base_keep_faces = torch.logical_not(door_keep_faces_base)

    full_mesh_verts_prim_internal_inside_torch = torch.from_numpy(full_mesh_verts_prim_internal_inside.copy())
    base_keep_faces = torch.logical_and(base_keep_faces, torch.logical_not(
        filter_faces_from_verts(full_mesh_verts_prim_internal_inside_torch.cpu(), textured_mesh_faces.cpu())))

    door_keep_faces_idx = torch.arange(F_cnt)[door_keep_faces]
    base_keep_faces_idx = torch.arange(F_cnt)[base_keep_faces]

    textured_mesh_base_p3d = textured_mesh_p3d.submeshes([[base_keep_faces_idx]])
    textured_mesh_door_p3d = textured_mesh_p3d.submeshes([[door_keep_faces_idx]])

    trimesh.exchange.export.export_mesh(
        trimesh.Trimesh(
            textured_mesh_door_p3d.verts_packed().cpu().numpy(),
            textured_mesh_door_p3d.faces_packed().cpu().numpy(),
            process=False
        ),
        os.path.join(save_dir_doors, f"door_{prim_i:0>2d}.ply")
    )

    trimesh.exchange.export.export_mesh(
        trimesh.Trimesh(
            internal_mesh_p3d.verts_packed().cpu().numpy(),
            internal_mesh_p3d.faces_packed().cpu().numpy(),
            process=False
        ),
        os.path.join(save_dir_boxes, f"box_{prim_i:0>2d}.ply")
    )

    door_verts_original = textured_mesh_door_p3d.verts_packed().reshape(-1, 3).clone()
    internal_verts_original = internal_mesh_p3d.verts_packed().reshape(-1, 3).clone()

    # filtering training views

    select_indices = []
    select_weights = []
    # for camera_i in range(num_cameras):
    #     select_indices.append(camera_i)

    drawer_interior_mesh_p3d = drawer_interior_mesh_p3d_list[prim_i]
    drawer_interior_mesh_p3d_verts_original = drawer_interior_mesh_p3d.verts_packed().clone()

    drawer_interior_mesh_p3d_door = drawer_interior_mesh_p3d_list_door[prim_i]
    drawer_interior_mesh_p3d_verts_original_door = drawer_interior_mesh_p3d_door.verts_packed().clone()
    prim_mesh_dict = {
        "internal_mesh_p3d": internal_mesh_p3d,
        "textured_mesh_base_p3d": textured_mesh_base_p3d,
        "textured_mesh_door_p3d": textured_mesh_door_p3d,
        "door_verts_original": door_verts_original,
        "internal_verts_original": internal_verts_original,
        "prim_rot_trans_original": prim_rot_trans_original,
        "trace_data_prim": trace_data_prim,
        "interact_type": interact_type,
        "select_indices": select_indices,
        "select_weights": select_weights,
        "drawer_interior_mesh_p3d": drawer_interior_mesh_p3d,
        "drawer_interior_mesh_p3d_verts_original": drawer_interior_mesh_p3d_verts_original,
        "drawer_interior_mesh_p3d_door": drawer_interior_mesh_p3d_door,
        "drawer_interior_mesh_p3d_verts_original_door": drawer_interior_mesh_p3d_verts_original_door,
        "scale": scale,
    }
    prim_mesh_dict_list.append(prim_mesh_dict)


door_mapping = torch.from_numpy(door_mapping).cuda()
train_cameras = pipeline.datamanager.train_dataset.cameras
num_cameras = train_cameras.camera_to_worlds.shape[0]

textured_mesh_verts_pad = F.pad(textured_mesh_verts.cuda(), pad=(0, 1), mode='constant', value=1.0)

print("rasterize for view point selecting...")
with torch.no_grad():

    for camera_i in tqdm(range(num_cameras)):

        mvp = train_cameras[camera_i].mvps.reshape(4, 4)
        vertices_clip = torch.matmul(textured_mesh_verts_pad, torch.transpose(mvp.cuda(), 0, 1)).float().unsqueeze(0)
        rast, _ = dr.rasterize(glctx, vertices_clip, textured_mesh_faces.int(), (544, 960))
        # rast = rast.flip([1]).cuda()
        pix_to_face = rast[..., -1].reshape(544, 960)
        valid_pix = pix_to_face > 0
        pix_to_face = (pix_to_face - 1).long()
        door_indices, door_cnts = torch.unique(door_mapping[torch.unique(pix_to_face[valid_pix].reshape(-1))].reshape(-1), return_counts=True)

        for door_idx, door_cnt in zip(door_indices, door_cnts):
            if door_idx >= 0:
                prim_mesh_dict_list[door_idx]["select_indices"].append(camera_i)
                prim_mesh_dict_list[door_idx]["select_weights"].append(door_cnt)

    for door_idx in range(len(prim_mesh_dict_list)):
        view_cnt = len(prim_mesh_dict_list[door_idx]["select_weights"])
        # print("door ", door_idx, " n_views: ", view_cnt)
        ws = torch.tensor(prim_mesh_dict_list[door_idx]["select_weights"]).cuda().float().reshape(-1)
        # print("maximum cnt: ", ws.max())
        ws = torch.nn.functional.softmax(ws * data_scale)
        # print("maximum possibility: ", ws.max())
        prim_mesh_dict_list[door_idx]["select_weights"] = ws.cpu().numpy()

# sds & finetune
gs_init_iterations = 1.0 * total_iterations
# guidance = Guidance(guidance_config, "cuda")
# guidance.init_text_embeddings(guidance_config.batch_size)
model.config.ssim_lambda = 0.2

for prim_i in range(total_drawers_num):
    prim_mesh_dict = prim_mesh_dict_list[prim_i]
    prim_mesh_dict["select_indices"] = np.array(prim_mesh_dict["select_indices"], dtype=np.int32)


select_indices_list = []
for prim_i in range(total_drawers_num):
    prim_mesh_dict = prim_mesh_dict_list[prim_i]
    select_indices = prim_mesh_dict["select_indices"]
    select_indices_list.append(select_indices)

max_move_frame_list = []
for prim_i in range(total_drawers_num):
    prim_mesh_dict = prim_mesh_dict_list[prim_i]
    trace_data_prim = prim_mesh_dict["trace_data_prim"]
    interact_type = prim_mesh_dict["interact_type"]
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

with open(os.path.join(save_dir, "traj.pkl"), 'wb') as f:
    pickle.dump(select_indices_list, f)

torch.cuda.empty_cache()

for iter_i in tqdm(range(total_iterations)):

    while True:
        prim_i = np.random.randint(low=0, high=total_drawers_num)
        if prim_i not in skip_door_indices:
            break
    prim_mesh_dict = prim_mesh_dict_list[prim_i]

    internal_mesh_p3d = prim_mesh_dict["internal_mesh_p3d"]
    textured_mesh_base_p3d = prim_mesh_dict["textured_mesh_base_p3d"]
    textured_mesh_door_p3d = prim_mesh_dict["textured_mesh_door_p3d"]
    door_verts_original = prim_mesh_dict["door_verts_original"]
    internal_verts_original = prim_mesh_dict["internal_verts_original"]
    prim_rot_trans_original = prim_mesh_dict["prim_rot_trans_original"]
    trace_data_prim = prim_mesh_dict["trace_data_prim"]
    interact_type = prim_mesh_dict["interact_type"]
    select_indices = prim_mesh_dict["select_indices"]
    select_weights = prim_mesh_dict["select_weights"]
    drawer_interior_mesh_p3d = prim_mesh_dict["drawer_interior_mesh_p3d"]
    drawer_interior_mesh_p3d_verts_original = prim_mesh_dict["drawer_interior_mesh_p3d_verts_original"]
    drawer_interior_mesh_p3d_door = prim_mesh_dict["drawer_interior_mesh_p3d_door"]
    drawer_interior_mesh_p3d_verts_original_door = prim_mesh_dict["drawer_interior_mesh_p3d_verts_original_door"]
    scale = prim_mesh_dict["scale"]

    # if iter_i % 2 != 0:
    if iter_i % 4 == 1:

        image_size = 384

        validation_threshold = 0.05
        target_camera = pipeline.datamanager.eval_dataset.cameras[0]
        fx = float(target_camera.fx) * 0.5
        fy = float(target_camera.fy) * 0.5
        fx = fy = (fx + fy) / 2
        cx = image_size * 0.5  # float(target_camera.cx)
        cy = image_size * 0.5  # float(target_camera.cy)
        # H, W = int(target_camera.height), int(target_camera.width)
        H, W = image_size, image_size
        resolution = [H, W]

        sample_cnt = 0
        with torch.no_grad():
            
            while True:

                # view selection
                rand_view = np.random.rand()
                if interact_type == "1.1":
                    interior_depth = scale[1]*1.5
                    
                    r_yz = ((scale[1]) ** 2 + (scale[2]) ** 2) ** 0.5
                    radius = np.random.uniform(low=1.0, high=2.0) * r_yz
                    _theta = np.random.uniform(low=-60, high=30) + 90
                    selected_frame = int(total_frames - 1)
                    target_y = np.random.uniform(low=-0.5, high=0.5) * scale[1]
                    target_z = np.random.uniform(low=-0.25, high=0.25) * scale[2]
                    camera_z = np.random.uniform(low=-1.0, high=1.0) * scale[2]

                    target_x = np.random.uniform(low=-interior_depth, high=0.2 * interior_depth)

                    end_point_original = np.array([target_x, target_y, target_z]).reshape(3)

                    max_move_frame = max_move_frame_list[prim_i]
                    assert max_move_frame > 0

                    # selected_frame = min(selected_frame, total_frames - 1)
                    selected_frame = np.random.randint(int(max_move_frame * 0.6), int(max_move_frame * 0.8)) #int(total_frames - 1)

                    theta = 90 - _theta

                    camera_x = radius * np.cos(theta * np.pi / 180.0)
                    camera_y = radius * np.sin(theta * np.pi / 180.0)  # - 0.5 * scale[1]
                    cam_pos = np.array([camera_x, camera_y, camera_z]).reshape(3)

                elif interact_type == "1.2":
                    interior_depth = scale[1]*1.5
                    
                    r_yz = ((scale[1]) ** 2 + (scale[2]) ** 2) ** 0.5
                    radius = np.random.uniform(low=1.0, high=2.0) * r_yz
                    _theta = np.random.uniform(low=-60, high=30) + 90
                    selected_frame = int(total_frames - 1)
                    target_y = np.random.uniform(low=-0.5, high=0.5) * scale[1]
                    target_z = np.random.uniform(low=-0.25, high=0.25) * scale[2]
                    camera_z = np.random.uniform(low=-1.0, high=1.0) * scale[2]

                    target_x = np.random.uniform(low=-interior_depth, high=0.2 * interior_depth)

                    end_point_original = np.array([target_x, target_y, target_z]).reshape(3)

                    max_move_frame = max_move_frame_list[prim_i]
                    assert max_move_frame > 0

                    selected_frame = np.random.randint(int(max_move_frame * 0.6), int(max_move_frame * 0.8)) #int(total_frames - 1)

                    # selected_frame = min(selected_frame, total_frames - 1)
                    theta = 90 - _theta

                    camera_x = radius * np.cos(theta * np.pi / 180.0)
                    camera_y = radius * np.sin(theta * np.pi / 180.0)  # - 0.5 * scale[1]
                    cam_pos = np.array([camera_x, camera_y, camera_z]).reshape(3)

                    end_point_original[..., :2] *= -1
                    cam_pos[..., :2] *= -1

                elif interact_type in ["2", "3.3"]:


                    interior_depth = scale[1]*1.25
                    
                    
                    direction = np.random.randint(4) - 1
                    max_move_frame = max_move_frame_list[prim_i]
                    yz_r = ((scale[1] * 0.5) ** 2 + (scale[2] * 0.5) ** 2) ** 0.5

                    target_y = np.random.uniform(low=-0.5, high=0.5) * scale[1]
                    target_z = np.random.uniform(low=-0.5, high=0.5) * scale[2]
                    end_point_original = np.array([0, target_y, target_z]).reshape(3)

                    if direction < 0:
                        radius = np.random.uniform(low=1.5, high=2.5) * yz_r
                        camera_z = np.random.uniform(low=0.1, high=0.3) * scale[2] * (-1)
                        _theta = np.random.uniform(low=-30, high=30) + 90
                        theta = 90 - _theta
                        selected_frame = np.random.randint(int(max_move_frame * 0.6),
                                                        int(max_move_frame * 1.0))

                        camera_x = radius * np.cos(theta * np.pi / 180.0)
                        camera_y = radius * np.sin(theta * np.pi / 180.0)
                        cam_pos = np.array([camera_x, camera_y, camera_z]).reshape(3)

                    elif direction >= 2:
                        camera_x = np.random.uniform(low=0.5, high=0.7) * interior_depth
                        camera_y = np.random.uniform(low=-0.45, high=0.45) * scale[1]
                        camera_z = np.random.uniform(low=-0.45, high=0.45) * scale[2]

                        cam_pos = np.array([camera_x, camera_y, camera_z]).reshape(3)

                        selected_frame = np.random.randint(int(max_move_frame * 0.8), int(max_move_frame * 1.3))
                    else:
                        radius = np.random.uniform(low=1.5, high=4.5) * yz_r
                        camera_z = np.random.uniform(low=2.0, high=3.5) * scale[2]
                        _theta = np.random.uniform(low=-30, high=30) + 90
                        theta = 90 - _theta
                        camera_x = radius * np.cos(theta * np.pi / 180.0)
                        camera_y = radius * np.sin(theta * np.pi / 180.0)

                        cam_pos = np.array([camera_x, camera_y, camera_z]).reshape(3)

                        selected_frame = np.random.randint(int(max_move_frame * 0.9), int(max_move_frame * 1.3))

                    selected_frame = min(selected_frame, total_frames - 1)




                elif interact_type == "3.1":

                    assert False, "not implemented"

                elif interact_type == "3.2":

                    assert False, "not implemented"


                else:
                    assert False

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
                # print("center: ", camera_p3d.get_camera_center())
                # print("eye cam_pos: ", cam_pos)
                # print("w2c: ", w2c)
                c2w = torch.inverse(w2c)
                c2w[:3, 0:2] *= -1
                # print("c2w: ", c2w)

                view_dict = {
                    "c2w": c2w,
                    "frame": selected_frame,
                }
                camera_p3d = camera_p3d.to('cuda')

                #
                # c2w = torch.from_numpy(view_dict["c2w"]).float()
                # intrinsic = torch.from_numpy(
                #     np.array([
                #         [fx, 0, cx],
                #         [0, fy, cy],
                #         [0, 0, 1]
                #     ])
                # ).reshape(3, 3).float()
                #

                # camera_p3d = cameras_from_opencv_projection(
                #     R=w2c[..., :3, :3].reshape(1, 3, 3),
                #     tvec=w2c[..., :3, 3].reshape(1, 3),
                #     camera_matrix=intrinsic.reshape(1, 3, 3),
                #     image_size=torch.from_numpy(np.array(resolution).astype(np.int32)).reshape(1, 2)
                # ).to('cuda')

                # print("znear: ", getattr(camera_p3d, "znear", None))
                # print("zfar: ", getattr(camera_p3d, "zfar", None))

                # read door articulate info
                trace_data_prim_original = trace_data_prim[0]
                original_translation_matrix = np.eye(4)
                original_translation = trace_data_prim_original["position"]
                original_translation_matrix[:3, 3] = np.array(original_translation).reshape(3)
                original_rotation_matrix = np.eye(4)
                original_rotation_matrix[:3, :3] = transformations.quaternion_matrix(
                    trace_data_prim_original["orientation"])[:3, :3]
                original_matrix = original_translation_matrix @ original_rotation_matrix

                trace_data_prim_frame = trace_data_prim[view_dict["frame"]]
                current_translation_matrix = np.eye(4)
                current_translation = trace_data_prim_frame["position"]
                current_translation_matrix[:3, 3] = np.array(current_translation).reshape(3)
                current_rotation_matrix = np.eye(4)
                current_rotation_matrix[:3, :3] = transformations.quaternion_matrix(
                    trace_data_prim_frame["orientation"])[:3, :3]
                current_matrix = current_translation_matrix @ current_rotation_matrix

                original_matrix = torch.from_numpy(original_matrix).float().cuda()
                current_matrix = torch.from_numpy(current_matrix).float().cuda()

                door_verts_p3d = door_verts_original.clone()
                door_verts_pad_p3d = torch.nn.functional.pad(door_verts_p3d, (0, 1), "constant", 1.0)
                door_verts_transformed_pad_p3d = door_verts_pad_p3d @ torch.inverse(original_matrix).T @ current_matrix.T
                door_verts_transformed_p3d = door_verts_transformed_pad_p3d[:, :3] / door_verts_transformed_pad_p3d[:, 3:]

                textured_mesh_door_p3d = textured_mesh_door_p3d.update_padded(door_verts_transformed_p3d.unsqueeze(0))

                if interact_type in ["2", "3.3"]:
                    internal_verts_p3d = internal_verts_original.clone()

                    internal_verts_pad_p3d = torch.nn.functional.pad(internal_verts_p3d, (0, 1), "constant", 1.0)
                    internal_verts_transformed_pad_p3d = internal_verts_pad_p3d @ torch.inverse(
                        original_matrix).T @ current_matrix.T
                    internal_verts_transformed_p3d = internal_verts_transformed_pad_p3d[:, :3] / internal_verts_transformed_pad_p3d[
                                                                                                :, 3:]

                    internal_mesh_p3d = internal_mesh_p3d.update_padded(internal_verts_transformed_p3d.unsqueeze(0))

                # door

                drawer_interior_mesh_p3d_verts_door = drawer_interior_mesh_p3d_verts_original_door.clone()

                drawer_interior_mesh_p3d_verts_pad_door = torch.nn.functional.pad(drawer_interior_mesh_p3d_verts_door, (0, 1), "constant", 1.0)
                drawer_interior_mesh_p3d_verts_transformed_pad_door = drawer_interior_mesh_p3d_verts_pad_door @ torch.inverse(
                    original_matrix).T @ current_matrix.T
                drawer_interior_mesh_p3d_verts_transformed_door = drawer_interior_mesh_p3d_verts_transformed_pad_door[:,
                                                :3] / drawer_interior_mesh_p3d_verts_transformed_pad_door[
                                                    :, 3:]

                drawer_interior_mesh_door_p3d = drawer_interior_mesh_p3d_door.update_padded(drawer_interior_mesh_p3d_verts_transformed_door.unsqueeze(0))

                # rendering
                composed_mesh = [textured_mesh_base_p3d, textured_mesh_door_p3d, drawer_interior_mesh_door_p3d, internal_mesh_p3d]

                lights = AmbientLights(device='cuda')

                raster_settings = RasterizationSettings(
                    image_size=resolution,
                    blur_radius=1e-10,
                    faces_per_pixel=1,
                    perspective_correct=False,
                    cull_backfaces=False
                )

                renderer = MeshRendererWithFragments(
                    rasterizer=MeshRasterizer(
                        cameras=camera_p3d,
                        raster_settings=raster_settings
                    ),
                    shader=SoftSilhouetteShader()
                )
                with torch.no_grad():
                    output = depth_blend(renderer, composed_mesh, resolution)

                internal_face_cnt = \
                torch.unique(output["pix_to_face"][len(composed_mesh) - 1][output["idx"] == len(composed_mesh)]).shape[0]

                # print("drawer_i: ", prim_i)
                # print("sample_cnt: ", sample_cnt)
                # print("internal_face_cnt: ", internal_face_cnt)
                # print("need faces cnt: ", validation_threshold * textured_mesh_door_p3d.faces_packed().shape[0])
                # print("all faces cnt: ", textured_mesh_door_p3d.faces_packed().shape[0])

                if sample_cnt >= 3 or internal_face_cnt > validation_threshold * internal_mesh_p3d.faces_packed().shape[0]:
                    # print("sample_cnt: ", sample_cnt)
                    mask = (output["idx"] == len(composed_mesh)).detach()
                    mask_door = (output["idx"] == len(composed_mesh) - 1).detach()

                    camera_params = {
                        'c2w': c2w,
                        # 'intrinsic': intrinsic,
                        'resolution': resolution,
                        'mask': mask,
                        'mask_door': mask_door,
                        'frame': view_dict["frame"],
                        'depth_map': output["depth_map"],
                        'camera_p3d': camera_p3d,
                    }

                    break
                sample_cnt += 1


        c2w = camera_params['c2w']
        # intrinsic = camera_params['intrinsic']
        resolution = camera_params['resolution']
        frame = camera_params['frame']
        mask = camera_params['mask']
        mask_door = camera_params['mask_door']

        mask = torch.logical_or(mask, mask_door)

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

        N_Gaussians = model.means.shape[0]
        prim_means_indices_prim_i = torch.zeros((N_Gaussians), dtype=torch.bool)
        prim_means_indices_prim_i[prim_means_indices[prim_i]] = True

        if interact_type in ["2", "3.3"]:
            prim_means_indices_prim_i[append_indices_prims[prim_i].to(prim_means_indices_prim_i.device)] = True

        # door
        prim_means_indices_prim_i[append_indices_prims_door[prim_i].to(prim_means_indices_prim_i.device)] = True

        trace_data_prim_transform_original = trace_data_prim[0]
        original_translation_matrix = np.eye(4)
        original_translation = trace_data_prim_transform_original["position"]
        original_translation_matrix[:3, 3] = np.array(original_translation).reshape(3)
        original_rotation_matrix = np.eye(4)
        original_rotation_matrix[:3, :3] = transformations.quaternion_matrix(
            trace_data_prim_transform_original["orientation"])[:3, :3]
        original_matrix = original_translation_matrix @ original_rotation_matrix
        original_matrix = torch.from_numpy(original_matrix).float().cuda()

        trace_data_prim_transform_current = trace_data_prim[frame]
        current_translation_matrix = np.eye(4)
        current_translation = trace_data_prim_transform_current["position"]
        current_translation_matrix[:3, 3] = np.array(current_translation).reshape(3)
        current_rotation_matrix = np.eye(4)
        current_rotation_matrix[:3, :3] = transformations.quaternion_matrix(
            trace_data_prim_transform_current["orientation"])[:3, :3]
        current_matrix = current_translation_matrix @ current_rotation_matrix
        current_matrix = torch.from_numpy(current_matrix).float().cuda()

        model.articulate_transform = {
            "transform_indices_list": [prim_means_indices_prim_i],
            "transform_matrix_means_list": [current_matrix @ torch.inverse(original_matrix)],
            "transform_matrix_quats_list": [torch.from_numpy(current_rotation_matrix[:3, :3]).float().cuda() @ torch.inverse(torch.from_numpy(original_rotation_matrix[:3, :3]).float().cuda())],
        }

        output = model.get_outputs(target_camera)

        rendering = output['rgb']
        mask_tensor = mask.to(rendering.device).reshape(H, W, 1).float().detach()
        rendering = rendering * mask_tensor + rendering.detach() * (1 - mask_tensor)

        camera_p3d = camera_params['camera_p3d']

        raster_settings = RasterizationSettings(
            image_size=resolution,
            blur_radius=1e-10,
            faces_per_pixel=1,
            perspective_correct=False,
            cull_backfaces=False
        )

        renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=camera_p3d,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(device='cuda', cameras=camera_p3d, lights=AmbientLights(device='cuda'))
        )

        if interact_type in ["2", "3.3"]:
            drawer_interior_mesh_p3d_verts = drawer_interior_mesh_p3d_verts_original.clone()

            drawer_interior_mesh_p3d_verts_pad = torch.nn.functional.pad(drawer_interior_mesh_p3d_verts, (0, 1), "constant", 1.0)
            drawer_interior_mesh_p3d_verts_pad_transformed = drawer_interior_mesh_p3d_verts_pad @ \
                                                             torch.inverse(original_matrix).T @ current_matrix.T
            drawer_interior_mesh_p3d_verts_transformed = drawer_interior_mesh_p3d_verts_pad_transformed[:, :3] \
                                                         / drawer_interior_mesh_p3d_verts_pad_transformed[:, 3:]

            drawer_interior_mesh_p3d = drawer_interior_mesh_p3d.update_padded(drawer_interior_mesh_p3d_verts_transformed.unsqueeze(0))

        # door
        drawer_interior_mesh_p3d_verts_door = drawer_interior_mesh_p3d_verts_original_door.clone()
        drawer_interior_mesh_p3d_verts_pad_door = torch.nn.functional.pad(drawer_interior_mesh_p3d_verts_door, (0, 1),
                                                                     "constant", 1.0)
        drawer_interior_mesh_p3d_verts_pad_transformed_door = drawer_interior_mesh_p3d_verts_pad_door @ \
                                                         torch.inverse(original_matrix).T @ current_matrix.T
        drawer_interior_mesh_p3d_verts_transformed_door = drawer_interior_mesh_p3d_verts_pad_transformed_door[:, :3] \
                                                     / drawer_interior_mesh_p3d_verts_pad_transformed_door[:, 3:]

        drawer_interior_mesh_p3d_door = drawer_interior_mesh_p3d_door.update_padded(
            drawer_interior_mesh_p3d_verts_transformed_door.unsqueeze(0))

        # drawer_interior_mesh_rendering, _ = renderer([drawer_interior_mesh_p3d, drawer_interior_mesh_p3d_door])
        with torch.no_grad():
            drawer_interior_mesh_rendering = depth_blend(renderer, [drawer_interior_mesh_p3d, drawer_interior_mesh_p3d_door], resolution)["rgb"]
            drawer_interior_mesh_rendering = drawer_interior_mesh_rendering[..., :3].reshape(H, W, 3)

        # do some mask dilation
        mask_np = mask.cpu().numpy().reshape(H, W)
        # print("1 mask_np: ", np.count_nonzero(mask_np))
        if np.any(mask_np):
            if mask_dilate_n2 > 0:
                mask_dilated = binary_dilation(mask_np, iterations=mask_dilate_n2)
            else:
                mask_dilated = (mask_np)

            mask_increased = np.logical_and(mask_dilated, np.logical_not(mask_np))

            if np.any(mask_increased):

                search_coords = np.stack(np.nonzero(mask_np), axis=-1)
                inpaint_coords = np.stack(np.nonzero(mask_increased), axis=-1)

                knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
                _, indices = knn.kneighbors(inpaint_coords)

                drawer_interior_mesh_rendering[tuple(inpaint_coords.T)] = drawer_interior_mesh_rendering[tuple(search_coords[indices[:, 0]].T)]
                mask = torch.from_numpy(mask_dilated).cuda()
        # print("2 mask_np: ", torch.count_nonzero(mask))

        loss_l1 = torch.abs((rendering - drawer_interior_mesh_rendering)[mask.reshape(H, W)]).mean()

        accumulation = output["accumulation"]
        acm_lambda = 50.0
        loss_acm = torch.mean(torch.nn.functional.relu(0.95 - accumulation)) * acm_lambda

        loss = loss_l1 + loss_acm
        
        torch.cuda.empty_cache()
        # print("GPU memory allocated: ", torch.cuda.memory_allocated() / 1024 ** 3, "GB / ", torch.cuda.max_memory_allocated() / 1024 ** 3, "GB")
        

        optimizer.zero_grad_all()
        loss.backward()

        # # check grad
        # means_grad = model.gauss_params["features_dc"].grad
        # print("0 means_grad: ", means_grad.shape, means_grad)
        # means_grad = model.gauss_params["features_dc"].grad[-N_Gaussians_added:]
        # print("1 means_grad: ", means_grad.shape, means_grad.max(), means_grad.min(),means_grad.mean(), means_grad)

        optimizer.optimizer_step_all()

        if iter_i % 10 == 1:
            texture_image_np = drawer_interior_mesh_rendering.detach().cpu().numpy()
            texture_image = np.clip(texture_image_np * 255, 0, 255).astype(np.uint8)
            pred_image = np.clip(rendering.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
            mask_np = mask.float().cpu().numpy().reshape(H, W)
            mask_image = np.clip(mask_np * 255, 0, 255).astype(np.uint8).reshape(H, W, -1)
            # pred_image[(mask_image > 0).reshape(H, W)] = np.clip(pred_image[(mask_image > 0).reshape(H, W)] * 0.7 + np.array([0.0, 1.0, 0.0]) * 255 * 0.3, 0, 255).astype(np.uint8)
            mask_image = np.concatenate([mask_image]*3, axis=-1)

            texture_image_np_masked = texture_image_np.copy()
            texture_image_np_masked[mask_np > 0] = texture_image_np_masked[mask_np > 0] * 0.6 + np.array([0., 0., 1.]).reshape(3) * 0.4
            texture_image_masked = np.clip(texture_image_np_masked * 255, 0, 255).astype(np.uint8)

            if camera_params["depth_map"] is not None:
                depth_map_image = camera_params["depth_map"].reshape(H, W, 1).cpu().numpy().astype(np.uint8)
                depth_map_image = np.concatenate([depth_map_image]*3, axis=-1)
                cmp_image = np.concatenate([texture_image, pred_image, mask_image, depth_map_image, texture_image_masked], axis=1)
            else:
                cmp_image = np.concatenate([texture_image, pred_image, mask_image, texture_image_masked], axis=1)
            cmp_image = Image.fromarray(cmp_image)
            cmp_image = wandb.Image(cmp_image, caption="cmp_image")
            wandb.log({"cmp_image": [cmp_image]})
        model.articulate_transform = None

    # finetune
    elif iter_i % 10 == 0:
        doors_or_object = np.random.randint(2)
        if training_with_segmented_objects and doors_or_object == 0:
            model.articulate_transform = None
            object_i = np.random.randint(object_cnt+1)
            H, W = render_H, render_W
            if object_i == object_cnt:
                sample_i = np.random.randint(len(main_mesh_sample_data))
                sample_data = main_mesh_sample_data[sample_i]

                target_camera = sample_data["gs_camera"].to("cuda")
                rgb = sample_data["rgb"].to("cuda")
                mask = sample_data["mask"].to("cuda")
                depth = sample_data["depth"].to("cuda")

                model.visible_gs_indices = gaussian_main_indices
                output = model.get_outputs(target_camera)
                model.visible_gs_indices = None

                loss_l1 = 30.0 * torch.mean(torch.abs(output["rgb"] - rgb)[mask])

                accumulation = output["accumulation"]
                acm_lambda = 2.0
                loss_acm = torch.mean(torch.nn.functional.relu(0.95 - accumulation)) * acm_lambda

                L1_depth = torch.abs(output["depth"].reshape(depth.shape) - depth)[mask].mean()

                loss = loss_l1 + loss_acm + L1_depth

                with torch.no_grad():
                    if iter_i % 100 == 0:
                        texture_image = np.clip(rgb.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
                        pred_image = np.clip(output["rgb"].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
                        mask_image = np.clip(mask.float().cpu().numpy() * 255, 0, 255).astype(np.uint8).reshape(H, W, -1)
                        mask_image = np.concatenate([mask_image] * 3, axis=-1)

                        # object_mask_image = np.clip(object_mask.float().cpu().numpy() * 255, 0, 255).astype(np.uint8).reshape(H,
                        # object_mask_image = np.concatenate([object_mask_image] * 3, axis=-1)

                        depth_map = np.clip((depth / depth.max()).cpu().numpy() * 255., 0, 255).astype(np.uint8)
                        depth_map_image = cv2.applyColorMap(depth_map, colormap=cv2.COLORMAP_PLASMA)
                        cmp_image = np.concatenate(
                            [texture_image, pred_image, mask_image, depth_map_image[..., ::-1]], axis=1)

                        cmp_image = Image.fromarray(cmp_image)
                        cmp_image = wandb.Image(cmp_image, caption="cmp_image")
                        wandb.log({"object_rendering_image": [cmp_image]})

            else:
                gaussian_in_object_indices = gaussian_in_object_indices_list[object_i]
                object_data = objects_sample_data[object_i]

                sample_i = np.random.randint(len(object_data))
                sample_data = object_data[sample_i]

                target_camera = sample_data["gs_camera"].to("cuda")
                rgb = sample_data["rgb"].to("cuda")
                mask = sample_data["mask"].to("cuda")
                depth = sample_data["depth"].to("cuda")
                object_mask = sample_data["object_mask"].to("cuda")

                model.visible_gs_indices = gaussian_in_object_indices
                output = model.get_outputs(target_camera)
                model.visible_gs_indices = None

                loss_l1 = torch.mean(torch.abs(output["rgb"] - rgb)[object_mask])

                accumulation = output["accumulation"]
                acm_lambda = 2.0
                loss_acm = torch.mean(torch.nn.functional.mse_loss(object_mask.float(), accumulation.reshape(object_mask.shape))) * acm_lambda

                L1_depth = torch.abs(output["depth"].reshape(depth.shape) - depth)[object_mask].mean()

                loss = loss_l1 + loss_acm + L1_depth

                with torch.no_grad():
                    if iter_i % 100 == 0:
                        texture_image = np.clip(rgb.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
                        pred_image = np.clip(output["rgb"].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
                        mask_image = np.clip(mask.float().cpu().numpy() * 255, 0, 255).astype(np.uint8).reshape(H, W, -1)
                        mask_image = np.concatenate([mask_image] * 3, axis=-1)

                        object_mask_image = np.clip(object_mask.float().cpu().numpy() * 255, 0, 255).astype(np.uint8).reshape(H,
                                                                                                                              W,
                                                                                                                              -1)
                        object_mask_image = np.concatenate([object_mask_image] * 3, axis=-1)

                        depth_map = np.clip((depth / depth.max()).cpu().numpy() * 255., 0, 255).astype(np.uint8)
                        depth_map_image = cv2.applyColorMap(depth_map, colormap=cv2.COLORMAP_PLASMA)
                        cmp_image = np.concatenate(
                            [texture_image, pred_image, object_mask_image, mask_image, depth_map_image[..., ::-1]], axis=1)

                        cmp_image = Image.fromarray(cmp_image)
                        cmp_image = wandb.Image(cmp_image, caption="cmp_image")
                        wandb.log({"object_rendering_image": [cmp_image]})

            torch.cuda.empty_cache()
            
            # print("GPU memory allocated: ", torch.cuda.memory_allocated() / 1024 ** 3, "GB / ", torch.cuda.max_memory_allocated() / 1024 ** 3, "GB")
            
            
            optimizer.zero_grad_all()
            loss.backward()
            optimizer.optimizer_step_all()


        else:
            # clean door boundary
            image_size = 384

            validation_threshold = 0.05
            target_camera = pipeline.datamanager.eval_dataset.cameras[0]
            fx = float(target_camera.fx) * 0.5
            fy = float(target_camera.fy) * 0.5
            fx = fy = (fx + fy) / 2
            cx = image_size * 0.5  # float(target_camera.cx)
            cy = image_size * 0.5  # float(target_camera.cy)
            # H, W = int(target_camera.height), int(target_camera.width)
            H, W = image_size, image_size
            resolution = [H, W]

            delta_angle = 45
            if interact_type == "1.1":
                radius = np.random.uniform(low=1.0, high=2.0)
                r_yz = (scale[1] ** 2 + scale[2] ** 2) ** 0.5
                # _theta = 90
                if np.random.rand() < 0.5:
                    _theta = np.random.uniform(low=delta_angle, high=180-delta_angle)
                else:
                    _theta = np.random.uniform(low=180+delta_angle, high=360-delta_angle)

                theta = 90 - _theta

                target_y = 0.
                target_z = 0.
                camera_z = np.random.uniform(low=-0.5, high=0.5) * scale[2]

                camera_x = radius * r_yz * np.cos(theta * np.pi / 180.0)
                camera_y = radius * r_yz * np.sin(theta * np.pi / 180.0)  # - 0.5 * scale[1]
                cam_pos = np.array([camera_x, camera_y, camera_z]).reshape(3)

                end_point_original = np.array([0, target_y, target_z]).reshape(3)

            elif interact_type == "1.2":
                radius = np.random.uniform(low=1.0, high=2.0)
                r_yz = (scale[1] ** 2 + scale[2] ** 2) ** 0.5
                # _theta = 90
                if np.random.rand() < 0.5:
                    _theta = np.random.uniform(low=delta_angle, high=180 - delta_angle)
                else:
                    _theta = np.random.uniform(low=180 + delta_angle, high=360 - delta_angle)
                theta = 90 - _theta
                target_y = 0.
                target_z = 0.
                camera_z = np.random.uniform(low=-0.5, high=0.5) * scale[2]

                # camera_z = 0.

                camera_x = radius * r_yz * np.cos(theta * np.pi / 180.0)
                camera_y = radius * r_yz * np.sin(theta * np.pi / 180.0)  # - 0.5 * scale[1]
                cam_pos = np.array([camera_x, camera_y, camera_z]).reshape(3)

                end_point_original = np.array([0, target_y, target_z]).reshape(3)

                end_point_original[..., :2] *= -1
                cam_pos[..., :2] *= -1

            elif interact_type in ["2", "3.3"]:
                radius = np.random.uniform(low=1.0, high=2.0)
                r_yz = (scale[1] ** 2 + scale[2] ** 2) ** 0.5

                # _theta = 90
                if np.random.rand() < 0.5:
                    _theta = np.random.uniform(low=delta_angle, high=180 - delta_angle)
                else:
                    _theta = np.random.uniform(low=180 + delta_angle, high=360 - delta_angle)
                theta = 90 - _theta

                target_y = 0.
                target_z = 0.
                camera_z = np.random.uniform(low=-0.5, high=0.5) * scale[2]

                # camera_z = 0.

                camera_x = radius * r_yz * np.cos(theta * np.pi / 180.0)
                camera_y = radius * r_yz * np.sin(theta * np.pi / 180.0)
                cam_pos = np.array([camera_x, camera_y, camera_z]).reshape(3)
                end_point_original = np.array([0, target_y, target_z]).reshape(3)

            else:
                assert  False

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

            view_dict = {
                "c2w": c2w,
                "frame": 0,
            }
            camera_p3d = camera_p3d.to('cuda')


            ### TODO mesh depth loss to make the door flat
            ## mesh:

            # textured_mesh_verts
            # textured_mesh_faces

            ## camera:

            n = 0.01
            f = 1e10  # infinite

            n00 = 2.0 * fx / W
            n11 = 2.0 * fy / H
            n02 = 2.0 * cx / W - 1.0
            n12 = 2.0 * cy / H - 1.0
            n32 = 1.0
            n22 = (f + n) / (f - n)
            n23 = (2 * f * n) / (n - f)
            camera_projmat = np.array([[n00, 0, n02, 0],
                                       [0, n11, n12, 0],
                                       [0, 0, n22, n23],
                                       [0, 0, n32, 0]], dtype=np.float32)
            camera_projmat = torch.from_numpy(camera_projmat).cuda().float().reshape(4, 4)
            w2c = torch.inverse(c2w.reshape(4, 4).cuda().float())
            mvp = camera_projmat @ w2c

            vertices_pad = F.pad(textured_mesh_verts, pad=(0, 1), mode='constant', value=1.0)
            vertices_clip = torch.matmul(vertices_pad, torch.transpose(mvp.cuda(), 0, 1)).float().unsqueeze(0)
            vertices_cam = torch.matmul(vertices_pad, torch.transpose(w2c.cuda(), 0, 1)).float()
            vertices_cam = vertices_cam[:, :3] / vertices_cam[:, 3:]
            vertices_depth_cam = vertices_cam[:, -1].reshape(-1, 1)

            rast, _ = dr.rasterize(glctx, vertices_clip, textured_mesh_faces.int(), (H, W))
            # rast = rast.flip([1]).cuda()
            bary = torch.stack([rast[..., 0], rast[..., 1], 1 - rast[..., 0] - rast[..., 1]], dim=-1).reshape(H, W, 3)
            pix_to_face = rast[..., -1].reshape(H, W)
            valid_pix = pix_to_face > 0
            pix_to_face = (pix_to_face - 1).long()

            pix_valid_depth = vertices_depth_cam[textured_mesh_faces.long()[pix_to_face[valid_pix]].reshape(-1)].reshape(-1, 3)
            pix_valid_bary = bary[valid_pix].reshape(-1).reshape(-1, 3)

            pix_valid_inverse_depth = 1 / (pix_valid_depth + 1e-10)
            pix_valid_depth = 1 / (torch.sum(pix_valid_inverse_depth * pix_valid_bary, dim=-1) + 1e-10)

            frame_depth = torch.zeros((H, W), device="cuda").float()
            frame_depth[valid_pix] = pix_valid_depth.reshape(-1)
            frame_depth = torch.abs(frame_depth)

            invalid_indices = torch.nonzero(torch.logical_not(valid_pix)).reshape(-1, 2)
            for dx, dy in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                invalid_indices_offset = invalid_indices + torch.tensor([dx, dy]).cuda().reshape(1, 2).int()
                invalid_indices_offset[:, 0] = torch.clip(invalid_indices_offset[:, 0], 0, H - 1)
                invalid_indices_offset[:, 1] = torch.clip(invalid_indices_offset[:, 1], 0, W - 1)
                frame_depth[torch.logical_not(valid_pix)] += frame_depth[
                                                                 invalid_indices_offset[:, 0], invalid_indices_offset[:,
                                                                                               1]] * 0.25
            frame_depth = frame_depth.detach()

            drawer_interior_mesh_p3d_verts_door = drawer_interior_mesh_p3d_verts_original_door.clone()
           
            drawer_interior_mesh_door_p3d = drawer_interior_mesh_p3d_door.update_padded(
                drawer_interior_mesh_p3d_verts_door.unsqueeze(0))

            # rendering

            lights = AmbientLights(device='cuda')

            raster_settings = RasterizationSettings(
                image_size=resolution,
                blur_radius=1e-10,
                faces_per_pixel=1,
                perspective_correct=False,
                cull_backfaces=False
            )

            renderer = MeshRendererWithFragments(
                rasterizer=MeshRasterizer(
                    cameras=camera_p3d,
                    raster_settings=raster_settings
                ),
                shader=SoftSilhouetteShader()
            )

            with torch.no_grad():
                _, frag = renderer(drawer_interior_mesh_door_p3d)

            mask_door = frag.pix_to_face.reshape(H, W) > 0

            door_i_vis_gs_indices = prim_means_indices[prim_i]

            model.visible_gs_indices = torch.cat([door_i_vis_gs_indices, append_indices_prims_door[prim_i].to(door_i_vis_gs_indices.device).reshape(-1, 1)], dim=0)

            output = model.get_outputs(target_camera)
            model.visible_gs_indices = None

            accumulation = output["accumulation"]
            accumulation = accumulation.reshape(mask_door.shape)

            gs_depth = output["depth"].reshape(H, W)

            # acm_lambda = 0.0
            acm_lambda = 200.0
            loss_acm = torch.mean(
                torch.nn.functional.mse_loss(mask_door.float(), accumulation)) * acm_lambda
            # depth_lambda = 0.0
            # depth_lambda = 100.0
            depth_lambda = .0
            loss_depth = torch.mean(torch.nn.functional.mse_loss(gs_depth[mask_door], frame_depth[mask_door])) * depth_lambda

            loss = loss_acm + loss_depth
            
            torch.cuda.empty_cache()
            
            # print("GPU memory allocated: ", torch.cuda.memory_allocated() / 1024 ** 3, "GB / ", torch.cuda.max_memory_allocated() / 1024 ** 3, "GB")

            optimizer.zero_grad_all()
            loss.backward()
            optimizer.optimizer_step_all()

            with torch.no_grad():
                if iter_i % 100 == 0:
                    pred_image = np.clip(output["rgb"].detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)
                    mask_image_door = np.clip(mask_door.float().cpu().numpy() * 255, 0, 255).astype(np.uint8).reshape(H, W, -1)
                    mask_image_door = np.concatenate([mask_image_door] * 3, axis=-1)

                    mask_image = np.clip(accumulation.float().cpu().numpy() * 255, 0, 255).astype(np.uint8).reshape(H, W, -1)
                    mask_image = np.concatenate([mask_image] * 3, axis=-1)

                    depth_min = float(gs_depth.min())
                    depth_max = float(gs_depth.max())

                    gs_depth_vis = np.clip((gs_depth.detach().cpu().numpy() - depth_min) / (depth_max - depth_min) * 255., 0, 255).astype(np.uint8)
                    frame_depth_vis = np.clip((frame_depth.detach().cpu().numpy() - depth_min) / (depth_max - depth_min) * 255., 0, 255).astype(np.uint8)

                    gs_depth_vis = cv2.applyColorMap(gs_depth_vis, cv2.COLORMAP_PLASMA)[..., ::-1]
                    frame_depth_vis = cv2.applyColorMap(frame_depth_vis, cv2.COLORMAP_PLASMA)[..., ::-1]

                    cmp_image = np.concatenate(
                        [pred_image, mask_image_door, mask_image, gs_depth_vis, frame_depth_vis], axis=1)

                    cmp_image = Image.fromarray(cmp_image)
                    cmp_image = wandb.Image(cmp_image, caption="cmp_image")
                    wandb.log({"object_rendering_image": [cmp_image]})

    else:
        model.articulate_transform = None

        target_idx = np.random.choice(len(pipeline.datamanager.train_dataset.cameras))

        target_camera = pipeline.datamanager.train_dataset.cameras[target_idx:target_idx + 1].to("cuda")
        output = model.get_outputs(target_camera)
        rgb = output["rgb"]
        rgb_gt = pipeline.datamanager.train_dataset.get_image_float32(target_idx).to(rgb.device)
        if rgb_gt.shape[-1] == 4:
            rgb_gt = rgb_gt[..., :3]
        loss_l1 = torch.abs(rgb_gt - rgb).mean()
        loss_sim = 1 - model.ssim(rgb_gt.permute(2, 0, 1)[None, ...], rgb.permute(2, 0, 1)[None, ...])
        loss = (1 - model.config.ssim_lambda) * loss_l1 + model.config.ssim_lambda * loss_sim
        loss *= 20.0

        accumulation = output["accumulation"]
        acm_lambda = 2.0
        loss_acm = torch.mean(torch.nn.functional.relu(1.0 - accumulation)) * acm_lambda

        depth_lambda = 5.0
        if "mesh_depths" in pipeline.datamanager.train_dataset.metadata:
            mesh_depth = pipeline.datamanager.train_dataset.metadata["mesh_depths"][target_idx].to("cuda")
            predicted_depth = output["depth"]
            if mesh_depth.shape[0] != predicted_depth.shape[0]:
                d = int(torch.floor(torch.tensor([mesh_depth.shape[0] / predicted_depth.shape[0]]) + 0.5))
                mesh_depth = resize_image(mesh_depth.unsqueeze(-1), d).squeeze(0)
            L1_depth = depth_lambda * torch.nn.functional.relu(torch.abs(mesh_depth.reshape(predicted_depth.shape) - predicted_depth) - 0.).mean()
        else:
            L1_depth = 0
        # print("L1_depth: ", L1_depth)
        loss = loss + loss_acm + L1_depth


        if model.config.use_scale_regularization:
            scale_exp = torch.exp(model.scales[:, :2])
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(model.config.max_gauss_ratio),
                )
                - model.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to("cuda")

        loss = loss + scale_reg
        
        torch.cuda.empty_cache()
        # print("GPU memory allocated: ", torch.cuda.memory_allocated() / 1024 ** 3, "GB / ", torch.cuda.max_memory_allocated() / 1024 ** 3, "GB")
        

        optimizer.zero_grad_all()
        loss.backward()
        optimizer.optimizer_step_all()

        if iter_i % 100 == 2:
            pred_image = Image.fromarray(np.clip(rgb.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8))
            pred_image = wandb.Image(pred_image, caption="finetune_image")
            wandb.log({"finetune_image": [pred_image]})

    torch.cuda.empty_cache()
    # show the allocated gpu memory in GB / total gpu memory in GB
    # print("GPU memory allocated: ", torch.cuda.memory_allocated() / 1024 ** 3, "GB / ", torch.cuda.max_memory_allocated() / 1024 ** 3, "GB")
    
    if iter_i % 1000 == 0 or iter_i == total_iterations - 1:
        with torch.no_grad():
            save_dict = model.export_splatfacto_on_mesh()
            torch.save(save_dict, os.path.join(save_dir, "som.pt"))

            if training_with_segmented_objects:
                trace_dict = {
                    "prim_means_indices": [elem.cpu().numpy() for elem in prim_means_indices],
                    "append_indices_prims": [elem.cpu().numpy() for elem in append_indices_prims],
                    "append_indices_prims_door": [elem.cpu().numpy() for elem in append_indices_prims_door],
                    "trace_data": trace_data,
                    "interact_type_list": interact_type_list,
                    "object_gs_inidices": [elem.cpu().numpy() for elem in gaussian_in_object_indices_list]
                }
            else:
                trace_dict = {
                    "prim_means_indices": [elem.cpu().numpy() for elem in prim_means_indices],
                    "append_indices_prims": [elem.cpu().numpy() for elem in append_indices_prims],
                    "append_indices_prims_door": [elem.cpu().numpy() for elem in append_indices_prims_door],
                    "trace_data": trace_data,
                    "interact_type_list": interact_type_list,
                }
            with open(os.path.join(save_dir, "trace.pkl"), 'wb') as f:
                pickle.dump(trace_dict, f)

            model.articulate_transform = None
            eval_cameras = pipeline.datamanager.eval_dataset.cameras

            total_psnr = 0
            print("eval all images...")
            for eval_i in tqdm(range(len(eval_cameras))):
                target_camera = eval_cameras[eval_i:eval_i + 1].to("cuda")
                output = model.get_outputs(target_camera)
                rgb = output["rgb"]
                rgb_gt = pipeline.datamanager.eval_dataset.get_image_float32(eval_i).to(rgb.device)
                if rgb_gt.shape[-1] == 4:
                    rgb_gt = rgb_gt[..., :3]
                psnr_i = float(psnr(rgb, rgb_gt.reshape(rgb.shape)))
                total_psnr += psnr_i

            total_psnr /= len(eval_cameras)
            wandb.log({"eval_psnr": total_psnr})

