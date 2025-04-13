from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
import tyro
from PIL import Image
import os
from rich.console import Console

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.misc import get_dict_to_torch
import nerfstudio.utils.pywavefront as pywavefront
from nerfstudio.cameras.camera_utils import rotation_matrix
import nvdiffrast.torch as dr
import torch.nn.functional as F
from copy import deepcopy
import pickle
from tqdm import tqdm
import imageio

import pytorch3d
from pytorch3d.io import load_objs_as_meshes, save_obj
from scipy.spatial.transform import Rotation
import transformations
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    MeshRendererWithFragments,
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex,
    AmbientLights,
    TexturesUV
)
from pytorch3d.utils import (
    cameras_from_opencv_projection
)
from pytorch3d.ops import (
    sample_points_from_meshes,
    interpolate_face_attributes
)
from pytorch3d.structures import (
    Meshes,
    Pointclouds,
)
from pytorch3d.loss import (
    chamfer_distance,
    point_mesh_face_distance,
)
import argparse

CONSOLE = Console(width=120)

import matplotlib.pyplot as plt


def image_grid(
    images,
    rows=None,
    cols=None,
    fill: bool = True,
    show_axes: bool = False,
    rgb: bool = True,
):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, H, W, 4) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im in zip(axarr.ravel(), images):
        if rgb:
            # only render RGB channels
            ax.imshow(im[..., :3])
        else:
            # only render Alpha channel
            ax.imshow(im[..., 3])
        if not show_axes:
            ax.set_axis_off()


# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")
device = "cuda"

parser = argparse.ArgumentParser()

parser.add_argument("--sdf_dir", type=str, required=True)
parser.add_argument("--image_dir", type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)

args = parser.parse_args()

ckpt_dir = args.sdf_dir
mask_dir = os.path.join(args.data_dir, "grounded_sam/propagate/sam")
mask_src_dir = os.path.join(args.data_dir, "grounded_sam")
interact_info_path = os.path.join(args.data_dir, "art_infer/interact_info.pkl")

stems = list(range(len(os.listdir(args.image_dir))))

# only use this if you want to manually overwrite the predicted articulate info
overwrite_articulate = {}

load_config = Path(f"{ckpt_dir}/config.yml")
mask_json_path = f"{ckpt_dir}/mask_mesh.pkl"


with open(interact_info_path, 'rb') as f:
    interact_info = pickle.load(f)
print("interact_info: ", interact_info.keys())
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

with open(os.path.join(mask_src_dir, "all.json"), 'r') as f:
    drawers = json.load(f)

os.makedirs(f"{ckpt_dir}/drawers/", exist_ok=True)
os.makedirs(f"{ckpt_dir}/drawers/logs", exist_ok=True)
os.makedirs(f"{ckpt_dir}/drawers/logs/opt1", exist_ok=True)
os.makedirs(f"{ckpt_dir}/drawers/logs/initialize", exist_ok=True)
os.makedirs(f"{ckpt_dir}/drawers/logs/raw", exist_ok=True)
os.makedirs(f"{ckpt_dir}/drawers/logs/rotline", exist_ok=True)
os.makedirs(f"{ckpt_dir}/drawers/logs/affordmax", exist_ok=True)
os.makedirs(f"{ckpt_dir}/drawers/logs/collision", exist_ok=True)
os.makedirs(f"{ckpt_dir}/drawers/results", exist_ok=True)
drawer_cnt = 0
cad_short_scale = 1e-10
config, pipeline, checkpoint_path = eval_setup(load_config)

image_filenames = pipeline.datamanager.train_dataset._dataparser_outputs.image_filenames
train_image_filenames = [os.path.splitext(os.path.basename(str(elem)))[0] for elem in image_filenames]
image_filenames = pipeline.datamanager.eval_dataset._dataparser_outputs.image_filenames
eval_image_filenames = [os.path.splitext(os.path.basename(str(elem)))[0] for elem in image_filenames]


door_extent = np.array([1e-4, 1, 1]).reshape(3)
door_transform = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0.5],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
]).reshape(4, 4)
drawer_cad_mesh_np = trimesh.primitives.Box(extents=door_extent, transform=door_transform)
trimesh.exchange.export.export_mesh(drawer_cad_mesh_np, f"{ckpt_dir}/door.ply")
print("save door mesh to: ", f"{ckpt_dir}/door.ply")

# box define
box_base_verts = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1.],
]).reshape(-1, 3)

box_base_verts_color = box_base_verts.copy()

box_base_verts = box_base_verts - 0.5

box_base_edges = [
    [0, 1],
    [0, 2],
    [1, 4],
    [2, 4],
    [0, 3],
    [1, 5],
    [2, 6],
    [4, 7],
    [3, 5],
    [3, 6],
    [5, 7],
    [6, 7],
]


def from_tensor_to_rotation_matrix(log_rot):
    log_rot = log_rot.reshape(1, 3)
    nrms = (log_rot * log_rot).sum(1)
    rot_angles = torch.clamp(nrms, 1e-4).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = torch.zeros((log_rot.shape[0], 3, 3), dtype=log_rot.dtype, device=log_rot.device)
    skews[:, 0, 1] = -log_rot[:, 2]
    skews[:, 0, 2] = log_rot[:, 1]
    skews[:, 1, 0] = log_rot[:, 2]
    skews[:, 1, 2] = -log_rot[:, 0]
    skews[:, 2, 0] = -log_rot[:, 1]
    skews[:, 2, 1] = log_rot[:, 0]
    skews_square = torch.bmm(skews, skews)

    ret = torch.eye(4, device=device).float().unsqueeze(0)
    ret[:, :3, :3] = (
            fac1[:, None, None] * skews
            + fac2[:, None, None] * skews_square
            + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )
    return ret.reshape(4, 4)


box_primitive = trimesh.primitives.Box()
box_primitive_verts = np.array(box_primitive.vertices).astype(np.float32).reshape(-1, 3)
box_primitive_faces = np.array(box_primitive.faces).astype(np.int32).reshape(-1, 3)

opt_dict = {}
opt_dict["params"] = []
opt_dict["camera_stems"] = []
opt_dict["drawer_mesh_short_axis"] = []
opt_dict["box_primitive_p3d"] = []
opt_dict["drawer_rotation_matrix"] = []
opt_dict["renderer_silhouette"] = []
opt_dict["render_cameras"] = []
opt_dict["sam_masks"] = []
opt_dict["depth_pmesh_list"] = []
opt_dict["stem"] = []
opt_dict["drawer_mesh_pt3d"] = []
opt_dict["mask_id"] = []
opt_dict["drawer_cad_scale_matrix"] = []
opt_dict["drawer_cad_to_origin"] = []
opt_dict["drawer_cad_mesh_verts"] = []
opt_dict["drawer_cad_mesh_faces"] = []
opt_dict["drawer_interact_info"] = []
for drawer_img in drawers:
    for mask_id in drawers[drawer_img]:
        drawer = (drawer_img, mask_id)
        print("drawer: ", drawer)

        drawer_cad_mesh = Meshes([torch.from_numpy(np.array(drawer_cad_mesh_np.vertices)).float()], [torch.from_numpy(np.array(drawer_cad_mesh_np.faces)).float()]).to(device)

        drawer_cad_mesh_verts = drawer_cad_mesh.verts_packed().cpu().numpy()
        drawer_cad_mesh_faces = drawer_cad_mesh.faces_packed().cpu().numpy()

        drawer_cad_to_origin = np.linalg.inv(door_transform.copy())
        drawer_cad_extents = door_extent.copy()

        drawer_cad_scale_matrix = np.eye(4)
        drawer_cad_scale_matrix[np.arange(3), np.arange(3)] = 1 / drawer_cad_extents.reshape(-1)

        drawer_cad_short_axis = drawer_cad_extents.argmin()

        # read mask info, mesh, and select mask image
        with open(mask_json_path, 'rb') as f:
            mask_full_info = pickle.load(f)

        mesh_mask = mask_full_info["mesh_mask"]
        drawer_mesh_info = mesh_mask[f"{drawer[0]}_{drawer[1]:0>2d}"]
        drawer_mesh = drawer_mesh_info["mesh"]
        drawer_mask_idx = drawer_mesh_info["idx"]

        sam_input = mask_full_info["sam_input"]
        camera_stems = []
        for stem in sam_input:
            for mask_info in sam_input[stem]:
                if drawer_mask_idx == mask_info["idx"] and os.path.exists(os.path.join(mask_dir, stem, f"{drawer_mask_idx}.png")) and int(stem.replace("frame_", "")) in stems:
                    camera_stems.append(stem)

        if len(camera_stems) > 40:
            camera_stems = [stem for idx, stem in enumerate(camera_stems) if idx in np.linspace(0, len(camera_stems)-1, 40).astype(np.int32)]


        print("camera_stems: ", camera_stems)

        # initialize drawer cad
        drawer_mesh_to_origin, drawer_mesh_extents = trimesh.bounds.oriented_bounds(
            trimesh.Trimesh(drawer_mesh[0], drawer_mesh[1]))

        # re-calculate the transform
        drawer_transform = np.linalg.inv(drawer_mesh_to_origin)
        box_base_verts_extended = box_base_verts * drawer_mesh_extents.reshape(1, 3)
        box_base_verts_pad = np.pad(box_base_verts_extended, ((0, 0), (0, 1)), constant_values=(0, 1))
        box_transformed_verts_pad = box_base_verts_pad @ drawer_transform.T
        box_transformed_verts = box_transformed_verts_pad[..., :3] / box_transformed_verts_pad[..., -1:]

        delta_z_list = []
        for edge in box_base_edges:
            delta_z = np.abs(box_transformed_verts[edge[0]][-1] - box_transformed_verts[edge[1]][-1])
            delta_z_list.append(float(delta_z))
        delta_zs = np.array(delta_z_list).reshape(-1)
        max_delta_z_edge_idxs = np.argsort(delta_zs)[-4:]
        joint_pos_list = []
        for max_delta_z_edge_idx in max_delta_z_edge_idxs:
            joint_edge = box_base_edges[max_delta_z_edge_idx]
            joint_pos = (box_transformed_verts[joint_edge[0]] + box_transformed_verts[joint_edge[1]]).reshape(1, 3) / 2
            joint_pos_list.append(joint_pos)
        joint_pos_list = np.concatenate(joint_pos_list, axis=0)
        joint_pos = joint_pos_list[0:1]
        joint_distance = np.sum((joint_pos - joint_pos_list) ** 2, axis=-1).reshape(4)
        left_joints = np.argsort(joint_distance)[:2]
        right_joints = np.argsort(joint_distance)[2:]

        drawer_center = np.mean(joint_pos_list, axis=0).reshape(3)
        ry = joint_pos_list[right_joints[0]] - joint_pos_list[left_joints[0]]
        ry = ry.reshape(3)
        sy = float(np.linalg.norm(ry, axis=0, ord=2))
        ry = ry / sy

        # judge the direction of ry
        right_pos = joint_pos_list[right_joints[0]].reshape(1, 3)
        left_pos = joint_pos_list[left_joints[0]].reshape(1, 3)
        both_pos = np.concatenate([right_pos, left_pos], axis=0)

        stem = os.path.splitext(os.path.basename(drawer_img))[0]
        idx = mask_id - 1
        mark = f"{stem}_{idx}"
        drawer_interact_info = interact_info[mark]

        opt_dict["stem"].append(stem)
        opt_dict["mask_id"].append(mask_id)

        if stem in train_image_filenames:
            target_dataset = pipeline.datamanager.train_dataset
            target_idx = train_image_filenames.index(stem)
        elif stem in eval_image_filenames:
            target_dataset = pipeline.datamanager.eval_dataset
            target_idx = eval_image_filenames.index(stem)
        else:
            assert False

        target_camera = deepcopy(target_dataset.cameras[target_idx])
        c2w = target_camera.camera_to_worlds.reshape(3, 4).clone().cpu()
        c2w[:3, 1:3] *= -1
        w2c = torch.inverse(torch.cat([c2w, torch.from_numpy(
            np.array([0, 0, 0, 1]).reshape(1, 4)
        ).to(c2w.device)], dim=0)).numpy()
        both_pos_pad = np.pad(both_pos, ((0, 0), (0, 1)), constant_values=(0, 1))
        both_pos_cv_pad = both_pos_pad @ w2c.T
        both_pos_cv = both_pos_cv_pad[:, :3] / both_pos_cv_pad[:, 3:]
        if both_pos_cv[0, 0] - both_pos_cv[1, 0] < 0:
            ry *= -1

        rz = box_transformed_verts[box_base_edges[max_delta_z_edge_idxs[0]][0]] - box_transformed_verts[
            box_base_edges[max_delta_z_edge_idxs[0]][1]]
        rz = rz.reshape(3)
        if rz[-1] < 0:
            rz = rz * -1
        sz = float(np.linalg.norm(rz, axis=0, ord=2))
        rz = rz / sz

        rx = np.cross(ry, rz).reshape(3)
        drawer_rot = np.stack([rx, ry, rz], axis=1)
        drawer_transform_new_rot = np.eye(4)
        drawer_transform_new_rot[:3, :3] = drawer_rot
        drawer_transform_new_trans = np.eye(4)
        drawer_transform_new_trans[:3, 3] = drawer_center
        drawer_transform_new_scale = np.eye(4)
        drawer_transform_new_scale[np.arange(3), np.arange(3)] = [1e-4, sy, sz]

        drawer_transform_new = drawer_transform_new_trans @ drawer_transform_new_rot @ drawer_transform_new_scale

        scale, _, angles, trans, _ = transformations.decompose_matrix(drawer_transform_new)

        drawer_mesh_pt3d = Meshes(verts=[torch.from_numpy(drawer_mesh[0]).float()], faces=[torch.from_numpy(drawer_mesh[1]).float()]).to(device)

        R = transformations.euler_matrix(axes='sxyz', *angles)
        drawer_rotation_matrix = R.reshape(4, 4)
        translate_matrix = np.eye(4)
        translate_matrix[:3, 3] = trans
        scale_matrix = np.eye(4)
        scale_matrix[np.arange(3), np.arange(3)] = scale

        drawer_mesh_transform = translate_matrix @ drawer_rotation_matrix @ scale_matrix
        drawer_mesh_to_origin = np.linalg.inv(drawer_mesh_transform)

        drawer_translate = torch.nn.Parameter(
            torch.from_numpy(np.array(trans, dtype=np.float32).reshape(-1)).to(device))
        drawer_scale = torch.nn.Parameter(
            torch.log(torch.from_numpy(np.array(scale, dtype=np.float32).reshape(-1))).to(device))
        drawer_additional_rotation = torch.nn.Parameter(torch.zeros((3), device=device).float())

        box_primitive_verts_pad = np.pad(box_primitive_verts, ((0, 0), (0, 1)), constant_values=(0, 1))
        drawer_initial_verts = (drawer_mesh_transform @ box_primitive_verts_pad.T).T
        drawer_initial_verts = drawer_initial_verts[..., :3] / drawer_initial_verts[..., -1:]
        drawer_template = trimesh.Trimesh(drawer_initial_verts, drawer_cad_mesh_faces,
                                          vertex_colors=box_base_verts_color.copy())
        trimesh.exchange.export.export_mesh(drawer_template,
                                            f"{ckpt_dir}/drawers/logs/initialize/drawer_template_initialized_{drawer_cnt}.ply")

        lr = 1e-1
        optimizer = torch.optim.SGD([drawer_translate, drawer_scale, drawer_additional_rotation], lr=lr, momentum=0.9)
        Niter = 200
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Niter, eta_min=lr * 0.01)

        opt_loop = tqdm(range(Niter))
        drawer_mesh_short_axis = 0

        box_primitive_verts_torch = torch.from_numpy(box_primitive_verts).float().to(device)
        box_primitive_faces_torch = torch.from_numpy(box_primitive_faces).to(device)
        box_primitive_p3d = Meshes(
            box_primitive_verts_torch.unsqueeze(0),
            box_primitive_faces_torch.unsqueeze(0),
        )

        drawer_rotation_matrix = torch.from_numpy(R).reshape(4, 4).to(device).float()

        for iter in opt_loop:
            additional_rotation = from_tensor_to_rotation_matrix(drawer_additional_rotation)
            drawer_scale_matrix = torch.eye(4).to(device)
            drawer_scale_matrix[torch.arange(3), torch.arange(3)] = torch.exp(drawer_scale)
            drawer_scale_matrix[drawer_mesh_short_axis, drawer_mesh_short_axis] = cad_short_scale
            drawer_translate_matrix = torch.eye(4).to(device)
            drawer_translate_matrix[:3, 3] = drawer_translate

            box_primitive_verts_torch = box_primitive_p3d.verts_packed().clone()
            box_primitive_verts_torch_pad = torch.nn.functional.pad(box_primitive_verts_torch, (0, 1), "constant", value=1)
            drawer_transformed_verts = (
                    drawer_translate_matrix @ additional_rotation @ drawer_rotation_matrix @ drawer_scale_matrix @ box_primitive_verts_torch_pad.T).T
            drawer_transformed_verts = drawer_transformed_verts[..., :3] / drawer_transformed_verts[..., -1:]

            drawer_offset = drawer_transformed_verts - box_primitive_verts_torch
            box_primitive_p3d_transformed = box_primitive_p3d.offset_verts(drawer_offset)

            cad_points = sample_points_from_meshes(box_primitive_p3d_transformed)
            mesh_points = sample_points_from_meshes(drawer_mesh_pt3d)
            loss, _ = chamfer_distance(cad_points, mesh_points)
            opt_loop.set_description(f"loss={loss.item():.5f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        with torch.no_grad():
            trimesh.exchange.export.export_mesh(
                trimesh.Trimesh(
                    box_primitive_p3d_transformed.verts_packed().cpu().numpy(),
                    box_primitive_p3d_transformed.faces_packed().cpu().numpy(),
                    vertex_colors=box_base_verts_color.copy()
                ), f"{ckpt_dir}/drawers/logs/opt1/drawer_template_opt1_{drawer_cnt}.ply")



        R_list = []
        tvec_list = []
        image_size_list = []
        camera_matrix_list = []
        mask_list = []
        image_list = []
        normal_list = []
        depth_list = []

        target_dataset = pipeline.datamanager.train_dataset
        target_idx = 0
        target_camera = deepcopy(target_dataset.cameras[target_idx])

        c2w = target_camera.camera_to_worlds.reshape(3, 4).cuda().clone()
        fx = float(target_camera.fx)
        fy = float(target_camera.fy)
        cx = float(target_camera.cx)
        cy = float(target_camera.cy)
        H, W = int(target_camera.height), int(target_camera.width)
        resolution = [H, W]

        raster_settings_silhouette = RasterizationSettings(
            image_size=[H, W],
            # blur_radius=1e-10,
            faces_per_pixel=1,
            perspective_correct=True,
            # bin_size = 0,
        )

        renderer_silhouette = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=None,
                raster_settings=raster_settings_silhouette
            ),
            shader=SoftSilhouetteShader()
        )

        for stem in tqdm(camera_stems):

            if stem in train_image_filenames:
                target_dataset = pipeline.datamanager.train_dataset
                target_idx = train_image_filenames.index(stem)
            elif stem in eval_image_filenames:
                target_dataset = pipeline.datamanager.eval_dataset
                target_idx = eval_image_filenames.index(stem)
            else:
                assert False

            target_camera = deepcopy(target_dataset.cameras[target_idx])

            c2w = target_camera.camera_to_worlds.reshape(3, 4).cuda().clone()
            fx = float(target_camera.fx)
            fy = float(target_camera.fy)
            cx = float(target_camera.cx)
            cy = float(target_camera.cy)
            H, W = int(target_camera.height), int(target_camera.width)
            resolution = [H, W]

            c2w[..., :3, 1:3] *= -1

            w2c = torch.inverse(torch.cat([c2w, torch.from_numpy(
                np.array([0, 0, 0, 1]).reshape(1, 4)
            ).to(c2w.device)], dim=0)).unsqueeze(0)
            w2c_R = w2c[:, :3, :3].float()

            R_list.append(w2c_R.float().to(device))
            tvec_list.append(w2c[:, :3, 3].float().to(device))
            image_size_list.append(torch.from_numpy(np.array(resolution).reshape(-1, 2)).to(device))
            camera_matrix_list.append(torch.from_numpy(
                np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])
            ).reshape(1, 3, 3).float().to(device))

            mask_path = os.path.join(mask_dir, stem, f"{drawer_mask_idx}.png")
            sam_mask = torch.from_numpy(np.array(Image.open(mask_path)) > 0).reshape(H, W)

            image_list.append(
                np.array(Image.open(
                    target_dataset._dataparser_outputs.image_filenames[target_idx]
                ).convert("RGB"), dtype=np.uint8)
            )

            mask_list.append(
                sam_mask.float().cpu()
            )

        render_cameras = cameras_from_opencv_projection(R=torch.cat(R_list, dim=0),
                                                        tvec=torch.cat(tvec_list, dim=0),
                                                        image_size=torch.cat(image_size_list, dim=0),
                                                        camera_matrix=torch.cat(camera_matrix_list, dim=0)
                                                        )
        sam_masks = torch.stack(mask_list, dim=0)
        stem_images = np.stack(image_list, axis=0)

        sigma = 1e-4


        lights = AmbientLights(device=device)

        depth_pmesh_list = []
        with torch.no_grad():
            for stem_i in tqdm(range(len(camera_stems))):
                silhouette_images_pmesh, fragments_pmesh = renderer_silhouette(drawer_mesh_pt3d,
                                                                               cameras=render_cameras[stem_i],
                                                                               lights=lights)
                silhouette_images_pmesh = silhouette_images_pmesh.reshape(H, W, 4)[..., -1]
                depth_pmesh = fragments_pmesh.zbuf[..., 0].reshape(H, W).cpu()
                depth_pmesh_list.append(depth_pmesh)

        opt_params = [drawer_translate, drawer_scale, drawer_additional_rotation]

        opt_dict["params"] += opt_params
        opt_dict["camera_stems"].append(camera_stems)
        opt_dict["drawer_mesh_short_axis"].append(drawer_mesh_short_axis)
        opt_dict["box_primitive_p3d"].append(box_primitive_p3d)
        opt_dict["drawer_rotation_matrix"].append(drawer_rotation_matrix)

        opt_dict["renderer_silhouette"].append(renderer_silhouette)
        opt_dict["render_cameras"].append(render_cameras)
        opt_dict["sam_masks"].append(sam_masks)
        opt_dict["depth_pmesh_list"].append(depth_pmesh_list)


        opt_dict["drawer_mesh_pt3d"].append(drawer_mesh_pt3d.clone())

        opt_dict["drawer_cad_scale_matrix"].append(drawer_cad_scale_matrix)
        opt_dict["drawer_cad_to_origin"].append(drawer_cad_to_origin)

        opt_dict["drawer_cad_mesh_verts"].append(drawer_cad_mesh_verts)
        opt_dict["drawer_cad_mesh_faces"].append(drawer_cad_mesh_faces)

        opt_dict["drawer_interact_info"].append(drawer_interact_info)

        drawer_cnt += 1


num_total_drawer = drawer_cnt
opt_dict["afford"] = []
opt_dict["interact_type"] = []
with torch.no_grad():
    for drawer_i in range(num_total_drawer):

        drawer_translate = opt_dict["params"][3 * drawer_i + 0]
        drawer_scale = opt_dict["params"][3 * drawer_i + 1]
        drawer_additional_rotation = opt_dict["params"][3 * drawer_i + 2]

        camera_stems = opt_dict["camera_stems"][drawer_i]
        drawer_mesh_short_axis = opt_dict["drawer_mesh_short_axis"][drawer_i]
        box_primitive_p3d = opt_dict["box_primitive_p3d"][drawer_i]
        drawer_rotation_matrix = opt_dict["drawer_rotation_matrix"][drawer_i]
        renderer_silhouette = opt_dict["renderer_silhouette"][drawer_i]

        stem = opt_dict["stem"][drawer_i]
        drawer_mesh_pt3d = opt_dict["drawer_mesh_pt3d"][drawer_i]
        mask_id = opt_dict["mask_id"][drawer_i]
        drawer_cad_scale_matrix = opt_dict["drawer_cad_scale_matrix"][drawer_i]
        drawer_cad_to_origin = opt_dict["drawer_cad_to_origin"][drawer_i]
        drawer_cad_mesh_verts = opt_dict["drawer_cad_mesh_verts"][drawer_i]
        drawer_cad_mesh_faces = opt_dict["drawer_cad_mesh_faces"][drawer_i]
        drawer_interact_info = opt_dict["drawer_interact_info"][drawer_i]

        additional_rotation = from_tensor_to_rotation_matrix(drawer_additional_rotation)
        drawer_scale_matrix = torch.eye(4).to(device)
        drawer_scale_matrix[torch.arange(3), torch.arange(3)] = torch.exp(drawer_scale)
        drawer_scale_matrix[drawer_mesh_short_axis, drawer_mesh_short_axis] = cad_short_scale
        drawer_translate_matrix = torch.eye(4).to(device)
        drawer_translate_matrix[:3, 3] = drawer_translate

        box_primitive_verts_torch = box_primitive_p3d.verts_packed().clone()
        box_primitive_verts_torch_pad = torch.nn.functional.pad(box_primitive_verts_torch, (0, 1), "constant",
                                                                value=1)
        total_transform = drawer_translate_matrix @ additional_rotation @ drawer_rotation_matrix @ drawer_scale_matrix
        drawer_transformed_verts = (total_transform @ box_primitive_verts_torch_pad.T).T
        drawer_transformed_verts = drawer_transformed_verts[..., :3] / drawer_transformed_verts[..., -1:]



        # re-align the drawer direction
        if stem in train_image_filenames:
            target_dataset = pipeline.datamanager.train_dataset
            target_idx = train_image_filenames.index(stem)
        elif stem in eval_image_filenames:
            target_dataset = pipeline.datamanager.eval_dataset
            target_idx = eval_image_filenames.index(stem)
        else:
            assert False

        idx = mask_id - 1
        mark = f"{stem}_{idx}"
        drawer_interact_info = interact_info[mark]

        R_list = []
        tvec_list = []
        image_size_list = []
        camera_matrix_list = []

        target_camera = deepcopy(target_dataset.cameras[target_idx])
        c2w = target_camera.camera_to_worlds.reshape(3, 4).cuda().clone()
        fx = float(target_camera.fx)
        fy = float(target_camera.fy)
        cx = float(target_camera.cx)
        cy = float(target_camera.cy)
        H, W = int(target_camera.height), int(target_camera.width)
        resolution = [H, W]

        c2w[..., :3, 1:3] *= -1

        w2c = torch.inverse(torch.cat([c2w, torch.from_numpy(
            np.array([0, 0, 0, 1]).reshape(1, 4)
        ).to(c2w.device)], dim=0)).unsqueeze(0)
        w2c_R = w2c[:, :3, :3].float()

        R_list.append(w2c_R.float().to(device))
        tvec_list.append(w2c[:, :3, 3].float().to(device))
        image_size_list.append(torch.from_numpy(np.array(resolution).reshape(-1, 2)).to(device))
        camera_matrix_list.append(torch.from_numpy(
            np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
        ).reshape(1, 3, 3).float().to(device))

        render_cameras = cameras_from_opencv_projection(R=torch.cat(R_list, dim=0),
                                                        tvec=torch.cat(tvec_list, dim=0),
                                                        image_size=torch.cat(image_size_list, dim=0),
                                                        camera_matrix=torch.cat(camera_matrix_list, dim=0)
                                                        )

        lights = AmbientLights(device=device)
        silhouette_images_pmesh, fragments_pmesh = renderer_silhouette(drawer_mesh_pt3d,
                                                                       cameras=render_cameras[0],
                                                                       lights=lights)

        drawer_mesh_pt3d_verts = drawer_mesh_pt3d.verts_packed().clone()
        drawer_mesh_pt3d_faces = drawer_mesh_pt3d.faces_packed().clone()

        trimesh.exchange.export.export_mesh(
            trimesh.Trimesh(
                drawer_mesh_pt3d_verts.cpu().numpy(),
                drawer_mesh_pt3d_faces.cpu().numpy(),
            ), os.path.join(f"{ckpt_dir}/drawers/logs/raw/{drawer_i}.ply")
        )

        drawer_mesh_pt3d_faces_flatten = drawer_mesh_pt3d_faces.reshape(-1)
        face_verts_3d = drawer_mesh_pt3d_verts[drawer_mesh_pt3d_faces_flatten].reshape(drawer_mesh_pt3d_faces.shape[0], 3,
                                                                                       3)

        pixel_3d_loc = interpolate_face_attributes(fragments_pmesh.pix_to_face, fragments_pmesh.bary_coords, face_verts_3d)

        pixel_3d_loc = pixel_3d_loc.reshape(H, W, 3)
        pixel_3d_mask = fragments_pmesh.pix_to_face.reshape(H, W) >= 0
        pixel_afford = drawer_interact_info['affordance']
        pixel_afford = torch.from_numpy(pixel_afford.reshape(H, W)).to(pixel_3d_mask.device)
        pixel_afford[torch.logical_not(pixel_3d_mask)] = -1e5
        pixel_afford = pixel_afford.reshape(-1)
        pixel_3d_loc = pixel_3d_loc.reshape(-1, 3)
        max_afford_pixel = torch.argmax(pixel_afford)

        vis_image = np.array(Image.open(str(target_dataset._dataparser_outputs.image_filenames[target_idx])).convert("RGB"),
                             dtype=np.float32) / 255.0
        vis_image = vis_image.reshape(-1, 3)
        vis_image[int(max_afford_pixel)] = np.array([1., 0., 0.])
        vis_image = vis_image.reshape(H , W , 3)
        Image.fromarray(np.clip(vis_image * 255, 0, 255).astype(np.uint8)).save(
            os.path.join(f"{ckpt_dir}/drawers/logs/affordmax", f"vis_line_{drawer_i}.png"))

        max_afford_pixel_3d_loc = pixel_3d_loc[max_afford_pixel]
        max_afford_pixel_3d_loc = max_afford_pixel_3d_loc.reshape(1, 3)
        max_afford_pixel_3d_loc_pad = torch.nn.functional.pad(max_afford_pixel_3d_loc, (0, 1),
                                                              "constant", value=1)
        max_afford_pixel_3d_loc_original_pad = max_afford_pixel_3d_loc_pad @ torch.inverse(total_transform.clone()).T
        max_afford_pixel_3d_loc_original = max_afford_pixel_3d_loc_original_pad[:,
                                           :3] / max_afford_pixel_3d_loc_original_pad[:, 3:]

        afford_y = max_afford_pixel_3d_loc_original[..., 1] + 0.5
        afford_z = max_afford_pixel_3d_loc_original[..., 2] + 0.5

        rotation_exist = kinematic_imap[drawer_interact_info['phy']['kinematic']] == 'rotation'
        #
        # print("drawer_i: ", drawer_i)
        # print("stem: ", stem)
        # print("idx: ", idx)
        # print("afford_y: ", afford_y)
        # print("afford_z: ", afford_z)

        if rotation_exist:

            start_pt = int(drawer_interact_info['axis'][1] * H), int(drawer_interact_info['axis'][0] * W)
            end_pt = int(drawer_interact_info['axis'][3] * H), int(drawer_interact_info['axis'][2] * W)

            line_h = np.linspace(start_pt[0], end_pt[0], 600).astype(np.int32).reshape(-1)
            line_w = np.linspace(start_pt[1], end_pt[1], 600).astype(np.int32).reshape(-1)

            valid_pts = np.logical_and(np.logical_and(line_h >= 0, line_h < H), np.logical_and(line_w >= 0, line_w < W))
            line_h = line_h[valid_pts]
            line_w = line_w[valid_pts]

            vis_line = np.zeros((H, W))
            vis_line[line_h, line_w] = 1.0
            line_valid = vis_line > 0
            vis_image = np.array(Image.open(str(target_dataset._dataparser_outputs.image_filenames[target_idx])).convert("RGB"), dtype=np.float32) / 255.0
            vis_image[vis_line > 0] = vis_image[vis_line > 0] * 0.7 + np.array([0., 1., 0.]) * 0.3
            Image.fromarray(np.clip(vis_image * 255, 0, 255).astype(np.uint8)).save(os.path.join(f"{ckpt_dir}/drawers/logs/rotline", f"vis_line_{drawer_i}.png"))

            line = torch.from_numpy(line_valid).to(pixel_3d_loc.device)
            line = torch.logical_and(line, pixel_3d_mask).reshape(-1)

            if not torch.any(line):
                print("invalid rot line")
            else:
                line_afford_pixel_3d_loc = pixel_3d_loc[line].reshape(-1, 3)
                line_afford_pixel_3d_loc_pad = torch.nn.functional.pad(line_afford_pixel_3d_loc, (0, 1),
                                                                      "constant", value=1)
                line_afford_pixel_3d_loc_original_pad = line_afford_pixel_3d_loc_pad @ torch.inverse(total_transform.clone()).T
                line_afford_pixel_3d_loc_original = line_afford_pixel_3d_loc_original_pad[:,
                                                   :3] / line_afford_pixel_3d_loc_original_pad[:, 3:]

                line_afford_y = line_afford_pixel_3d_loc_original[..., 1] + 0.5
                line_afford_z = line_afford_pixel_3d_loc_original[..., 2] + 0.5
                total_cnt = line_afford_z.shape[0]
                high_cnt = torch.count_nonzero(line_afford_z >= 0.75)
                low_cnt = torch.count_nonzero(line_afford_z <= 0.25)

                # print("line_afford_y: ", line_afford_y)
                # print("line_afford_z: ", line_afford_z)
                #
                # print("total_cnt: ", total_cnt)
                # print("high_cnt: ", high_cnt)
                # print("low_cnt: ", low_cnt)

                if afford_z >= 0.75 and low_cnt < 0.5 * total_cnt:
                    rotation_exist = False
                if afford_z <= 0.25 and high_cnt < 0.5 * total_cnt:
                    rotation_exist = False

        if str(drawer_i) in overwrite_articulate.keys():
            interact_type = overwrite_articulate[str(drawer_i)]
            if interact_type == "1.1":
                afford_y = 1.0
            elif interact_type == "1.2":
                afford_y = 0.0
            elif interact_type == "2":
                afford_z = 0.5
            elif interact_type == "3.1":
                afford_z = 1.0
            elif interact_type == "3.2":
                afford_z = 0.0
            else:
                assert False
            print("overwrite drawer ", drawer_i, " articulation type to ", interact_type)
        else:
            if afford_y > 0.75:
                interact_type = "1.1"
            elif afford_y < 0.25:
                interact_type = "1.2"
            elif afford_z > 0.25 and afford_z < 0.75:
                interact_type = "2"
            elif afford_z >= 0.75:
                if rotation_exist:
                    interact_type = "3.1"
                else:
                    interact_type = "3.3"
            else:  # afford_z <= 0.25:
                if rotation_exist:
                    interact_type = "3.2"
                else:
                    interact_type = "3.3"
            print("drawer ", drawer_i, " has the articulation type of ", interact_type)

        opt_dict["afford"].append((afford_y, afford_z))
        opt_dict["interact_type"].append(interact_type)

print("interact_type: ", opt_dict["interact_type"])
# set up optimizer
lr = 1e-3
optimizer = torch.optim.SGD(opt_dict["params"], lr=lr, momentum=0.9)
Niter = int(300 * num_total_drawer)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, Niter, eta_min=lr * 0.05)

basic_width_revolute = 0.05
basic_width_prismatic = 0.4

# start optimization
opt_loop = tqdm(range(Niter))
for iter in opt_loop:
    optimizer.zero_grad()

    drawer_i = np.random.randint(num_total_drawer)
    # drawer_i = 0

    drawer_translate = opt_dict["params"][3*drawer_i+0]
    drawer_scale = opt_dict["params"][3*drawer_i+1]
    drawer_additional_rotation = opt_dict["params"][3*drawer_i+2]

    drawer_mesh_pt3d = opt_dict["drawer_mesh_pt3d"][drawer_i]

    camera_stems = opt_dict["camera_stems"][drawer_i]
    n_views = min(len(camera_stems), 10)

    drawer_mesh_short_axis = opt_dict["drawer_mesh_short_axis"][drawer_i]

    additional_rotation = from_tensor_to_rotation_matrix(drawer_additional_rotation)
    drawer_scale_matrix = torch.eye(4).to(device)
    drawer_scale_matrix[torch.arange(3), torch.arange(3)] = torch.exp(drawer_scale)
    drawer_scale_matrix[drawer_mesh_short_axis, drawer_mesh_short_axis] = cad_short_scale
    drawer_translate_matrix = torch.eye(4).to(device)
    drawer_translate_matrix[:3, 3] = drawer_translate

    box_primitive_p3d = opt_dict["box_primitive_p3d"][drawer_i]

    drawer_rotation_matrix = opt_dict["drawer_rotation_matrix"][drawer_i]

    box_primitive_verts_torch = box_primitive_p3d.verts_packed().clone()
    box_primitive_verts_torch_pad = torch.nn.functional.pad(box_primitive_verts_torch, (0, 1), "constant",
                                                            value=1)
    drawer_transformed_verts = (
            drawer_translate_matrix @ additional_rotation @ drawer_rotation_matrix @ drawer_scale_matrix @ box_primitive_verts_torch_pad.T).T
    drawer_transformed_verts = drawer_transformed_verts[..., :3] / drawer_transformed_verts[..., -1:]

    drawer_offset = drawer_transformed_verts - box_primitive_verts_torch
    box_primitive_p3d_transformed = box_primitive_p3d.offset_verts(drawer_offset)

    stems = np.random.choice(len(camera_stems), n_views, replace=False)
    loss_mask = 0
    loss_depth = 0

    renderer_silhouette = opt_dict["renderer_silhouette"][drawer_i]
    render_cameras = opt_dict["render_cameras"][drawer_i]

    lights = AmbientLights(device=device)

    sam_masks = opt_dict["sam_masks"][drawer_i]
    depth_pmesh_list = opt_dict["depth_pmesh_list"][drawer_i]

    for stem_i in stems.tolist():
        silhouette_images, fragments = renderer_silhouette(box_primitive_p3d_transformed,
                                                           cameras=render_cameras[stem_i], lights=lights)
        silhouette = silhouette_images.reshape(H, W, 4)[..., -1]

        sam_mask = sam_masks[stem_i].to(device)
        loss_mask += ((silhouette - sam_mask) ** 2).mean()

        depth = fragments.zbuf[..., 0].reshape(H, W)
        depth_pmesh = depth_pmesh_list[stem_i].to(device)

        valid = torch.logical_and(
            torch.logical_and(depth > 0, depth_pmesh > 0),
            sam_mask > 0.
        )
        if torch.any(valid):
            loss_depth += ((depth_pmesh[valid] - depth[valid]) ** 2).mean()

    loss_mask = loss_mask / stems.shape[0]
    loss_depth = loss_depth / stems.shape[0]

    # global optimize
    def sdf_box(points):
        """
        Calculate the Signed Distance Function (SDF) for a point to the surface of a unit box centered at the origin.

        :param point: A tuple or list (x, y, z) representing the coordinates of the point.
        :param box_half_size: A tuple representing the half-size of the box. Default is (1, 1, 1) for a unit box.
        :return: The SDF value as a float.
        """
        box_half_size = (0.5, 0.5, 0.5)

        points_abs = torch.abs(points)
        bx, by, bz = box_half_size

        dist_x = points_abs[:, 0] - bx
        dist_y = points_abs[:, 1] - by
        dist_z = points_abs[:, 2] - bz

        dx = torch.where(dist_x > 0, dist_x, 0)
        dy = torch.where(dist_y > 0, dist_y, 0)
        dz = torch.where(dist_z > 0, dist_z, 0)

        outside_distance = torch.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        # max(px - bx, py - by, pz - bz)
        dist_xyz_max = torch.where(dist_x > dist_y, dist_x, dist_y)
        dist_xyz_max = torch.where(dist_xyz_max > dist_z, dist_xyz_max, dist_z)

        # inside_distance = min(max(px - bx, py - by, pz - bz), 0)
        inside_distance = torch.where(dist_xyz_max > 0, 0, dist_xyz_max)

        return outside_distance + inside_distance


    sampled_pts_list = []
    for drawer_glb_opt_i in range(num_total_drawer):
        if drawer_glb_opt_i == drawer_i:
            continue
        drawer_translate_glb_opt = opt_dict["params"][3 * drawer_glb_opt_i + 0]
        drawer_scale_glb_opt = opt_dict["params"][3 * drawer_glb_opt_i + 1]
        drawer_additional_rotation_glb_opt = opt_dict["params"][3 * drawer_glb_opt_i + 2]
        drawer_mesh_short_axis_glb_opt = opt_dict["drawer_mesh_short_axis"][drawer_glb_opt_i]
        box_primitive_p3d_glb_opt = opt_dict["box_primitive_p3d"][drawer_glb_opt_i]
        drawer_rotation_matrix_glb_opt = opt_dict["drawer_rotation_matrix"][drawer_glb_opt_i]
        interact_type_glb_opt = opt_dict["interact_type"][drawer_glb_opt_i]

        additional_rotation_glb_opt = from_tensor_to_rotation_matrix(drawer_additional_rotation_glb_opt)
        drawer_scale_matrix_glb_opt = torch.eye(4).to(device)
        drawer_scale_matrix_glb_opt[torch.arange(3), torch.arange(3)] = torch.exp(drawer_scale_glb_opt)
        if interact_type_glb_opt == "2" or interact_type_glb_opt == "3.3":
            drawer_scale_matrix_glb_opt[drawer_mesh_short_axis_glb_opt, drawer_mesh_short_axis_glb_opt] = basic_width_prismatic
        else:
            drawer_scale_matrix_glb_opt[drawer_mesh_short_axis_glb_opt, drawer_mesh_short_axis_glb_opt] = basic_width_revolute

        drawer_translate_matrix_glb_opt = torch.eye(4).to(device)
        drawer_translate_matrix_glb_opt[:3, 3] = drawer_translate_glb_opt

        drawer_total_transform_glb_opt = drawer_translate_matrix_glb_opt @ additional_rotation_glb_opt @ drawer_rotation_matrix_glb_opt @ drawer_scale_matrix_glb_opt

        sampled_pts = sample_points_from_meshes(box_primitive_p3d_glb_opt, num_samples=10000).detach().reshape(-1, 3)
        sampled_pts_pad = torch.nn.functional.pad(sampled_pts, (0, 1), "constant", value=1)
        sampled_pts_transformed = sampled_pts_pad @ drawer_total_transform_glb_opt.T
        sampled_pts_transformed = sampled_pts_transformed[:, :3] / sampled_pts_transformed[:, 3:]

        sampled_pts_list.append(sampled_pts_transformed)

    sampled_pts_glb_opt = torch.cat(sampled_pts_list, dim=0)

    drawer_scale_matrix_glb_opt = torch.eye(4).to(device)
    drawer_scale_matrix_glb_opt[torch.arange(3), torch.arange(3)] = torch.exp(drawer_scale)

    interact_type_target = opt_dict["interact_type"][drawer_i]
    if interact_type_target == "2" or interact_type_target == "3.3":
        drawer_scale_matrix_glb_opt[
            drawer_mesh_short_axis, drawer_mesh_short_axis] = basic_width_prismatic
    else:
        drawer_scale_matrix_glb_opt[
            drawer_mesh_short_axis, drawer_mesh_short_axis] = basic_width_revolute

    # drawer_scale_matrix_glb_opt[drawer_mesh_short_axis, drawer_mesh_short_axis] = basic_width

    drawer_total_transformed_target_glb_opt = drawer_translate_matrix @ additional_rotation @ drawer_rotation_matrix @ drawer_scale_matrix_glb_opt

    sampled_pts_glb_opt_pad = torch.nn.functional.pad(sampled_pts_glb_opt, (0, 1), "constant", value=1)
    sampled_pts_glb_opt_transformed = sampled_pts_glb_opt_pad @ torch.inverse(drawer_total_transformed_target_glb_opt).T
    sampled_pts_glb_opt_transformed = sampled_pts_glb_opt_transformed[:, :3] / sampled_pts_glb_opt_transformed[:, 3:]

    loss_glb = torch.nn.functional.relu(-sdf_box(sampled_pts_glb_opt_transformed) + 0.01).mean()

    cad_points = sample_points_from_meshes(box_primitive_p3d_transformed)
    mesh_points = sample_points_from_meshes(drawer_mesh_pt3d)
    loss_chamfer, _ = chamfer_distance(cad_points, mesh_points)

    # print("loss_glb: ", loss_glb.grad_fn, loss_glb)

    loss = 5 * loss_mask + 0.5 * loss_depth + 10 * loss_glb + 30 * loss_chamfer

    loss.backward()

    # for p in opt_dict["params"]:
    #     print(p.grad, end=" ")

    optimizer.step()
    scheduler.step()

    with torch.no_grad():
        # print(f"loss_mask={float(loss_mask):.6f}; loss_depth={float(loss_depth):.6f}")
        opt_loop.set_description(f"loss_mask={float(loss_mask):.6f}; loss_depth={float(loss_depth):.6f}; loss_glb={float(loss_glb):.6f};")
# save results
with torch.no_grad():
    for drawer_i in range(num_total_drawer):

        drawer_translate = opt_dict["params"][3 * drawer_i + 0]
        drawer_scale = opt_dict["params"][3 * drawer_i + 1]
        drawer_additional_rotation = opt_dict["params"][3 * drawer_i + 2]

        camera_stems = opt_dict["camera_stems"][drawer_i]
        drawer_mesh_short_axis = opt_dict["drawer_mesh_short_axis"][drawer_i]
        box_primitive_p3d = opt_dict["box_primitive_p3d"][drawer_i]
        drawer_rotation_matrix = opt_dict["drawer_rotation_matrix"][drawer_i]
        renderer_silhouette = opt_dict["renderer_silhouette"][drawer_i]
        # render_cameras = opt_dict["render_cameras"][drawer_i]
        # sam_masks = opt_dict["sam_masks"][drawer_i]
        # depth_pmesh_list = opt_dict["depth_pmesh_list"][drawer_i]
        stem = opt_dict["stem"][drawer_i]
        drawer_mesh_pt3d = opt_dict["drawer_mesh_pt3d"][drawer_i]
        mask_id = opt_dict["mask_id"][drawer_i]
        drawer_cad_scale_matrix = opt_dict["drawer_cad_scale_matrix"][drawer_i]
        drawer_cad_to_origin = opt_dict["drawer_cad_to_origin"][drawer_i]
        drawer_cad_mesh_verts = opt_dict["drawer_cad_mesh_verts"][drawer_i]
        drawer_cad_mesh_faces = opt_dict["drawer_cad_mesh_faces"][drawer_i]
        drawer_interact_info = opt_dict["drawer_interact_info"][drawer_i]

        afford = opt_dict["afford"][drawer_i]
        afford_y, afford_z = afford

        interact_type = opt_dict["interact_type"][drawer_i]

        additional_rotation = from_tensor_to_rotation_matrix(drawer_additional_rotation)
        drawer_scale_matrix = torch.eye(4).to(device)
        drawer_scale_matrix[torch.arange(3), torch.arange(3)] = torch.exp(drawer_scale)
        drawer_scale_matrix[drawer_mesh_short_axis, drawer_mesh_short_axis] = cad_short_scale
        drawer_translate_matrix = torch.eye(4).to(device)
        drawer_translate_matrix[:3, 3] = drawer_translate

        box_primitive_verts_torch = box_primitive_p3d.verts_packed().clone()
        box_primitive_verts_torch_pad = torch.nn.functional.pad(box_primitive_verts_torch, (0, 1), "constant",
                                                                value=1)
        total_transform = drawer_translate_matrix @ additional_rotation @ drawer_rotation_matrix @ drawer_scale_matrix
        drawer_transformed_verts = (total_transform @ box_primitive_verts_torch_pad.T).T
        drawer_transformed_verts = drawer_transformed_verts[..., :3] / drawer_transformed_verts[..., -1:]

        # re-align the drawer direction
        if stem in train_image_filenames:
            target_dataset = pipeline.datamanager.train_dataset
            target_idx = train_image_filenames.index(stem)
        elif stem in eval_image_filenames:
            target_dataset = pipeline.datamanager.eval_dataset
            target_idx = eval_image_filenames.index(stem)
        else:
            assert False

        # find the ry and rz
        door_extent = np.array([1e-4, 1, 1]).reshape(3)

        box_base_verts_pad = np.pad(box_base_verts, ((0, 0), (0, 1)), constant_values=(0, 1))
        box_transformed_verts_pad = box_base_verts_pad @ total_transform.cpu().numpy().T
        box_transformed_verts = box_transformed_verts_pad[..., :3] / box_transformed_verts_pad[..., -1:]

        delta_z_list = []
        for edge in box_base_edges:
            delta_z = np.abs(box_transformed_verts[edge[0]][-1] - box_transformed_verts[edge[1]][-1])
            delta_z_list.append(float(delta_z))
        delta_zs = np.array(delta_z_list).reshape(-1)
        max_delta_z_edge_idxs = np.argsort(delta_zs)[-4:]
        joint_pos_list = []
        for max_delta_z_edge_idx in max_delta_z_edge_idxs:
            joint_edge = box_base_edges[max_delta_z_edge_idx]
            joint_pos = (box_transformed_verts[joint_edge[0]] + box_transformed_verts[
                joint_edge[1]]).reshape(1, 3) / 2
            joint_pos_list.append(joint_pos)
        joint_pos_list = np.concatenate(joint_pos_list, axis=0)
        joint_pos = joint_pos_list[0:1]
        joint_distance = np.sum((joint_pos - joint_pos_list) ** 2, axis=-1).reshape(4)
        left_joints = np.argsort(joint_distance)[:2]
        right_joints = np.argsort(joint_distance)[2:]

        drawer_center = np.mean(joint_pos_list, axis=0).reshape(3)
        ry = joint_pos_list[right_joints[0]] - joint_pos_list[left_joints[0]]
        ry = ry.reshape(3)
        sy = float(np.linalg.norm(ry, axis=0, ord=2))
        ry = ry / sy

        right_pos = joint_pos_list[right_joints[0]].reshape(1, 3)
        left_pos = joint_pos_list[left_joints[0]].reshape(1, 3)
        both_pos = np.concatenate([right_pos, left_pos], axis=0)

        target_camera = deepcopy(target_dataset.cameras[target_idx])
        c2w = target_camera.camera_to_worlds.reshape(3, 4).clone().cpu()
        c2w[:3, 1:3] *= -1
        w2c = torch.inverse(torch.cat([c2w, torch.from_numpy(
            np.array([0, 0, 0, 1]).reshape(1, 4)
        ).to(c2w.device)], dim=0)).numpy()
        both_pos_pad = np.pad(both_pos, ((0, 0), (0, 1)), constant_values=(0, 1))
        both_pos_cv_pad = both_pos_pad @ w2c.T
        both_pos_cv = both_pos_cv_pad[:, :3] / both_pos_cv_pad[:, 3:]
        if both_pos_cv[0, 0] - both_pos_cv[1, 0] < 0:
            ry *= -1

        rz = box_transformed_verts[box_base_edges[max_delta_z_edge_idxs[0]][0]] - box_transformed_verts[
            box_base_edges[max_delta_z_edge_idxs[0]][1]]
        rz = rz.reshape(3)
        if rz[-1] < 0:
            rz = rz * -1
        sz = float(np.linalg.norm(rz, axis=0, ord=2))
        rz = rz / sz

        rx = np.cross(ry, rz).reshape(3)
        drawer_rot = np.stack([rx, ry, rz], axis=1)
        drawer_transform_new_rot = np.eye(4)
        drawer_transform_new_rot[:3, :3] = drawer_rot
        drawer_transform_new_trans = np.eye(4)
        drawer_transform_new_trans[:3, 3] = drawer_center
        drawer_transform_new_scale = np.eye(4)
        drawer_transform_new_scale[np.arange(3), np.arange(3)] = [1e-4, sy, sz]

        total_transform = drawer_transform_new_trans @ drawer_transform_new_rot @ drawer_transform_new_scale
        total_transform = torch.from_numpy(total_transform).to(device).float()


        # rx = np.cross(ry, rz).reshape(3)
        _ry = ry.copy()
        _rz = rz.copy()

        _sy = sy + 0.
        _sz = sz + 0.

        if afford_y > 0.75:
            interact_type = "1.1"
            # no need to tune the direction
            ry = _ry
            rz = _rz
            sy = _sy
            sz = _sz
        elif afford_y < 0.25:
            interact_type = "1.2"
            ry = -_ry
            rz = _rz
            sy = _sy
            sz = _sz
        elif afford_z > 0.25 and afford_z < 0.75:
            interact_type = "2"
            ry = _ry
            rz = _rz
            sy = _sy
            sz = _sz
        elif afford_z >= 0.75:
            if rotation_exist:
                interact_type = "3.1"
                ry = _rz
                rz = -_ry
                sy = _sz
                sz = _sy
            else:
                interact_type = "3.3"
                ry = _ry
                rz = _rz
                sy = _sy
                sz = _sz
        else:  # afford_z <= 0.25:
            if rotation_exist:
                interact_type = "3.2"
                ry = -_rz
                rz = -_ry
                sy = _sz
                sz = _sy
            else:
                interact_type = "3.3"
                ry = _ry
                rz = _rz
                sy = _sy
                sz = _sz

        rx = np.cross(ry, rz).reshape(3)
        drawer_rot = np.stack([rx, ry, rz], axis=1)
        drawer_transform_new_rot = np.eye(4)
        drawer_transform_new_rot[:3, :3] = drawer_rot
        drawer_transform_new_trans = np.eye(4)
        drawer_transform_new_trans[:3, 3] = drawer_center
        drawer_transform_new_scale = np.eye(4)
        drawer_transform_new_scale[np.arange(3), np.arange(3)] = [1e-4, sy, sz]

        drawer_transform_new = drawer_transform_new_trans @ drawer_transform_new_rot @ drawer_transform_new_scale

        total_transform_matrix = drawer_transform_new.reshape(4, 4) @ drawer_cad_scale_matrix @ drawer_cad_to_origin.reshape(4, 4)
        total_transform_matrix = torch.from_numpy(total_transform_matrix).to(device).float()

        drawer_cad_mesh_verts_pad = np.pad(drawer_cad_mesh_verts, ((0, 0), (0, 1)), constant_values=(0, 1))
        drawer_cad_mesh_verts_pad = torch.from_numpy(drawer_cad_mesh_verts_pad).to(device).float()
        drawer_transformed_verts = (total_transform_matrix @ drawer_cad_mesh_verts_pad.T).T
        drawer_transformed_verts = drawer_transformed_verts[..., :3] / drawer_transformed_verts[..., -1:]

        drawer_template = trimesh.Trimesh(drawer_transformed_verts.cpu().numpy(), drawer_cad_mesh_faces,
                                          vertex_colors=box_base_verts_color)

        trimesh.exchange.export.export_mesh(drawer_template, f"{ckpt_dir}/drawers/results/drawer_{drawer_i}.ply")

        with open(f"{ckpt_dir}/drawers/results/drawer_{drawer_i}.pkl", 'wb') as f:
            pickle.dump({
                "transform": drawer_transform_new,
                "interact": interact_type,
            }, f)

        # collision vis
        if interact_type == "2" or interact_type == "3.3":
            sx = basic_width_prismatic
        else:
            sx = basic_width_revolute

        drawer_transform_new_scale_collision = np.eye(4)
        drawer_transform_new_scale_collision[np.arange(3), np.arange(3)] = [sx, sy, sz]

        drawer_transform_new_collision = drawer_transform_new_trans @ drawer_transform_new_rot @ drawer_transform_new_scale_collision

        total_transform_matrix_collision = drawer_transform_new_collision.reshape(4,
                                                              4) @ drawer_cad_scale_matrix @ drawer_cad_to_origin.reshape(
            4, 4)
        total_transform_matrix_collision = torch.from_numpy(total_transform_matrix_collision).to(device).float()

        drawer_cad_mesh_verts_pad = np.pad(drawer_cad_mesh_verts, ((0, 0), (0, 1)), constant_values=(0, 1))
        drawer_cad_mesh_verts_pad = torch.from_numpy(drawer_cad_mesh_verts_pad).to(device).float()
        drawer_transformed_verts_collision = (total_transform_matrix_collision @ drawer_cad_mesh_verts_pad.T).T
        drawer_transformed_verts_collision = drawer_transformed_verts_collision[..., :3] / drawer_transformed_verts_collision[..., -1:]

        drawer_template_collision = trimesh.Trimesh(drawer_transformed_verts_collision.cpu().numpy(), drawer_cad_mesh_faces,
                                          vertex_colors=box_base_verts_color)

        trimesh.exchange.export.export_mesh(drawer_template_collision, f"{ckpt_dir}/drawers/logs/collision/drawer_{drawer_i}.ply")