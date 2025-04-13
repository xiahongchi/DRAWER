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
from scipy.ndimage import binary_dilation, binary_erosion
import argparse

CONSOLE = Console(width=120)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--image_dir", type=str, required=True)
parser.add_argument("--sdf_dir", type=str, required=True)

args = parser.parse_args()

ckpt_dir = args.sdf_dir
mask_src_dir = os.path.join(args.data_dir, "grounded_sam")
mask_save_dir = os.path.join(args.data_dir, "grounded_sam")
input_dir = args.image_dir


load_config = Path(f"{ckpt_dir}/config.yml")

config, pipeline, checkpoint_path = eval_setup(load_config)

image_filenames = pipeline.datamanager.train_dataset._dataparser_outputs.image_filenames
train_image_filenames = [os.path.splitext(os.path.basename(str(elem)))[0] for elem in image_filenames]
image_filenames = pipeline.datamanager.eval_dataset._dataparser_outputs.image_filenames
eval_image_filenames = [os.path.splitext(os.path.basename(str(elem)))[0] for elem in image_filenames]


white_mesh_path = f"{ckpt_dir}/mesh.ply"
mask_mesh_path = f"{ckpt_dir}/mask_mesh.ply"
mask_json_path = f"{ckpt_dir}/mask_mesh.pkl"
import pymeshlab
ms = pymeshlab.MeshSet()
ms.load_new_mesh(white_mesh_path)
target_faces_num = 5000000
print("simplify mesh...")
ms.simplification_quadric_edge_collapse_decimation(targetfacenum=target_faces_num)
m = ms.current_mesh()
v_np = m.vertex_matrix()
f_np = m.face_matrix()
white_mesh = trimesh.Trimesh(v_np, f_np, process=False)
# white_mesh = trimesh.exchange.load.load_mesh(white_mesh_path)
# v_np = white_mesh.vertices
# f_np = white_mesh.faces



vertices = torch.tensor(v_np.astype(np.float32)).float()
pos_idx = torch.tensor(f_np.astype(np.int32))

mesh_dict = {
    'vertices': F.pad(vertices, pad=(0, 1), mode='constant', value=1.0).cuda().contiguous(),
    'pos_idx': pos_idx.cuda().contiguous(),
}

glctx = dr.RasterizeGLContext(output_db=False)


# mask_save_dir = "/home/hongchix/codes/segment-anything/results/2e67a32314/"

os.makedirs(mask_save_dir, exist_ok=True)
os.makedirs(os.path.join(mask_save_dir, "mesh_mask_vis"), exist_ok=True)
os.makedirs(os.path.join(mask_save_dir, "mesh_mask"), exist_ok=True)
# src_names = sorted(os.listdir(os.path.join(mask_src_dir, "mask")))
src_names_json = os.path.join(mask_src_dir, "all.json")
with open(src_names_json, 'r') as f:
    src_names = json.load(f)
# input_dir = "/home/hongchix/scratch/scannetpp/2e67a32314/images/"


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


def rasterize_texture(mesh_dict, projection, glctx, resolution, c2w=None):

    vertices = mesh_dict['vertices']
    pos_idx = mesh_dict['pos_idx']

    vertices_clip = torch.matmul(vertices, torch.transpose(projection, 0, 1)).float().unsqueeze(0)
    rast_out, _ = dr.rasterize(glctx, vertices_clip, pos_idx, resolution=resolution)
    # rast_out = rast_out.flip([1])

    H, W = resolution
    valid = (rast_out[..., -1] > 0).reshape(H, W)
    triangle_id = (rast_out[..., -1] - 1).long().reshape(H, W)

    if c2w is None:
        depth = None
    else:
        w2c = torch.inverse(torch.cat([c2w, torch.tensor([0, 0, 0, 1]).reshape(1, 4).to(c2w.device)], dim=0))
        vert_cam = (w2c.to(vertices.device) @ vertices.permute(1, 0)).permute(1, 0)
        vert_cam = vert_cam[..., :3] / vert_cam[..., 3:4]
        depth = -vert_cam[..., -1:]
        depth_inverse = 1 / (depth + 1e-20)
        depth_inverse, _ = dr.interpolate(depth_inverse.unsqueeze(0).contiguous(), rast_out, pos_idx)
        depth = 1 / (depth_inverse + 1e-20)
        depth = depth.reshape(H, W)


    return valid, triangle_id, depth


mask_list = []
mesh_mask_save_dict = {}
mask_meshes = []
for name in src_names:
    mask_info_json_path = os.path.join(mask_src_dir, name)
    idxs = src_names[name]
    with open(mask_info_json_path, 'r') as f:
        mask_info_json = json.load(f)
    stem = os.path.splitext(os.path.basename(name))[0]

    if stem in train_image_filenames:
        target_dataset = pipeline.datamanager.train_dataset
        target_idx = train_image_filenames.index(stem)
    elif stem in eval_image_filenames:
        target_dataset = pipeline.datamanager.eval_dataset
        target_idx = eval_image_filenames.index(stem)
    else:
        assert False

    target_camera = deepcopy(target_dataset.cameras[target_idx])
    c2w = target_camera.camera_to_worlds.reshape(3, 4).cuda()
    H, W = int(target_camera.height), int(target_camera.width)
    resolution = [H, W]

    n = 0.001
    f = 10  # infinite

    # offset = np.linspace(-0.5, 0.5, num=6, endpoint=False)
    offset = [0]

    fx = float(target_camera.fx)
    fy = float(target_camera.fy)
    cx = float(target_camera.cx)
    cy = float(target_camera.cy)

    valid_list = []
    triangle_id_list = []

    for offset_x in offset:
        for offset_y in offset:
            _fx = fx + offset_x
            _fy = fy + offset_y

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

            camera_projmat = torch.from_numpy(camera_projmat)
            bottom = torch.tensor([0, 0, 0, 1]).reshape(1, 4)

            i_pose = c2w.clone()
            i_pose[..., :3, 1:3] *= -1
            square_pose = torch.cat((i_pose, bottom.to(i_pose.device)), dim=0)
            projection = camera_projmat.to(square_pose.device) @ torch.inverse(square_pose)

            valid, triangle_id, _ = rasterize_texture(mesh_dict, projection, glctx, resolution)

            valid_list.append(valid)
            triangle_id_list.append(triangle_id)

    valid = torch.stack(valid_list, dim=-1)
    triangle_id = torch.stack(triangle_id_list, dim=-1)

    with open(os.path.join(mask_src_dir, "mask", stem+".json"), 'r') as f:
        mask_src_json = json.load(f)
    with open(os.path.join(mask_src_dir, "mask_data", stem+".pkl"), 'rb') as f:
        mask_src_data = pickle.load(f)["mask"]
    for mask_dict in mask_src_json:
        value = mask_dict["value"]
        if value not in idxs:
            continue
        if value > 0:
            idx = value - 1
            binary_mask = mask_src_data[idx].reshape(H, W, 1)
            binary_mask_np = binary_mask.reshape(H, W).cpu().numpy()
            binary_mask_np = binary_erosion(binary_mask_np, iterations=2)
            binary_mask = torch.from_numpy(binary_mask_np).unsqueeze(-1)
            mask_triangle_ids = torch.unique(
                triangle_id[torch.logical_and(binary_mask.to(valid.device), valid)].reshape(-1)
            ).cpu().numpy()

            marked_vert_idxs = np.unique(f_np[mask_triangle_ids].reshape(-1))
            dilation = 2
            dilate_idxs = []
            to_dilate_idxs = marked_vert_idxs
            for dilate_i in range(dilation):
                neighbors = []
                for index in to_dilate_idxs:
                    vertex_neighbors = white_mesh.vertex_neighbors[index]
                    neighbors.extend(vertex_neighbors)

                neighbors = list(set(neighbors) - set(to_dilate_idxs))
                dilate_idxs.extend(neighbors)
                to_dilate_idxs = neighbors

            marked_vert_idxs = np.unique(np.concatenate([marked_vert_idxs, np.array(dilate_idxs).reshape(-1)], axis=0))

            keep = np.zeros((v_np.shape[0])).astype(np.bool_)
            keep[marked_vert_idxs] = True
            v_filtered, f_filtered, face_mapping = filter_mesh_from_vertices(keep, v_np, f_np)
            filtered_mesh = trimesh.Trimesh(v_filtered, f_filtered, process=False)
            # Create a graph from the mesh faces
            edges = filtered_mesh.edges_sorted.reshape((-1, 2))
            components = trimesh.graph.connected_components(edges, min_len=1, engine='scipy')
            split_meshes = []
            total_length = 0
            largest_cc = np.argmax(np.array([comp.shape[0] for comp in components]).reshape(-1), axis=0)
            keep = np.zeros((v_filtered.shape[0])).astype(np.bool_)
            keep[components[largest_cc].reshape(-1)] = True
            v_filtered, f_filtered, face_mapping_next = filter_mesh_from_vertices(keep, v_filtered, f_filtered)
            face_mapping = face_mapping[face_mapping_next]

            mesh_mask_save_dict[f"{name}_{value:0>2d}"] = {
                "mesh": (v_filtered, f_filtered),
                "idx": len(mask_list),
            }

            mask_list.append((v_filtered, f_filtered, face_mapping))
            mask_meshes.append(
                trimesh.Trimesh(v_filtered, f_filtered, vertex_colors=np.random.rand(1, 3).repeat(v_filtered.shape[0], axis=0), process=False)
            )


mask_meshes = trimesh.util.concatenate(mask_meshes)
CONSOLE.print("save mask mesh to: ", mask_mesh_path)
trimesh.exchange.export.export_mesh(mask_meshes, mask_mesh_path)
def sample_negative(mask_pixel_grid, region):
    min_h, min_w, max_h, max_w = region

    # n_pts = 30
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
    # valid = np.ones((mask_pixel_grid.shape[0], mask_pixel_grid.shape[1])).astype(np.bool_)
    # valid[:b_min_h] = False
    # valid[:, :b_min_w] = False
    # valid[b_max_h:] = False
    # valid[:, b_max_w:] = False
    # valid[ex_min_h:ex_max_h, ex_min_w:ex_max_w] = False
    # valid_mask_pixel_grid = mask_pixel_grid[valid].reshape(-1, 2)
    # negative_pts = valid_mask_pixel_grid[np.random.choice(valid_mask_pixel_grid.shape[0], min(n_pts, valid_mask_pixel_grid.shape[0]), replace=False).reshape(-1)].tolist()
    # while True:
    #     select_h = np.random.randint(
    #         max(int(min_h - delta*(max_h - min_h)), 0),
    #         min(int(max_h + delta*(max_h - min_h)), mask_pixel_grid.shape[0] - 1)
    #     )
    #     select_w = np.random.randint(
    #         max(int(min_w - delta*(max_w - min_w)), 0),
    #         min(int(max_w + delta*(max_w - min_w)), mask_pixel_grid.shape[1] - 1)
    #     )
    #
    #     ex_min_h = max(int(min_h - ex_delta * (max_h - min_h)), 0)
    #     ex_max_h = min(int(max_h + ex_delta * (max_h - min_h)), mask_pixel_grid.shape[0] - 1)
    #
    #     ex_min_w = max(int(min_w - ex_delta * (max_w - min_w)), 0)
    #     ex_max_w = min(int(max_w + ex_delta * (max_w - min_w)), mask_pixel_grid.shape[1] - 1)
    #
    #     if select_h >= ex_min_h and select_h <= ex_max_h and select_w >= ex_min_w and select_w <= ex_max_w:
    #         continue
    #     current_pts += 1
    #     negative_pts.append((select_h, select_w))
    #     if current_pts >= n_pts:
    #         break
    return negative_pts

def resize_mask(
        ref_mask: np.ndarray, longest_side: int = 256
) -> tuple[np.ndarray, int, int]:
    """
    Resize an image to have its longest side equal to the specified value.

    Args:
        ref_mask (np.ndarray): The image to be resized.
        longest_side (int, optional): The length of the longest side after resizing. Default is 256.

    Returns:
        tuple[np.ndarray, int, int]: The resized image and its new height and width.
    """
    height, width = ref_mask.shape[:2]
    if height > width:
        new_height = longest_side
        new_width = int(width * (new_height / height))
    else:
        new_width = longest_side
        new_height = int(height * (new_width / width))

    return (
        cv2.resize(
            ref_mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        ),
        new_height,
        new_width,
    )

def pad_mask(
        ref_mask: np.ndarray,
        new_height: int,
        new_width: int,
        pad_all_sides: bool = False,
) -> np.ndarray:
    """
    Add padding to an image to make it square.

    Args:
        ref_mask (np.ndarray): The image to be padded.
        new_height (int): The height of the image after resizing.
        new_width (int): The width of the image after resizing.
        pad_all_sides (bool, optional): Whether to pad all sides of the image equally. If False, padding will be added to the bottom and right sides. Default is False.

    Returns:
        np.ndarray: The padded image.
    """
    pad_height = 256 - new_height
    pad_width = 256 - new_width
    if pad_all_sides:
        padding = (
            (pad_height // 2, pad_height - pad_height // 2),
            (pad_width // 2, pad_width - pad_width // 2),
        )
    else:
        padding = ((0, pad_height), (0, pad_width))

    # Padding value defaults to '0' when the `np.pad`` mode is set to 'constant'.
    return np.pad(ref_mask, padding, mode="minimum")

def reference_to_sam_mask(
        ref_mask: np.ndarray, pad_all_sides: bool = False
) -> np.ndarray:
    """
    Convert a grayscale mask to a binary mask, resize it to have its longest side equal to 256, and add padding to make it square.

    Args:
        ref_mask (np.ndarray): The grayscale mask to be processed.
        threshold (int, optional): The threshold value for the binarization. Default is 127.
        pad_all_sides (bool, optional): Whether to pad all sides of the image equally. If False, padding will be added to the bottom and right sides. Default is False.

    Returns:
        np.ndarray: The processed binary mask.
    """

    # Convert a grayscale mask to a binary mask.
    # Values over the threshold are set to 1, values below are set to -1.
    ref_mask = np.where(ref_mask, 1, -8)

    # Resize to have the longest side 256.
    resized_mask, new_height, new_width = resize_mask(ref_mask)

    # Add padding to make it square.
    sam_mask = pad_mask(resized_mask, new_height, new_width, pad_all_sides)

    # Expand SAM mask's dimensions to 1xHxW (1x256x256).
    return np.expand_dims(sam_mask, axis=0)


image_names = sorted(os.listdir(input_dir))
frames_sam_input = {}
mask_cnt = len(mask_list)
mask_colors = np.random.rand(mask_cnt, 3)
vis_frames = []
for name in tqdm(image_names):
    image_path = os.path.join(input_dir, name)
    stem = os.path.splitext(name)[0]
    os.makedirs(os.path.join(mask_save_dir, "mesh_mask", stem), exist_ok=True)
    if stem in train_image_filenames:
        target_dataset = pipeline.datamanager.train_dataset
        target_idx = train_image_filenames.index(stem)
    elif stem in eval_image_filenames:
        target_dataset = pipeline.datamanager.eval_dataset
        target_idx = eval_image_filenames.index(stem)
    else:
        continue
        # assert False
    frames_sam_input[stem] = []

    target_camera = deepcopy(target_dataset.cameras[target_idx])
    projection = target_camera.projections.reshape(4, 4).cuda()
    c2w = target_camera.camera_to_worlds.reshape(3, 4).cuda()
    H, W = int(target_camera.height), int(target_camera.width)
    resolution = [H, W]

    pixel_grid = np.stack(np.meshgrid(
        np.linspace(0, H - 1, H).astype(np.int32),
        np.linspace(0, W - 1, W).astype(np.int32),
        indexing='ij'
    ), axis=-1)
    mask_pixel_grid = pixel_grid.copy()

    vis_frame = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    frame_valid, frame_triangle_id, frame_depth = rasterize_texture(mesh_dict, projection, glctx, resolution, c2w)

    # depth_vis_path = os.path.join(mask_save_dir, f"{stem}_depth.png")
    # min_depth = 0
    # max_depth = 4
    # depth = (frame_depth.cpu().numpy() - min_depth) / (max_depth - min_depth)
    # depth = np.clip(depth, a_min=0., a_max=1.)
    # depth_img = cv2.applyColorMap((depth * 255).astype(np.uint8),
    #                               cv2.COLORMAP_PLASMA)
    # depth_img = depth_img[..., ::-1]
    # Image.fromarray(depth_img).save(depth_vis_path)

    # depth_buffer = torch.ones((H, W)).cuda() * 1e6
    for idx, (v_filtered, f_filtered, face_mapping) in enumerate(mask_list):

        v_filtered = torch.tensor(v_filtered.astype(np.float32)).float()
        f_filtered = torch.tensor(f_filtered.astype(np.int32))

        v_filtered = F.pad(v_filtered, pad=(0, 1), mode='constant', value=1.0).cuda().contiguous()
        f_filtered = f_filtered.cuda().contiguous()

        filtered_mesh_dict = {}

        filtered_mesh_dict['vertices'] = v_filtered
        filtered_mesh_dict['pos_idx'] = f_filtered

        filtered_valid, filtered_triangle_id, filtered_depth = rasterize_texture(filtered_mesh_dict, projection, glctx, resolution, c2w)

        # depth_vis_path = os.path.join(mask_save_dir, f"{stem}_depth_mask_{idx}.png")
        # min_depth = 0
        # max_depth = 4
        # depth = (filtered_depth.cpu().numpy() - min_depth) / (max_depth - min_depth)
        # depth = np.clip(depth, a_min=0., a_max=1.)
        # depth_img = cv2.applyColorMap((depth * 255).astype(np.uint8),
        #                               cv2.COLORMAP_PLASMA)
        # depth_img = depth_img[..., ::-1]
        # Image.fromarray(depth_img).save(depth_vis_path)

        # filtered_triangle_id_mapped = torch.from_numpy(face_mapping).cuda()[filtered_triangle_id]

        # inters = np.intersect1d(
        #     filtered_triangle_id_mapped[filtered_valid].cpu().numpy(),
        #     frame_triangle_id[frame_valid].cpu().numpy()
        # )
        # if inters.shape[0] < 10:
        #     continue

        # frames_sam_input[stem]
        num_valid_pixel = torch.nonzero(filtered_valid).shape[0]
        filtered_valid = torch.logical_and(filtered_valid, torch.abs(filtered_depth - frame_depth) < 5e-3)
        num_valid_pixel_after = torch.nonzero(filtered_valid).shape[0]

        occluded = num_valid_pixel - num_valid_pixel_after > 500

        if not torch.any(filtered_valid):
            continue

        filtered_valid_numpy = filtered_valid.cpu().numpy()

        coords = np.stack(np.nonzero(filtered_valid_numpy), axis=1)  # [:, (1, 0)]

        if coords.shape[0] < 500:
            continue
        min_h, min_w = np.min(coords, 0)
        max_h, max_w = np.max(coords, 0)
        if max_w - min_w < 20 or max_h - min_h < 20:
            continue

        vis_frame[filtered_valid_numpy] = np.clip(vis_frame[filtered_valid_numpy] * 0.3 + mask_colors[idx] * 255 * 0.7, 0, 255).astype(np.uint8)

        negative_pts = sample_negative(mask_pixel_grid, (min_h, min_w, max_h, max_w))

        # mean_h, mean_w = np.mean(mask_pixel_grid[filtered_valid.cpu().numpy()].reshape(-1, 2), axis=0)
        # mean_h = int(mean_h)
        # mean_w = int(mean_w)

        delta = 0.2
        # possible_positive_pts = [(mean_h, mean_w)]
        positive_pts = []
        possible_positive_pts = [
            (int(min_h * 0.5 + max_h * 0.5), int(min_w * 0.5 + max_w * 0.5)),
            (int(min_h * (0.5 - delta) + max_h * (0.5 + delta)), int(min_w * (0.5 - delta) + max_w * (0.5 + delta))),
            (int(min_h * (0.5 - delta) + max_h * (0.5 + delta)), int(min_w * (0.5 + delta) + max_w * (0.5 - delta))),
            (int(min_h * (0.5 + delta) + max_h * (0.5 - delta)), int(min_w * (0.5 - delta) + max_w * (0.5 + delta))),
            (int(min_h * (0.5 + delta) + max_h * (0.5 - delta)), int(min_w * (0.5 + delta) + max_w * (0.5 - delta))),
        ]
        for pt in possible_positive_pts:
            if filtered_valid[pt[0], pt[1]]:
                positive_pts.append(pt)

        if len(positive_pts) == 0:
            continue
        n_pts = 3

        Image.fromarray(np.clip(filtered_valid.cpu().numpy().astype(np.float32) * 255, 0, 255).astype(np.uint8)).save(
            os.path.join(mask_save_dir, "mesh_mask", stem, f"{idx}.png"))

        sam_mask_input = reference_to_sam_mask(filtered_valid.cpu().numpy())

        frames_sam_input[stem].append({
            "negative_pts": negative_pts,
            "positive_pts": positive_pts,
            "idx": idx,
            "boundary": [int(min_h), int(min_w), int(max_h), int(max_w)],
            "sam_mask_input": sam_mask_input,
            "occluded": occluded,
        })

    vis_frames.append(vis_frame)
    Image.fromarray(vis_frame).save(os.path.join(mask_save_dir, "mesh_mask_vis", name))
    for pair_pts_i in range(len(frames_sam_input[stem])):

        min_h, min_w, max_h, max_w = frames_sam_input[stem][pair_pts_i]["boundary"]
        delta = 0.2
        b_min_h = max(int(min_h - delta * (max_h - min_h)), 0)
        b_max_h = min(int(max_h + delta * (max_h - min_h)), H - 1)
        b_min_w = max(int(min_w - delta * (max_w - min_w)), 0)
        b_max_w = min(int(max_w + delta * (max_w - min_w)), W - 1)

        for pair_pts_j in range(len(frames_sam_input[stem])):
            if pair_pts_i != pair_pts_j:
                possible_neg_pts = frames_sam_input[stem][pair_pts_j]["positive_pts"]
                for pt in possible_neg_pts:
                    if pt[0] >= b_min_h and pt[0] <= b_max_h and pt[1] >= b_min_w and pt[1] <= b_max_w:
                        frames_sam_input[stem][pair_pts_i]["negative_pts"].append(pt)

CONSOLE.print("mask video will be save at: ", f"{ckpt_dir}/mask.mp4")
imageio.mimsave(os.path.join(f"{ckpt_dir}/mask.mp4"),
                        vis_frames,
                        fps=24, macro_block_size=1)
CONSOLE.print("mask pkl will be save at: ", mask_json_path)
with open(mask_json_path, 'wb') as f:
    pickle.dump({
        "sam_input": frames_sam_input,
        "mask_cnt": len(mask_list),
        "mesh_mask": mesh_mask_save_dict,
    }, f)

# face_color = np.zeros((f_np.shape[0], 3))
# for mask in mask_list:
#     face_color[mask] = np.random.rand(1, 3)
#
# mesh = trimesh.Trimesh(v_np, f_np, face_colors=face_color)
# trimesh.exchange.export.export_mesh(mesh, mask_mesh_path)