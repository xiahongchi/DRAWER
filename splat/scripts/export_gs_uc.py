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
from collections import OrderedDict
import typing
import trimesh
import sys
sys.path.append("./scripts")

def write_ply(
    filename: str,
    count: int,
    map_to_tensors: typing.OrderedDict[str, np.ndarray],
):
    """
    Writes a PLY file with given vertex properties and a tensor of float or uint8 values in the order specified by the OrderedDict.
    Note: All float values will be converted to float32 for writing.

    Parameters:
    filename (str): The name of the file to write.
    count (int): The number of vertices to write.
    map_to_tensors (OrderedDict[str, np.ndarray]): An ordered dictionary mapping property names to numpy arrays of float or uint8 values.
        Each array should be 1-dimensional and of equal length matching 'count'. Arrays should not be empty.
    """

    # Ensure count matches the length of all tensors
    if not all(len(tensor) == count for tensor in map_to_tensors.values()):
        raise ValueError("Count does not match the length of all tensors")

    # Type check for numpy arrays of type float or uint8 and non-empty
    if not all(
        isinstance(tensor, np.ndarray)
        and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
        and tensor.size > 0
        for tensor in map_to_tensors.values()
    ):
        raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")

    with open(filename, "wb") as ply_file:
        # Write PLY header
        ply_file.write(b"ply\n")
        ply_file.write(b"format binary_little_endian 1.0\n")

        ply_file.write(f"element vertex {count}\n".encode())

        # Write properties, in order due to OrderedDict
        for key, tensor in map_to_tensors.items():
            data_type = "float" if tensor.dtype.kind == "f" else "uchar"
            ply_file.write(f"property {data_type} {key}\n".encode())

        ply_file.write(b"end_header\n")

        # Write binary data
        # Note: If this is a performance bottleneck consider using numpy.hstack for efficiency improvement
        for i in range(count):
            for tensor in map_to_tensors.values():
                value = tensor[i]
                if tensor.dtype.kind == "f":
                    ply_file.write(np.float32(value).tobytes())
                elif tensor.dtype == np.uint8:
                    ply_file.write(value.tobytes())

def compose_for_export(means, shs_0, shs_rest, colors, opacities, scales, quats, prim_means_indices_prim_i):
    means_drawer_i = means[prim_means_indices_prim_i].cpu().numpy()
    shs_0_drawer_i = shs_0[prim_means_indices_prim_i].contiguous().cpu().numpy()
    shs_rest_drawer_i = shs_rest[prim_means_indices_prim_i].transpose(1, 2).cpu().numpy()
    colors_drawer_i = torch.clamp(colors[prim_means_indices_prim_i], 0.0, 1.0).data.cpu().numpy()
    opacities_drawer_i = opacities[prim_means_indices_prim_i].cpu().numpy()
    scales_drawer_i = scales[prim_means_indices_prim_i].cpu().numpy()
    quats_drawer_i = quats[prim_means_indices_prim_i].cpu().numpy()

    map_to_tensors = OrderedDict()

    with torch.no_grad():

        positions = means_drawer_i
        count = positions.shape[0]
        n = count
        map_to_tensors["x"] = positions[:, 0]
        map_to_tensors["y"] = positions[:, 1]
        map_to_tensors["z"] = positions[:, 2]
        map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

        if model.config.sh_degree > 0:
            shs_0 = shs_0_drawer_i
            for i in range(shs_0.shape[1]):
                map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

            # transpose(1, 2) was needed to match the sh order in Inria version
            shs_rest = shs_rest_drawer_i
            shs_rest = shs_rest.reshape((n, -1))
            for i in range(shs_rest.shape[-1]):
                map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
        else:
            colors = colors_drawer_i
            map_to_tensors["colors"] = (colors * 255).astype(np.uint8)

        map_to_tensors["opacity"] = opacities_drawer_i

        scales = scales_drawer_i
        for i in range(3):
            map_to_tensors[f"scale_{i}"] = scales[:, i, None]

        quats = quats_drawer_i
        for i in range(4):
            map_to_tensors[f"rot_{i}"] = quats[:, i, None]

    select = np.ones(n, dtype=bool)
    for k, t in map_to_tensors.items():
        n_before = np.sum(select)
        select = np.logical_and(select, np.isfinite(t).all(axis=-1))
        n_after = np.sum(select)
        if n_after < n_before:
            print(f"{n_before - n_after} NaN/Inf elements in {k}")

    if np.sum(select) < n:
        print(f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}")
        for k, t in map_to_tensors.items():
            map_to_tensors[k] = map_to_tensors[k][select]
        count = np.sum(select)

    return map_to_tensors, count



load_config = "outputs/meshgauss/241005_cs_kitchen_splatfacto_on_mesh_bakedsdf_sdfstudio_normal_mono_depth_mono/splatfacto_on_mesh_uc/2024-10-19_180446/config.yml"
save_note_gs_internal_uc = "same_opt_50percent"

config, pipeline, checkpoint_path, _ = eval_setup(Path(load_config))
model = pipeline.model
model = model.eval()

load_dir = os.path.join(os.path.dirname(load_config), save_note_gs_internal_uc)
save_dir = os.path.join(load_dir, "decompose")
os.makedirs(save_dir, exist_ok=True)

save_dir_doors = os.path.join(load_dir, "doors")
save_dir_boxes = os.path.join(load_dir, "boxes")
save_dir_remaining = os.path.join(load_dir, "remaining")
remaining_mesh = trimesh.exchange.load.load_mesh(os.path.join(save_dir_remaining, "remaining.ply"))

collision_dir = os.path.join(load_dir, "collisions")
os.makedirs(collision_dir, exist_ok=True)

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
total_frames = len(trace_data["00"])
total_drawers_num = len(interact_type_list)

prim_means_indices = [torch.from_numpy(elem).cuda() for elem in prim_means_indices]
append_indices_prims = [torch.from_numpy(elem).cuda() for elem in append_indices_prims]
append_indices_prims_door = [torch.from_numpy(elem).cuda() for elem in append_indices_prims_door]

means = model.means.detach().clone()
shs_0 = model.shs_0.detach().clone()
shs_rest = model.shs_rest.detach().clone()
colors = model.colors.detach().clone()
opacities = model.opacities.detach().clone()
scales = model.scales.detach().clone()
quats = model.quats.detach().clone()

N_Gaussians = means.shape[0]
print("N_Gaussians: ", N_Gaussians)
remaining_indices = torch.ones(N_Gaussians, device="cuda") > 0

for drawer_i in tqdm(range(total_drawers_num)):
    interact_type = interact_type_list[drawer_i]
    prim_means_indices_prim_i = torch.zeros((N_Gaussians), dtype=torch.bool, device="cuda")
    prim_means_indices_prim_i[prim_means_indices[drawer_i]] = True

    prim_means_indices_prim_i[
        append_indices_prims_door[drawer_i].to(prim_means_indices_prim_i.device)] = True

    remaining_indices[prim_means_indices_prim_i] = False

    if interact_type in ["2", "3.3"]:
        prim_means_indices_prim_i[
            append_indices_prims[drawer_i].to(prim_means_indices_prim_i.device)] = True

    map_to_tensors, count = compose_for_export(means, shs_0, shs_rest, colors, opacities, scales, quats, prim_means_indices_prim_i)
    export_path = os.path.join(save_dir, f"splat_{drawer_i:0>2d}.ply")
    print("export to ", export_path)
    write_ply(export_path, count, map_to_tensors)

    collision_mesh_i_path = os.path.join(save_dir_doors, f"door_{drawer_i:0>2d}.ply")
    collision_mesh_i = trimesh.exchange.load.load_mesh(collision_mesh_i_path)

    collision_mesh_i_box_path = os.path.join(save_dir_boxes, f"box_{drawer_i:0>2d}.ply")
    collision_mesh_i_box = trimesh.exchange.load.load_mesh(collision_mesh_i_box_path)

    if interact_type in ["2", "3.3"]:
        collision_mesh_i = trimesh.util.concatenate([collision_mesh_i, collision_mesh_i_box])
    else:
        remaining_mesh = trimesh.util.concatenate([remaining_mesh, collision_mesh_i_box])

    trimesh.exchange.export.export_mesh(collision_mesh_i, os.path.join(collision_dir, f"drawer_{drawer_i:0>2d}.ply"))


if True:
    main_idx = 0
    while torch.count_nonzero(remaining_indices) > 1999999:
        print("0 remaining_indices: ", remaining_indices.shape, torch.count_nonzero(remaining_indices))
        select_indices = torch.argsort(means[remaining_indices, -1])[:2000000]
        current_delete_indices = torch.arange(remaining_indices.shape[0], device="cuda")[remaining_indices][select_indices]
        remaining_indices[current_delete_indices] = False
        print("1 remaining_indices: ", remaining_indices.shape, torch.count_nonzero(remaining_indices))
        current_delete = torch.zeros(remaining_indices.shape[0], device="cuda") > 0
        current_delete[current_delete_indices] = True
        print("current_delete: ", current_delete.shape, torch.count_nonzero(current_delete))

        map_to_tensors, count = compose_for_export(means, shs_0, shs_rest, colors, opacities, scales, quats,
                                                   current_delete)
        export_path = os.path.join(save_dir, f"splat_main_{main_idx:0>2d}.ply")
        print("export to ", export_path)
        write_ply(export_path, count, map_to_tensors)
        main_idx += 1

    map_to_tensors, count = compose_for_export(means, shs_0, shs_rest, colors, opacities, scales, quats,
                                               remaining_indices)
    export_path = os.path.join(save_dir, f"splat_main_{main_idx:0>2d}.ply")
    print("export to ", export_path)
    write_ply(export_path, count, map_to_tensors)
    trimesh.exchange.export.export_mesh(remaining_mesh, os.path.join(collision_dir, f"main.ply"))
else:
    map_to_tensors, count = compose_for_export(means, shs_0, shs_rest, colors, opacities, scales, quats, remaining_indices)
    export_path = os.path.join(save_dir, f"splat_main.ply")
    print("export to ", export_path)
    write_ply(export_path, count, map_to_tensors)
    trimesh.exchange.export.export_mesh(remaining_mesh, os.path.join(collision_dir, f"main.ply"))
