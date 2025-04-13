import os

import torch
from pytorch3d.io import save_obj, load_objs_as_meshes
import pickle
import transformations
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sdf_dir", type=str, required=True)
parser.add_argument("--interior_dir", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)

args = parser.parse_args()
bakedsdf_dir = args.sdf_dir
textured_drawer_dir = args.interior_dir
save_dir = args.save_dir
os.makedirs(save_dir, exist_ok=True)

drawer_dir = os.path.join(bakedsdf_dir, "drawers", "results")
total_drawers_num = len([name for name in os.listdir(drawer_dir) if os.path.splitext(name)[1] == '.pkl'])

for drawer_i in tqdm(range(total_drawers_num)):

    # read the transform
    prim_i = drawer_i

    prim_mark = f"{drawer_i:0>2d}"

    drawer_path = os.path.join(drawer_dir, f"drawer_{drawer_i}.pkl")
    with open(drawer_path, 'rb') as f:
        prim_info = pickle.load(f)
        prim_transform = prim_info["transform"]
        interact_type = prim_info["interact"]

    # get transform
    scale, _, angles, trans, _ = transformations.decompose_matrix(prim_transform)
    prim_rotation = transformations.euler_matrix(axes='sxyz', *angles).reshape(4, 4)
    prim_translation = np.eye(4)
    prim_translation[:3, 3] = trans
    prim_rot_trans_original = prim_translation @ prim_rotation

    ## drawer_prim_internal is a trimesh.Trimesh
    ### internal_mesh is the textured drawer interior that hasn't transformed
    mesh_path = os.path.join(textured_drawer_dir, prim_mark, f"mesh_{prim_mark}.obj")
    door_path = os.path.join(textured_drawer_dir, prim_mark, f"mesh_door_{prim_mark}.obj")

    mesh = load_objs_as_meshes([mesh_path])
    door = load_objs_as_meshes([door_path])

    prim_rot_trans_original_inversed = np.linalg.inv(prim_rot_trans_original)

    drawer_prim_internal_verts_pad = np.pad(mesh.verts_packed().cpu().numpy(), ((0, 0), (0, 1)),
                                            constant_values=(0, 1))
    drawer_prim_internal_verts_transformed_pad = drawer_prim_internal_verts_pad @ prim_rot_trans_original_inversed.T
    drawer_prim_internal_verts_transformed = drawer_prim_internal_verts_transformed_pad[:,
                                             :3] / drawer_prim_internal_verts_transformed_pad[:, 3:]
    mesh = mesh.update_padded(torch.from_numpy(drawer_prim_internal_verts_transformed).unsqueeze(0))

    drawer_prim_internal_verts_pad = np.pad(door.verts_packed().cpu().numpy(), ((0, 0), (0, 1)),
                                            constant_values=(0, 1))
    drawer_prim_internal_verts_transformed_pad = drawer_prim_internal_verts_pad @ prim_rot_trans_original_inversed.T
    drawer_prim_internal_verts_transformed = drawer_prim_internal_verts_transformed_pad[:,
                                             :3] / drawer_prim_internal_verts_transformed_pad[:, 3:]
    door = door.update_padded(torch.from_numpy(drawer_prim_internal_verts_transformed).unsqueeze(0))

    os.makedirs(os.path.join(save_dir, f"{prim_i:0>2d}"), exist_ok=True)

    save_obj(os.path.join(save_dir, f"{prim_i:0>2d}", f"mesh_{prim_i:0>2d}.obj"),
             verts=mesh.verts_packed(),
             faces=mesh.faces_packed(),
             verts_uvs=mesh.textures.verts_uvs_padded().squeeze(0),
             faces_uvs=mesh.textures.faces_uvs_padded().squeeze(0),
             texture_map=mesh.textures.maps_padded().squeeze(0)
             )

    save_obj(os.path.join(save_dir, f"{prim_i:0>2d}", f"mesh_door_{prim_i:0>2d}.obj"),
             verts=door.verts_packed(),
             faces=door.faces_packed(),
             verts_uvs=door.textures.verts_uvs_padded().squeeze(0),
             faces_uvs=door.textures.faces_uvs_padded().squeeze(0),
             texture_map=door.textures.maps_padded().squeeze(0)
             )


