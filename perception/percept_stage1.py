import os
import numpy as np
import pickle
from tqdm import tqdm
import nvdiffrast.torch as dr
import trimesh
import torch
import torch.nn.functional as F
import torchvision
import networkx as nx
from itertools import combinations
import pymeshlab as pml
import subprocess
from scipy.ndimage import binary_erosion
import argparse
from PIL import Image
import json

import random

import trimesh.exchange
import trimesh.exchange.export
random.seed(42)

import kaolin as kal

def rasterize_kaolin(verts, faces, w2c, proj, resolution, sigma_inv=1/(3*(10**(-5))), box_len=0.02, k_num=30):
    
    verts = verts.unsqueeze(0)
    faces = faces.long()
    
    verts_h = F.pad(verts, pad=(0, 1), mode='constant', value=1.0)
    
    verts_camera_h = torch.matmul(verts_h, w2c.t())
    coord_transform = torch.eye(4).to(verts_camera_h.device)
    coord_transform[0, 0] = -1
    coord_transform[2, 2] = -1
    verts_camera_h = verts_camera_h @ coord_transform
    # verts_camera_h[..., [0, 2]] *= -1
    
    verts_camera = verts_camera_h[:, :, :3] / verts_camera_h[:, :, 3:]
    
    verts_clip_h = torch.matmul(verts_camera_h, proj.t())
    verts_clip = verts_clip_h[:, :, :3] / verts_clip_h[:, :, 3:]
    
    face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(verts_camera, faces) # (1, F, 3(v), 3(xyz))
    face_vertices_clip = kal.ops.mesh.index_vertices_by_faces(verts_clip, faces) # (1, F, 3(v), 3(xyz))
    
    face_normals = kal.ops.mesh.face_normals(face_vertices_camera)
    
    face_vertices_z = face_vertices_camera[..., 2]
    
    visible_faces = torch.all(face_vertices_z <= 0, dim=-1)
    face_mapping = torch.arange(faces.shape[0]).to(visible_faces.device)[visible_faces.reshape(-1)]
    
    face_vertices_z = face_vertices_z[visible_faces].unsqueeze(0)
    face_vertices_clip = face_vertices_clip[visible_faces].unsqueeze(0)
    face_vertices_camera = face_vertices_camera[visible_faces].unsqueeze(0)
    face_normals = face_normals[visible_faces].unsqueeze(0)
    
    h, w = resolution
    
    rendered_features, soft_mask, face_idx_map = kal.render.mesh.dibr_rasterization(
        height=h,
        width=w,
        face_vertices_z=face_vertices_z,  # Depth values
        face_vertices_image=face_vertices_clip[..., :2],  # Image plane coordinates
        face_features=-face_vertices_camera[..., 2:3],  # Optional: per-vertex features like colors
        face_normals_z=face_normals[..., 2],  # Z-component of face normals
        sigmainv=sigma_inv,
        boxlen=box_len,
        knum=k_num
    )
    
    soft_mask = soft_mask.squeeze(0)
    face_idx_map = face_mapping[face_idx_map.reshape(-1)].reshape(h, w)
    depth_map = rendered_features.reshape(h, w)
    
    return soft_mask, face_idx_map, depth_map
    
def rasterize_texture_kaolin(verts, faces, w2c, proj, resolution):
    soft_mask, face_idx_map, depth_map = rasterize_kaolin(verts, faces, w2c, proj, resolution, sigma_inv=7*10**8)
    valid = face_idx_map >= 0
    triangle_id = face_idx_map.long()
    
    return valid, triangle_id

def rasterize_texture(vertices, faces, projection, glctx, resolution):

    vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), torch.transpose(projection, 0, 1)).float().unsqueeze(0)
    rast_out, _ = dr.rasterize(glctx, vertices_clip, faces, resolution=resolution)
    # rast_out = rast_out.flip([1])

    H, W = resolution
    valid = (rast_out[..., -1] > 0).reshape(H, W)
    triangle_id = (rast_out[..., -1] - 1).long().reshape(H, W)

    return valid, triangle_id

def filter_mesh_from_faces(keep, mesh_points, faces):

    keep_verts_idxs = faces[keep].reshape(-1)

    keep = np.zeros((mesh_points.shape[0])) > 0
    keep[keep_verts_idxs] = True

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

def filter_mesh_from_vertices(keep, mesh_points, faces, tex_pos):
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
    if tex_pos is not None:
        tex_pos = tex_pos[keep_faces]
    face_mapping = np.arange(keep_faces.shape[0])[keep_faces]
    faces[:, 0] = filter_unmapping[faces[:, 0]]
    faces[:, 1] = filter_unmapping[faces[:, 1]]
    faces[:, 2] = filter_unmapping[faces[:, 2]]
    return mesh_points, faces, face_mapping, tex_pos

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--sdf_dir', type=str, required=True)
    parser.add_argument('--num_max_frames', type=int, default=1000)
    parser.add_argument('--num_faces_simplified', type=int, default=80000)

    args = parser.parse_args()
    mask_dir = os.path.join(args.data_dir, 'grounded_sam')
    pose_path = os.path.join(args.data_dir, 'pose.pkl')
    save_dir = os.path.join(args.data_dir, 'perception', 'vis_groups')
    image_dir = args.image_dir
    ckpt_dir = args.sdf_dir
    
    mask_data_dir = os.path.join(mask_dir, "mask_data")
    mask_save_path = os.path.join(mask_dir, "separate_vis_gsam.pkl")
    
    glctx = dr.RasterizeGLContext(output_db=False)
    
    mesh_path = os.path.join(ckpt_dir, "mesh-simplify.ply")

    os.makedirs(save_dir, exist_ok=True)

    H, W = np.array(Image.open(os.path.join(image_dir, os.listdir(image_dir)[0]))).shape[:2]
    resolution_raw = (H, W)

    while H % 8 != 0 or W % 8 != 0:
        H = H * 2
        W = W * 2

    resolution = (H, W)

    mesh = trimesh.exchange.load.load_mesh(mesh_path)
    
    mesh_verts_np = mesh.vertices
    mesh_faces_np = mesh.faces
    
    m = pml.Mesh(mesh_verts_np, mesh_faces_np)
    ms = pml.MeshSet()
    ms.add_mesh(m, 'mesh') # will copy!
    target = args.num_faces_simplified
    optimalplacement  = False
    print("Simplifying mesh...")
    ms.simplification_quadric_edge_collapse_decimation(targetfacenum=int(target), optimalplacement=optimalplacement)
    m = ms.current_mesh()
    mesh_verts_np = m.vertex_matrix()
    mesh_faces_np = m.face_matrix()
    
    mesh_verts = torch.from_numpy(mesh_verts_np).float().cuda().contiguous()
    mesh_faces = torch.from_numpy(mesh_faces_np).int().cuda().contiguous()
    

    #### create all_mask_dict
    print("create all_mask_dict...")
    stem_list = [fname[:-4] for fname in sorted(os.listdir(mask_data_dir))]
    all_mask_dict = {}
    for stem in tqdm(stem_list):

        image_path = os.path.join(image_dir, stem+".png")
        H, W = resolution_raw

        with open(os.path.join(mask_dir, "mask", stem+".json"), 'r') as f:
            mask_src_json = json.load(f)
        with open(os.path.join(mask_dir, "mask_data", stem+".pkl"), 'rb') as f:
            mask_src_data = pickle.load(f)["mask"]


        all_mask_dict[stem] = {}

        for mask_dict in mask_src_json:
            value = mask_dict["value"]
            if value == 0:
                continue
            idx = value - 1

            binary_mask = mask_src_data[idx].reshape(H, W).cpu().numpy() > 0

            all_mask_dict[stem][value] = binary_mask
    
    with open(mask_save_path, 'wb') as f:
        pickle.dump(all_mask_dict, f)
    
    with open(pose_path, 'rb') as f:
        pose_dict = pickle.load(f)
        
    stem_list = sorted(list(all_mask_dict.keys()))
    
    stem_list = [stem for stem in stem_list if stem in pose_dict]
    
    if len(stem_list) > args.num_max_frames:
        stem_list = [elem for sidx, elem in enumerate(stem_list) if sidx in np.linspace(0, len(stem_list)-1, args.num_max_frames).astype(np.int32)]
    
    
    # build graph for mesh
    G = nx.Graph()
    # Dictionary to track edges to be added or updated
    edge_updates = {}
    # edge_updates_nvis = {}
    pairs_list = []
    for stem in tqdm(stem_list):
        mask_idxs = sorted(list(all_mask_dict[stem].keys()))
        
        pose = pose_dict[stem]
        c2w = pose["c2w"]
        mvp = pose["mvp"]
        camproj = pose["camproj"]

        mvp = torch.from_numpy(mvp).cuda().float()
        camproj = torch.from_numpy(camproj).cuda().float()
        

        valid, triangle_id = rasterize_texture(mesh_verts, mesh_faces, mvp, glctx, resolution)
        
        H, W = resolution[0], resolution[1]
        
        
        for mask_idx in mask_idxs:
            binary_mask = all_mask_dict[stem][mask_idx]

            binary_mask = binary_erosion(binary_mask, iterations=4)

            binary_mask = torchvision.transforms.functional.resize(torch.from_numpy(binary_mask).unsqueeze(0) > 0, (H, W), torchvision.transforms.InterpolationMode.NEAREST).cpu().numpy()

            faces_idxs = triangle_id[binary_mask][valid[binary_mask]].cpu().numpy().astype(np.int32)

            assert np.all(faces_idxs >= 0)
            
            # Sort vertices to ensure deterministic ordering
            faces_idxs = np.unique(faces_idxs)
            faces_idxs = np.sort(faces_idxs)

            # Iterate through each unique pair using combinations
            pairs = list(combinations(faces_idxs, 2))
            pairs_list.append(pairs)

            for u, v in pairs:
                # Use a tuple to represent the edge
                edge = (u, v)
                if edge in edge_updates:
                    # Increment weight if edge exists in the dictionary
                    edge_updates[edge] += 1
                else:
                    # Initialize weight if edge is new
                    edge_updates[edge] = 1
                    
    edge_filtered_updates = {}

    for pairs in tqdm(pairs_list):
        pairs_sample = []
        for u, v in pairs:
            edge = (u, v)
            if edge in edge_updates  and edge_updates[edge] >= 5:
                pairs_sample.append((u, v))
                
        Max_P = 1000
        if len(pairs_sample) > Max_P:
            pairs_sample = random.sample(pairs_sample, Max_P)

        for u, v in pairs_sample:
            # Use a tuple to represent the edge
            edge = (u, v)
            if edge in edge_filtered_updates:
                # Increment weight if edge exists in the dictionary
                edge_filtered_updates[edge] += 1
            else:
                # Initialize weight if edge is new
                edge_filtered_updates[edge] = 1
                
    cnt = 0
    for edge, weight in tqdm(edge_filtered_updates.items()):
        nvis = 1.0
        G.add_edge(edge[0], edge[1], weight=weight / nvis)
        cnt += 1
        
    
    potential_meshes = []
    faces_idxs_list = []
    potential_i = 0
    subprocess.run(f"rm {save_dir}/*", shell=True)
    for res in [1.0, 2.0, 4.0, 5.0, 10.0, 20.0, 35.0, 50.0, 75.0, 100.0, 200.0, 350.0, 500.0]:
        partition = nx.community.louvain_communities(G, seed=42, resolution=res)
        parts = []
        for part_idx, faces_idxs in enumerate(partition):
            faces_idxs = np.array(list(faces_idxs)).reshape(-1).astype(np.int32)
            if faces_idxs.shape[0] > 10:
                parts.append(faces_idxs)
        print("res: ", res, "total partition: ", len(parts))

        for part_idx, faces_idxs in enumerate(parts):

            mesh_mask_verts_np, mesh_mask_faces_np, face_map_0 = filter_mesh_from_faces(faces_idxs, mesh_verts_np, mesh_faces_np)

            faces_idxs = face_map_0
            mask_mesh = trimesh.Trimesh(
                mesh_mask_verts_np,
                mesh_mask_faces_np,
                process=False
            )

            edges = mask_mesh.edges_sorted.reshape((-1, 2))
            components = trimesh.graph.connected_components(edges, min_len=1, engine='scipy')
            largest_cc = np.argmax(np.array([comp.shape[0] for comp in components]).reshape(-1), axis=0)
            keep = np.zeros((mesh_mask_verts_np.shape[0])).astype(np.bool_)
            keep[components[largest_cc].reshape(-1)] = True
            _, _, face_map_1, _ = filter_mesh_from_vertices(keep, mesh_mask_verts_np, mesh_mask_faces_np, None)


            faces_idxs = faces_idxs[face_map_1]

            potential_meshes.append(mask_mesh)
            trimesh.exchange.export.export_mesh(
                mask_mesh,
                os.path.join(save_dir, f"mask_mesh_{potential_i:0>3d}.ply")
            )
            faces_idxs_list.append(faces_idxs)
            potential_i += 1
            
    with open(os.path.join(save_dir, f"faces_idxs_list.pkl"), 'wb') as f:
        pickle.dump(faces_idxs_list, f)