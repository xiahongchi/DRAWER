import os
from pytorch3d.structures import Meshes
import trimesh
from sklearn.cluster import DBSCAN
import subprocess
import torch
import numpy as np
import pickle
from tqdm import tqdm
import torch.nn.functional as F
import torchvision
import nvdiffrast.torch as dr
from PIL import Image
import pymeshlab as pml
import json
import argparse

def rasterize_texture(vertices, faces, projection, glctx, resolution):

    vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), torch.transpose(projection, 0, 1)).float().unsqueeze(0)
    rast_out, _ = dr.rasterize(glctx, vertices_clip, faces, resolution=resolution)
    # rast_out = rast_out.flip([1])

    H, W = resolution
    valid = (rast_out[..., -1] > 0).reshape(H, W)
    triangle_id = (rast_out[..., -1] - 1).long().reshape(H, W)

    return valid, triangle_id

def iou_list(a, b):
    iou = np.unique(np.intersect1d(a, b)).shape[0] / np.unique(np.union1d(a, b)).shape[0]
    return iou

def strict_cluster_by_similarity(similarity_matrix, alpha):
    """
    Cluster data based on a strict similarity condition where all pairs in a cluster
    have similarity greater than a threshold alpha.

    Parameters:
    - similarity_matrix (np.ndarray): N x N similarity matrix with values between 0 and 1.
    - alpha (float): Threshold for similarity (between 0 and 1).

    Returns:
    - clusters (list of lists): List of clusters, each containing indices of data points in that cluster.
    """
    N = similarity_matrix.shape[0]
    unclustered = set(range(N))
    clusters = []

    while unclustered:
        # Start a new cluster with an arbitrary unclustered point
        cluster = [unclustered.pop()]
        added = True

        # Try to expand the cluster with any point meeting the similarity threshold to all current members
        while added:
            added = False
            for point in list(unclustered):
                if all(similarity_matrix[point, member] > alpha for member in cluster):
                    cluster.append(point)
                    unclustered.remove(point)
                    added = True
        
        clusters.append(cluster)

    return clusters

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

def check_normal_consistency(face_normals):
    mean_normal = face_normals.mean(axis=0)
    mean_normal /= np.linalg.norm(mean_normal)  # Normalize the mean normal

    # Step 3: Calculate angular deviation for each face normal from the mean normal
    angular_deviation = []
    for normal in face_normals:
        # Dot product between mean normal and face normal gives the cosine of the angle
        normal /= np.linalg.norm(normal)  # Normalize the mean normal
        cos_theta = np.dot(mean_normal, normal)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Angle in radians
        angle_degrees = np.abs(np.degrees(angle))  # Convert to degrees for easier interpretation
        angular_deviation.append(angle_degrees)

    # Step 4: Check if the normals are consistent (e.g., within 10 degrees of mean)
    tolerance = 40  # degrees
    consistent_normals = np.array(angular_deviation) < tolerance

    return (np.count_nonzero(consistent_normals) / len(consistent_normals)) > 0.6

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--sdf_dir", type=str, required=True)
    
    args = parser.parse_args()
    
    ckpt_dir = args.sdf_dir

    back_match_vis_dir = os.path.join(args.data_dir, 'perception', "vis_groups_back_match")
    binary_mask_dir = os.path.join(args.data_dir, 'perception', "vis_groups_handle_note_masks")
    gpt_filtered_dir = os.path.join(args.data_dir, 'perception', "vis_groups_gpt4_api")
    save_dir = os.path.join(args.data_dir, 'perception', "vis_groups_final_mesh")
    pose_path = os.path.join(args.data_dir, "pose.pkl")


    os.makedirs(save_dir, exist_ok=True)
    subprocess.run(f"rm {save_dir}/*", shell=True)


    with open(pose_path, 'rb') as f:
        pose_dict = pickle.load(f)

    resolution = [540*2, 960*2]
    H, W = resolution[0], resolution[1]
    mesh_path = os.path.join(ckpt_dir, "mesh-simplify.ply")

    mesh = trimesh.exchange.load.load_mesh(mesh_path)

    mesh_verts_np = mesh.vertices
    mesh_faces_np = mesh.faces

    mesh_verts = torch.from_numpy(mesh_verts_np).float().cuda().contiguous()
    mesh_faces = torch.from_numpy(mesh_faces_np).int().cuda().contiguous()

    filtered_fnames = sorted(os.listdir(gpt_filtered_dir))
    filtered_fnames = [name for name in filtered_fnames if name.endswith('.png')]
    faces_idxs_list = []
    glctx = dr.RasterizeGLContext(output_db=False)

    for fname in tqdm(filtered_fnames):
        # back_match_mesh_001_frame_00366_01.png

        binary_mask_path = os.path.join(back_match_vis_dir, fname.replace(".png", "_mask.png"))
        binary_mask = np.array(Image.open(binary_mask_path)).astype(np.float32) / 255. > 0

        stem = fname[len("back_match_mesh_001_"):-len("_01.png")]

        pose = pose_dict[stem]
        c2w = pose["c2w"]
        mvp = pose["mvp"]

        mvp = torch.from_numpy(mvp).cuda().float()
        valid, triangle_id = rasterize_texture(mesh_verts, mesh_faces, mvp, glctx, resolution)

        binary_mask = torchvision.transforms.functional.resize(torch.from_numpy(binary_mask).unsqueeze(0) > 0, (H, W), torchvision.transforms.InterpolationMode.NEAREST).cpu().numpy()
        faces_idxs = np.unique(triangle_id[binary_mask][valid[binary_mask]].cpu().numpy().astype(np.int32)).reshape(-1)

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

        faces_idxs_list.append(faces_idxs)


    iou_matrix = np.eye(len(faces_idxs_list))
    for i in range(len(faces_idxs_list)):
        for j in range(len(faces_idxs_list)):
            iou_matrix[i, j] = iou_list(faces_idxs_list[i], faces_idxs_list[j])

    clusters = strict_cluster_by_similarity(iou_matrix, 0.2)

    select_idxs = []
    for cluster in clusters:
        print(cluster)
        # min_f_cnt = 1e10
        # min_c_i = None
        
        max_sqratio = 0.
        max_c_i = None
        
        for c_i in cluster:
            binary_mask_path = os.path.join(binary_mask_dir, filtered_fnames[c_i])
            binary_mask = np.array(Image.open(binary_mask_path)).astype(np.float32) / 255. > 0
            
            num_mask_pixel = np.count_nonzero(binary_mask)
            
            w_mask = np.any(binary_mask, axis=0).reshape(-1) # (w)
            h_mask = np.any(binary_mask, axis=1).reshape(-1) # (h)
            w_min = np.min(np.nonzero(w_mask))
            w_max = np.max(np.nonzero(w_mask))
            h_min = np.min(np.nonzero(h_mask))
            h_max = np.max(np.nonzero(h_mask))
            mask_w = w_max - w_min
            mask_h = h_max - h_min
            
            sqratio = num_mask_pixel / (mask_w * mask_h)
            
            if sqratio > max_sqratio:
                max_sqratio = sqratio
                max_c_i = c_i
                        
            
            # if min_f_cnt > len(faces_idxs_list[c_i]):
            #     min_f_cnt = len(faces_idxs_list[c_i])
            #     min_c_i = c_i

        select_idxs.append(max_c_i)

    faces_idxs_list = [faces_idxs for idx, faces_idxs in enumerate(faces_idxs_list) if idx in select_idxs]
    filtered_fnames = [fname for idx, fname in enumerate(filtered_fnames) if idx in select_idxs]
    print(len(faces_idxs_list))
    print("filtered_fnames:", filtered_fnames)

    select_idxs = []
    mask_mesh_list = []
    for mask_idx, faces_idxs in enumerate(faces_idxs_list):

        mesh_mask_verts_np, mesh_mask_faces_np, _ = filter_mesh_from_faces(faces_idxs, mesh_verts_np, mesh_faces_np)

        mesh_mask_p3d = Meshes(
            [torch.from_numpy(mesh_mask_verts_np)],
            [torch.from_numpy(mesh_mask_faces_np)],
        )
        if check_normal_consistency(mesh_mask_p3d.faces_normals_packed().cpu().numpy()):

            mask_mesh = trimesh.Trimesh(
                mesh_mask_verts_np,
                mesh_mask_faces_np,
                vertex_colors=np.random.rand(3).reshape(1, 3).repeat(mesh_mask_verts_np.shape[0], axis=0),
                process=False
            )
        
            select_idxs.append(mask_idx)
            mask_mesh_list.append(mask_mesh)

    print("after normal filtering")
    faces_idxs_list = [faces_idxs for idx, faces_idxs in enumerate(faces_idxs_list) if idx in select_idxs]
    filtered_fnames = [fname for idx, fname in enumerate(filtered_fnames) if idx in select_idxs]
    print(len(faces_idxs_list))
    print("filtered_fnames:", filtered_fnames)

    for mask_idx, mask_mesh in enumerate(mask_mesh_list):

        trimesh.exchange.export.export_mesh(
            mask_mesh,
            os.path.join(save_dir, f"mask_mesh_{mask_idx}.ply")
        )

    mask_dict = {}
    for fname in tqdm(filtered_fnames):
        Image.open(os.path.join(gpt_filtered_dir, fname)).save(os.path.join(save_dir, fname))

        stem = fname[len("back_match_mesh_001_"):-len("_01.png")]
        mask_idx = int(fname[len("back_match_mesh_001_frame_00000_"):-len(".png")])
        mask_stem = f"mask/{stem}.json"
        if mask_stem not in mask_dict:
            mask_dict[mask_stem] = [mask_idx]
        else:
            mask_dict[mask_stem].append(mask_idx)

    with open(os.path.join(save_dir, f"all.json"), 'w') as f:
        json.dump(mask_dict, f, indent=4)