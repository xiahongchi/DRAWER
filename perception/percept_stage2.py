import os
from pytorch3d.structures import Meshes
import trimesh
from sklearn.cluster import DBSCAN
import subprocess
import torch
import numpy as np
import pickle
import argparse

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
    return mesh_points, faces, tex_pos


def check_normal_consistency(face_normals):
    mean_normal = face_normals.mean(axis=0)
    mean_normal /= np.linalg.norm(mean_normal)  # Normalize the mean normal

    # Step 3: Calculate angular deviation for each face normal from the mean normal
    angular_deviation = []
    for normal in face_normals:
        # Dot product between mean normal and face normal gives the cosine of the angle
        cos_theta = np.dot(mean_normal, normal)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Angle in radians
        angle_degrees = np.degrees(angle)  # Convert to degrees for easier interpretation
        angular_deviation.append(angle_degrees)

    # Step 4: Check if the normals are consistent (e.g., within 10 degrees of mean)
    tolerance = 40  # degrees
    consistent_normals = np.array(angular_deviation) < tolerance

    return np.sum(consistent_normals) / len(consistent_normals) > 0.8

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--min_faces', type=int, default=50)
    
    args = parser.parse_args()
    data_dir = args.data_dir
    
    load_mask_mesh_dir = os.path.join(data_dir, 'perception', 'vis_groups')
    save_dir = os.path.join(data_dir, 'perception', 'vis_groups_normal_filtered')
    
    mask_mesh_fnames = sorted(os.listdir(load_mask_mesh_dir))
    mask_mesh_fnames = [elem for elem in mask_mesh_fnames if elem.endswith(".ply")]
    faces_idxs_list_path = os.path.join(load_mask_mesh_dir, f"faces_idxs_list.pkl")
    os.makedirs(save_dir, exist_ok=True)
    subprocess.run(f"rm {save_dir}/*", shell=True)

    print("mask_mesh_fnames: ", len(mask_mesh_fnames))

    # with open(faces_idxs_list_path, 'rb') as f:
    #     faces_idxs_list = pickle.load(f)

    # iou_matrix = np.eye(len(faces_idxs_list))
    # for i in range(len(faces_idxs_list)):
    #     for j in range(len(faces_idxs_list)):
    #         iou_matrix[i, j] = iou_list(faces_idxs_list[i], faces_idxs_list[j])

    # clusters = strict_cluster_by_similarity(iou_matrix, 0.8)
    # print("clusters: ", len(clusters))

    # select_idxs = []
    # for cluster in clusters:
    #     select_idxs.append(cluster[0])
    # print("select_idxs: ", len(select_idxs))
    
    select_idxs = range(len(mask_mesh_fnames))
    
    mask_mesh_fnames = [fname for idx, fname in enumerate(mask_mesh_fnames) if idx in select_idxs]
    example_i = 0
    for fname in mask_mesh_fnames:
        mask_mesh_fpath = os.path.join(load_mask_mesh_dir, fname)
        mask_mesh = trimesh.exchange.load.load_mesh(mask_mesh_fpath)

        mask_mesh_verts = mask_mesh.vertices
        mask_mesh_faces = mask_mesh.faces

        if mask_mesh_faces.shape[0] < args.min_faces:
            continue

        # Create a graph from the mesh faces
        edges = mask_mesh.edges_sorted.reshape((-1, 2))
        components = trimesh.graph.connected_components(edges, min_len=1, engine='scipy')
        largest_cc = np.argmax(np.array([comp.shape[0] for comp in components]).reshape(-1), axis=0)
        keep = np.zeros((mask_mesh_verts.shape[0])).astype(np.bool_)
        keep[components[largest_cc].reshape(-1)] = True
        mask_mesh_verts, mask_mesh_faces, _ = filter_mesh_from_vertices(keep, mask_mesh_verts, mask_mesh_faces, None)

        # mask_mesh_p3d = Meshes(
        #     torch.from_numpy(mask_mesh_verts).float().unsqueeze(0),
        #     torch.from_numpy(mask_mesh_faces).int().unsqueeze(0),
        # )

        # mask_mesh_normals = mask_mesh_p3d.faces_normals_packed().float().cpu().numpy().reshape(-1, 3)

        # if not check_normal_consistency(mask_mesh_normals):
        #     continue

        
        mask_mesh = trimesh.Trimesh(
            mask_mesh_verts,
            mask_mesh_faces,
            vertex_colors=np.random.rand(3).reshape(1, 3).repeat(mask_mesh_verts.shape[0], axis=0),
            process=False
        )

        trimesh.exchange.export.export_mesh(
            mask_mesh,
            os.path.join(save_dir, f"mask_mesh_{example_i:0>3d}.ply")
        )

        example_i += 1

    print("example_i: ", example_i)