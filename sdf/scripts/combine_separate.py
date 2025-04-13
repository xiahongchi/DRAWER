import os
import trimesh
import numpy as np
import torch
from pytorch3d.io import save_obj, load_objs_as_meshes
import pickle
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import nvdiffrast.torch as dr
import pymeshlab as pml
from collections import deque

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



def find_closest_vertices_bfs(graph, input_verts):
    # Convert input vertices to a set for fast look-up
    input_verts_set = set(input_verts)
    closest_verts = {}

    for vert in input_verts:
        # Use a queue for BFS, initialized with the starting vertex and distance 0
        queue = deque([(vert, 0)])
        visited = set([vert])

        while queue:
            current_node, distance = queue.popleft()

            # Explore neighbors
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)

                    # If the neighbor is not in the input list, we found the closest vertex
                    if neighbor not in input_verts_set:
                        closest_verts[vert] = (neighbor, distance + 1)
                        queue.clear()  # Clear the queue to stop the BFS
                        break

                    # Otherwise, continue BFS
                    queue.append((neighbor, distance + 1))

    return closest_verts

def largest_connected_component_faces(mesh, face_indices=None):
    # Get the face adjacency pairs (each row is a pair of adjacent faces)
    adjacency = mesh.face_adjacency

    # Create a graph from the adjacency list
    G = nx.Graph()
    G.add_edges_from(adjacency)

    # If a specific subset of faces is provided, filter the graph
    if face_indices is not None:
        G = G.subgraph(face_indices)

    # Find all connected components in the graph
    connected_components = nx.connected_components(G)

    # Find the largest connected component
    largest_component = max(connected_components, key=len)

    # Return the list of faces in the largest connected component
    return np.array(list(largest_component), dtype=np.int32)


@torch.no_grad()
def mask_faces_in_texture(v, f, vt, ft, uv_size, map, glctx, face_indices):

    device = "cuda"

    uv_size_h = uv_size[0]
    uv_size_w = uv_size[1]

    vt = vt.float().to(device)
    ft = ft.int().to(device)
    map = map.float().to(device)

    # render uv maps
    uv = vt * 2.0 - 1.0  # uvs to range [-1, 1]
    uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)  # [N, 4]

    rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (uv_size_h, uv_size_w))  # [1, h, w, 4]

    map_flatten = map.reshape(-1, 3)
    face_id = (rast[..., -1] - 1).reshape(uv_size_h, uv_size_w).long()
    valid_pixel = (rast[..., -1] > 0).reshape(uv_size_h, uv_size_w)
    face_id = torch.flip(face_id, [0])
    valid_pixel = torch.flip(valid_pixel, [0])
    valid_pixel_flatten = valid_pixel.reshape(-1).clone()
    map_flatten_valid = map_flatten[valid_pixel_flatten]

    # v = v.to(device).reshape(-1, 3).float()
    # f = f.to(device).reshape(-1, 3).int()

    mesh = trimesh.Trimesh(
        v.cpu().numpy(),
        f.cpu().numpy(),
    )
    adjacency = mesh.face_adjacency

    print("face_indices: ", face_indices.shape)

    adjacency_unique = np.unique(adjacency).reshape(-1)
    face_indices = face_indices.reshape(-1)
    face_indices = face_indices[np.in1d(face_indices, adjacency_unique)]

    G = nx.Graph()
    G.add_edges_from(adjacency)

    faces_adj = find_closest_vertices_bfs(G, face_indices)
    # faces_adj = torch.from_numpy(faces_adj).long()
    face_indices = torch.from_numpy(face_indices).long()

    adj_face_color = {}
    for face_idx, (face_adj_idx, _) in faces_adj.items():
        mask_tex_faces = np.zeros((f.shape[0]), dtype=np.bool_)
        mask_tex_faces[face_adj_idx] = True

        mask_tex_faces = torch.from_numpy(mask_tex_faces).to(device)
        mask_tex_faces = mask_tex_faces[face_id[valid_pixel].reshape(-1)]

        adj_face_color[face_idx] = map_flatten_valid[mask_tex_faces].mean(dim=0).reshape(1, 3)

    # all_tex_mask = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f)[0].reshape(uv_size_h, uv_size_w)
    # all_tex_mask = torch.flip(all_tex_mask, [0])

    for face_idx in face_indices:
        mask_tex_faces = np.zeros((f.shape[0]), dtype=np.bool_)
        mask_tex_faces[int(face_idx)] = True

        mask_tex_faces = torch.from_numpy(mask_tex_faces).to(device)
        mask_tex_faces = mask_tex_faces[face_id[valid_pixel].reshape(-1)]
        if int(face_idx) in adj_face_color:
            map_flatten_valid[mask_tex_faces] = adj_face_color[int(face_idx)]
        else:
            map_flatten_valid[mask_tex_faces] = torch.tensor([1., 1., 1.]).reshape(1, 3).to(device)

        map_flatten[valid_pixel_flatten] = map_flatten_valid

    map = map_flatten.reshape(uv_size_h, uv_size_w, 3)

    return map

def combine_texture_meshes(v_list, f_list, vt_list, ft_list, map_list):
    combined_v = []
    combined_f_sep = []
    combined_vt = []
    combined_ft = []
    v_cnt = 0
    vt_cnt = 0
    for v, f, vt, ft in zip(v_list, f_list, vt_list, ft_list):
        combined_v.append(v)
        combined_f_sep.append(f + v_cnt)
        v_cnt += v.shape[0]

        vt[:, 0] = vt[:, 0] + 1.0 * (len(combined_v) - 1)

        combined_vt.append(vt)
        combined_ft.append(ft + vt_cnt)
        vt_cnt += vt.shape[0]

    combined_v = torch.cat(combined_v, dim=0)
    combined_f = torch.cat(combined_f_sep, dim=0)
    combined_vt = torch.cat(combined_vt, dim=0)
    combined_ft = torch.cat(combined_ft, dim=0)
    combined_map = torch.cat(map_list, dim=1)

    combined_vt[:, 0] = combined_vt[:, 0] / len(v_list)

    return combined_v, combined_f, combined_vt, combined_ft, combined_map, combined_f_sep

v_list = []
f_list = []
vt_list = []
ft_list = []
map_list = []

glctx = dr.RasterizeGLContext(output_db=False)

main_mesh_name = "mesh.obj"

# separate_mesh_dir = "/projects/perception/personals/hongchix/codes/psdf/outputs/uw_kitchen/241024_cs_kitchen_n_bakedsdf_sdfstudio_normal_mono_5em1_depth_mono_2e0/bakedsdf/2024-10-24_141450/separate/texture_mesh"
# mesh_object_names = [
#     "mesh-box_0.obj",
#     "mesh-box_1.obj",
#     "mesh-box_2.obj",
#     "mesh-box_3.obj",
# ]
# separate_mesh_dir = "/projects/perception/personals/hongchix/codes/psdf/outputs/cs_kitchen/241003_cs_kitchen_bakedsdf_sdfstudio_normal_mono_1em1_depth_mono_2e0/bakedsdf/2024-10-03_180313/separate/texture_mesh"
# mesh_object_names = [
#     "mesh-box_0.obj",
#     "mesh-box_1.obj",
#     "mesh-box_2.obj",
# ]
# separate_mesh_dir = "/projects/perception/personals/hongchix/codes/psdf/outputs/cs_kitchen/241013_uw_kitchen_bakedsdf_sdfstudio_normal_mono_5em1_depth_mono_2e0/bakedsdf/2024-10-13_021348/separate/texture_mesh"
# mesh_object_names = [
#     "mesh-box_0.obj",
# ]
# separate_mesh_dir = "outputs/uw_kitchen/250123_cc_bedroom_bakedsdf_sdfstudio_normal_mono_5em1_depth_mono_2e0/bakedsdf/2025-01-23_010641/separate/texture_mesh"
# mesh_object_names = [
#     "mesh-box_0.obj",
# ]
separate_mesh_dir = "outputs/uw_kitchen/250123_cs_office_bakedsdf_sdfstudio_normal_mono_5em1_depth_mono_2e0/bakedsdf/2025-01-23_134107/separate/texture_mesh"
mesh_object_names = [
    "mesh-box_0.obj",
    "mesh-box_1.obj",
]
close_dist = 0.006
main_overlap_faces_list = []

main_mesh_path = os.path.join(separate_mesh_dir, main_mesh_name)
main_mesh = load_objs_as_meshes([main_mesh_path], device="cpu")
main_mesh_verts = main_mesh.verts_packed().cpu().numpy()
main_mesh_faces = main_mesh.faces_packed().cpu().numpy()
main_mesh_face_centroids = main_mesh_verts[main_mesh_faces.reshape(-1)].reshape(-1, 3, 3).mean(axis=1).reshape(-1, 3)
knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(main_mesh_face_centroids)

for mesh_name in mesh_object_names:
    mesh_object_path = os.path.join(separate_mesh_dir, mesh_name)
    object_mesh = load_objs_as_meshes([mesh_object_path], device="cpu")

    object_mesh_verts = object_mesh.verts_packed().cpu().numpy()
    object_mesh_faces = object_mesh.faces_packed().cpu().numpy()

    print("object_mesh_verts: ", object_mesh_verts.shape)
    print("object_mesh_faces: ", object_mesh_faces.shape)

    object_mesh_trimesh = trimesh.Trimesh(
        object_mesh_verts,
        object_mesh_faces,
        process=False
    )

    selected_face_indices = largest_connected_component_faces(object_mesh_trimesh)
    selected_face_indices = torch.from_numpy(selected_face_indices).long()

    object_mesh_tex = object_mesh.textures
    object_mesh_tex = object_mesh_tex.submeshes(None, [[selected_face_indices]])
    object_mesh = object_mesh.submeshes([[selected_face_indices]])
    object_mesh.textures = object_mesh_tex

    object_mesh_verts = object_mesh.verts_packed().cpu().numpy()
    object_mesh_faces = object_mesh.faces_packed().cpu().numpy()

    print("object_mesh_verts: ", object_mesh_verts.shape)
    print("object_mesh_faces: ", object_mesh_faces.shape)

    object_mesh_face_centroids = object_mesh_verts[object_mesh_faces.reshape(-1)].reshape(-1, 3, 3).mean(
        axis=1).reshape(-1, 3)
    dists, indices = knn.kneighbors(object_mesh_face_centroids)
    dists = dists.reshape(-1)
    indices = indices.reshape(-1)

    object_overlap_faces = np.arange(indices.shape[0])[dists < close_dist]

    # keep_verts_indices = np.unique(object_mesh_faces[dists < close_dist])
    # keep_verts = np.zeros(object_mesh_verts.shape[0]) > 0
    # keep_verts[keep_verts_indices] = True
    #
    # object_mesh_verts_overlapped, object_mesh_faces_overlapped, _ = filter_mesh_from_vertices(keep_verts, object_mesh_verts, object_mesh_faces)
    #
    # m = pml.Mesh(object_mesh_verts_overlapped, object_mesh_faces_overlapped)
    # ms = pml.MeshSet()
    # ms.add_mesh(m, 'mesh')  # will copy!
    # ms.laplacian_smooth_surface_preserving(angledeg=45, iterations=100)
    # m = ms.current_mesh()
    # object_mesh_verts_overlapped = m.vertex_matrix()
    # object_mesh_faces_overlapped = m.face_matrix()
    #
    # object_mesh_verts[keep_verts] = object_mesh_verts_overlapped
    #
    # object_mesh.update_padded(torch.from_numpy(object_mesh_verts).float().unsqueeze(0))

    object_mesh_mask_colors = np.ones((indices.shape[0], 3))
    object_mesh_mask_colors[object_overlap_faces] = 0.

    print("object_overlap_faces: ", object_overlap_faces.shape)

    updated_map = mask_faces_in_texture(
        object_mesh.verts_packed(),
        object_mesh.faces_packed(),
        object_mesh.textures.verts_uvs_padded().reshape(-1, 2),
        object_mesh.textures.faces_uvs_padded().reshape(-1, 3),
        object_mesh.textures.maps_padded().squeeze(0).shape[0:2],
        object_mesh.textures.maps_padded().squeeze(0),
        glctx,
        object_overlap_faces
    )


    save_obj(
        os.path.join(separate_mesh_dir, os.path.splitext(mesh_name)[0]+"_connected.obj"),
        object_mesh.verts_packed(),
        object_mesh.faces_packed(),
        verts_uvs=object_mesh.textures.verts_uvs_padded().reshape(-1, 2),
        faces_uvs=object_mesh.textures.faces_uvs_padded().reshape(-1, 3),
        texture_map=updated_map
    )

    v_list.append(object_mesh.verts_packed())
    f_list.append(object_mesh.faces_packed())
    vt_list.append(object_mesh.textures.verts_uvs_padded().reshape(-1, 2))
    ft_list.append(object_mesh.textures.faces_uvs_padded().reshape(-1, 3))
    map_list.append(updated_map)


    trimesh.exchange.export.export_mesh(
        trimesh.Trimesh(
            object_mesh_verts,
            object_mesh_faces,
            face_colors=object_mesh_mask_colors
        ),
        os.path.join(separate_mesh_dir, os.path.splitext(mesh_name)[0]+"_mask.ply")
    )

    knn_object = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(object_mesh_face_centroids)
    dists, indices = knn_object.kneighbors(main_mesh_face_centroids)
    dists = dists.reshape(-1)
    indices = indices.reshape(-1)
    main_overlap_faces = np.arange(indices.shape[0])[dists < close_dist]

    main_overlap_faces_list.append(main_overlap_faces)

main_overlap_faces = np.concatenate(main_overlap_faces_list, axis=0)


keep_verts_indices = np.unique(main_mesh_faces[main_overlap_faces])
keep_verts = np.zeros(main_mesh_verts.shape[0]) > 0
keep_verts[keep_verts_indices] = True

main_mesh_verts_overlapped, main_mesh_faces_overlapped, _ = filter_mesh_from_vertices(keep_verts, main_mesh_verts, main_mesh_faces)

m = pml.Mesh(main_mesh_verts_overlapped, main_mesh_faces_overlapped)
ms = pml.MeshSet()
ms.add_mesh(m, 'mesh')  # will copy!
ms.laplacian_smooth_surface_preserving(angledeg=45, iterations=100)

m = ms.current_mesh()
main_mesh_verts_overlapped = m.vertex_matrix()
main_mesh_faces_overlapped = m.face_matrix()

main_mesh_verts[keep_verts] = main_mesh_verts_overlapped

main_mesh.update_padded(torch.from_numpy(main_mesh_verts).float().unsqueeze(0))


updated_map = mask_faces_in_texture(
    main_mesh.verts_packed(),
    main_mesh.faces_packed(),
    main_mesh.textures.verts_uvs_padded().reshape(-1, 2),
    main_mesh.textures.faces_uvs_padded().reshape(-1, 3),
    main_mesh.textures.maps_padded().squeeze(0).shape[0:2],
    main_mesh.textures.maps_padded().squeeze(0),
    glctx,
    main_overlap_faces
)

main_mesh_mask_colors = np.ones((main_mesh_faces.shape[0], 3))
main_mesh_mask_colors[main_overlap_faces] = 0.

trimesh.exchange.export.export_mesh(
    trimesh.Trimesh(
        main_mesh_verts,
        main_mesh_faces,
        face_colors=main_mesh_mask_colors
    ),
    os.path.join(separate_mesh_dir, os.path.splitext(main_mesh_name)[0]+"_mask.ply")
)

save_obj(
    os.path.join(separate_mesh_dir, os.path.splitext(main_mesh_name)[0]+"_inpainted.obj"),
    main_mesh.verts_packed(),
    main_mesh.faces_packed(),
    verts_uvs=main_mesh.textures.verts_uvs_padded().reshape(-1, 2),
    faces_uvs=main_mesh.textures.faces_uvs_padded().reshape(-1, 3),
    texture_map=updated_map
)

v_list.append(main_mesh.verts_packed())
f_list.append(main_mesh.faces_packed())
vt_list.append(main_mesh.textures.verts_uvs_padded().reshape(-1, 2))
ft_list.append(main_mesh.textures.faces_uvs_padded().reshape(-1, 3))
map_list.append(updated_map)

combined_v, combined_f, combined_vt, combined_ft, combined_map, combined_f_sep = combine_texture_meshes(v_list, f_list, vt_list, ft_list, map_list)

save_obj(
    os.path.join(separate_mesh_dir, "combined.obj"),
    combined_v,
    combined_f,
    verts_uvs=combined_vt,
    faces_uvs=combined_ft,
    texture_map=combined_map
)

combined_f_sep = [f_sep.cpu().numpy() for f_sep in combined_f_sep]
with open(os.path.join(separate_mesh_dir, "combined.pkl"), 'wb') as f:
    pickle.dump(combined_f_sep, f)

