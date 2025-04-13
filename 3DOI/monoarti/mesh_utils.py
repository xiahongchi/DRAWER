import numpy as np
import torch
import torch.nn.functional as F
import collections
from collections import defaultdict
import cv2
import random
import math
import quaternion
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import TexturesVertex, TexturesUV, Textures
import os
import shutil
import imageio
from typing import Optional, List


def triangulate_pcd(sem_mask, semantic_pred, render_size, border_size):
    #verts = torch.ones(render_size).nonzero().numpy()
    verts = sem_mask.nonzero().numpy()
    vert_id_map = defaultdict(dict)
    for idx, vert in enumerate(verts):
        vert_id_map[vert[0]][vert[1]] = idx# + len(verts)

    height = render_size[0] - border_size * 2
    width = render_size[1] - border_size * 2

    semantic_pred = semantic_pred.numpy()

    triangles = []
    for vert in verts:
        # upper right triangle
        if (
            vert[0] < height - 1
            and vert[1] < width - 1
            and sem_mask[vert[0] + 1][vert[1] + 1]
            and sem_mask[vert[0]][vert[1] + 1]
            and semantic_pred[vert[0]][vert[1]] == semantic_pred[vert[0] + 1][vert[1] + 1]
            and semantic_pred[vert[0]][vert[1]] == semantic_pred[vert[0]][vert[1] + 1]
        ):
            triangles.append(
                [
                    vert_id_map[vert[0]][vert[1]],
                    vert_id_map[vert[0] + 1][vert[1] + 1],
                    vert_id_map[vert[0]][vert[1] + 1],
                ]
            )
        # bottom left triangle
        if (
            vert[0] < height - 1
            and vert[1] < width - 1
            and sem_mask[vert[0] + 1][vert[1] + 1]
            and sem_mask[vert[0]][vert[1] + 1]
            and semantic_pred[vert[0]][vert[1]] == semantic_pred[vert[0] + 1][vert[1]]
            and semantic_pred[vert[0]][vert[1]] == semantic_pred[vert[0] + 1][vert[1] + 1]
        ):
            triangles.append(
                [
                    vert_id_map[vert[0]][vert[1]],
                    vert_id_map[vert[0] + 1][vert[1]],
                    vert_id_map[vert[0] + 1][vert[1] + 1],
                ]
            )
    triangles = np.array(triangles)
    triangles = torch.LongTensor(triangles)

    return triangles


def get_pcd(verts, normal, offset, h=480, w=640, focal_length=517.97):
    """
    Copy from 
    https://github.com/JasonQSY/Articulation3D/blob/master/articulation3d/articulation3d/utils/vis.py

    convert 2d verts to 3d point cloud based on plane normal and offset
    depth = offset / n \dot K^{-1}q
    """
    offset_x = w/2
    offset_y = h/2
    K = [[focal_length, 0, offset_x],
        [0, focal_length, offset_y],
        [0, 0, 1]]
    K_inv = np.linalg.inv(np.array(K))
    homogeneous = np.hstack((verts, np.ones(len(verts)).reshape(-1,1)))
    ray = K_inv@homogeneous.T
    depth = offset / np.dot(normal, ray)
    pcd = depth.reshape(-1,1) * ray.T

    #import pdb; pdb.set_trace()
    return pcd


def fit_homography(src_pts, tgt_pts):
    """
    Fit a homography from src_pts to tgt_pts.

    src_pts: torch.LongTensor shape (N x 2)
    tgt_pts: torch.LongTensor shape (N x 2)
    
    """
    src_pts = src_pts.numpy().astype(np.float32)
    tgt_pts = tgt_pts.numpy().astype(np.float32)

    N = 4
    
    # randomly pick up 4 control points
    #ids = random.sample(range(src_pts.shape[0]), 4)
    #src_pts = 
    #import pdb; pdb.set_trace()
    H, mask = cv2.findHomography(src_pts, tgt_pts, cv2.RANSAC, 5.0)
    #H = cv2.getPerspectiveTransform(obj_mask.nonzero().cpu().numpy().astype(np.float32), pts_reproj[1].numpy().astype(np.float32))

    return H

def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):

    def compute_length_vec3(vec3):
        return math.sqrt(vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2])
    
    def rotation(axis, angle):
        rot = np.eye(4)
        c = np.cos(-angle)
        s = np.sin(-angle)
        t = 1.0 - c
        axis /= compute_length_vec3(axis)
        x = axis[0]
        y = axis[1]
        z = axis[2]
        rot[0,0] = 1 + t*(x*x-1)
        rot[0,1] = z*s+t*x*y
        rot[0,2] = -y*s+t*x*z
        rot[1,0] = -z*s+t*x*y
        rot[1,1] = 1+t*(y*y-1)
        rot[1,2] = x*s+t*y*z
        rot[2,0] = y*s+t*x*z
        rot[2,1] = -x*s+t*y*z
        rot[2,2] = 1+t*(z*z-1)
        return rot


    verts = []
    indices = []
    diff = (p1 - p0).astype(np.float32)
    height = compute_length_vec3(diff)
    for i in range(stacks+1):
        for i2 in range(slices):
            theta = i2 * 2.0 * math.pi / slices
            pos = np.array([radius*math.cos(theta), radius*math.sin(theta), height*i/stacks])
            verts.append(pos)
    for i in range(stacks):
        for i2 in range(slices):
            i2p1 = math.fmod(i2 + 1, slices)
            indices.append( np.array([(i + 1)*slices + i2, i*slices + i2, i*slices + i2p1], dtype=np.uint32) )
            indices.append( np.array([(i + 1)*slices + i2, i*slices + i2p1, (i + 1)*slices + i2p1], dtype=np.uint32) )
    transform = np.eye(4)
    va = np.array([0, 0, 1], dtype=np.float32)
    vb = diff
    vb /= compute_length_vec3(vb)
    axis = np.cross(vb, va)
    angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
    if angle != 0:
        if compute_length_vec3(axis) == 0:
            dotx = va[0]
            if (math.fabs(dotx) != 1.0):
                axis = np.array([1,0,0]) - dotx * va
            else:
                axis = np.array([0,1,0]) - va[1] * va
            axis /= compute_length_vec3(axis)
        transform = rotation(axis, -angle)
    transform[:3,3] += p0
    verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
    verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]
        
    return verts, indices


def create_arrow_mesh(radius, p0, p1, stacks=10, slices=10, arrow_height=0):

    def compute_length_vec3(vec3):
        return math.sqrt(vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2])
    
    def rotation(axis, angle):
        rot = np.eye(4)
        c = np.cos(-angle)
        s = np.sin(-angle)
        t = 1.0 - c
        axis /= compute_length_vec3(axis)
        x = axis[0]
        y = axis[1]
        z = axis[2]
        rot[0,0] = 1 + t*(x*x-1)
        rot[0,1] = z*s+t*x*y
        rot[0,2] = -y*s+t*x*z
        rot[1,0] = -z*s+t*x*y
        rot[1,1] = 1+t*(y*y-1)
        rot[1,2] = x*s+t*y*z
        rot[2,0] = y*s+t*x*z
        rot[2,1] = -x*s+t*y*z
        rot[2,2] = 1+t*(z*z-1)
        return rot


    verts = []
    indices = []
    diff = (p1 - p0).astype(np.float32)
    height = compute_length_vec3(diff)
    for i in range(stacks+2):
        if i == stacks+1:
            # arrow tip
            cur_radius = 0
            cur_height = height
        elif i == stacks:
            # arrow base
            cur_radius = radius*3
            cur_height = height * (1-arrow_height) * (i-1)/stacks
        else:
            # cylinder
            cur_radius = radius
            cur_height = height * (1-arrow_height) * i/stacks
        for i2 in range(slices):
            theta = i2 * 2.0 * math.pi / slices
            pos = np.array([cur_radius*math.cos(theta), cur_radius*math.sin(theta), cur_height])
            verts.append(pos)
    for i in range(stacks+1):
        for i2 in range(slices):
            i2p1 = math.fmod(i2 + 1, slices)
            indices.append( np.array([(i + 1)*slices + i2, i*slices + i2, i*slices + i2p1], dtype=np.uint32) )
            indices.append( np.array([(i + 1)*slices + i2, i*slices + i2p1, (i + 1)*slices + i2p1], dtype=np.uint32) )
    transform = np.eye(4)
    va = np.array([0, 0, 1], dtype=np.float32)
    vb = diff
    vb /= compute_length_vec3(vb)
    axis = np.cross(vb, va)
    angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
    if angle != 0:
        if compute_length_vec3(axis) == 0:
            dotx = va[0]
            if (math.fabs(dotx) != 1.0):
                axis = np.array([1,0,0]) - dotx * va
            else:
                axis = np.array([0,1,0]) - va[1] * va
            axis /= compute_length_vec3(axis)
        transform = rotation(axis, -angle)
    transform[:3,3] += p0
    verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
    verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]
        
    return verts, indices



def get_camera_meshes(camera_list, radius=0.02):
    verts_list = []
    faces_list = []
    color_list = []
    rots = np.array([quaternion.as_rotation_matrix(camera_info['rotation']) for camera_info in camera_list])

    # ai habitat frame
    lookat = np.array([0,0,-1])
    vertical = np.array([0,1,0])

    positions = np.array([camera_info['position'] for camera_info in camera_list])
    lookats = rots@lookat.T
    verticals = rots@vertical.T
    predetermined_color = [
        [0.10196, 0.32157, 1.0],
        [1.0, 0.0667, 0.1490],# [0.8314, 0.0667, 0.3490],
        # [0.0, 0.4392156862745098, 0.7529411764705882],
        # [0.3764705882352941, 0.08627450980392155, 0.47843137254901963],
    ]
    for idx, (position, lookat, vertical, color) in enumerate(zip(positions, lookats, verticals, predetermined_color)):
        cur_num_verts = 0
        # r, g, b = create_color_palette()[idx+10]
        edges = get_cone_edges(position, lookat, vertical)
        # color = [r/255.0,g/255.0,b/255.0]
        cam_verts = []
        cam_inds = []
        for k in range(len(edges)):
            cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
            cyl_verts = [x for x in cyl_verts]
            cyl_ind = [x + cur_num_verts for x in cyl_ind]
            cur_num_verts += len(cyl_verts)
            cam_verts.extend(cyl_verts)
            cam_inds.extend(cyl_ind)
        # Create a textures object
        verts_list.append(torch.tensor(cam_verts, dtype=torch.float32))
        faces_list.append(torch.tensor(cam_inds, dtype=torch.float32))
        color_list.append(color)

    color_tensor = torch.tensor(color_list, dtype=torch.float32).unsqueeze_(1)
    #tex = Textures(verts_uvs=None, faces_uvs=None, verts_rgb=color_tensor)
    tex = TexturesVertex(verts_features=color_tensor)

    # Initialise the mesh with textures
    meshes = Meshes(verts=verts_list, faces=faces_list, textures=tex)
    return meshes


def get_cone_edges(position, lookat, vertical):
    def get_cone_verts(position, lookat, vertical):
        vertical = np.array(vertical) / np.linalg.norm(vertical)
        lookat = np.array(lookat) / np.linalg.norm(lookat)
        right = np.cross(np.array(lookat), np.array(vertical))
        right = right / np.linalg.norm(right)
        top = np.cross(right, lookat)
        top = top / np.linalg.norm(top)

        right *= .4
        lookat *= .4
        top *= .1
        verts = {
            'topR': position + lookat + top + right,
            'topL': position + lookat + top - right,
            'center': position,
            'bottomR': position + lookat - top + right,
            'bottomL': position + lookat - top - right,
        }
        return verts

    cone_verts = get_cone_verts(position, lookat, vertical)
    edges = [
        (cone_verts['center'], cone_verts['topR']),
        (cone_verts['center'], cone_verts['topL']),
        (cone_verts['center'], cone_verts['bottomR']),
        (cone_verts['center'], cone_verts['bottomL']),
        (cone_verts['topR'], cone_verts['topL']),
        (cone_verts['bottomR'], cone_verts['topR']),
        (cone_verts['bottomR'], cone_verts['bottomL']),
        (cone_verts['topL'], cone_verts['bottomL']),
    ]
    return edges




def get_axis_mesh(radius, pt1, pt2):
    verts_list = []
    faces_list = []
    color_list = []
    cyl_verts, cyl_ind = create_arrow_mesh(radius, pt1.numpy(), pt2.numpy())
    cyl_verts = [x for x in cyl_verts]
    cyl_ind = [x for x in cyl_ind]

    # Create a textures object
    verts_list.append(torch.tensor(cyl_verts, dtype=torch.float32))
    faces_list.append(torch.tensor(cyl_ind, dtype=torch.float32))
    # color_list.append([0.10196, 0.32157, 1.0])
    # color_tensor = torch.tensor(color_list, dtype=torch.float32).unsqueeze_(1)
    # Textures(verts_uvs=axis_verts_rgb, faces_uvs=axis_pt1.faces_list(), maps=torch.zeros((1,5,5,3)).cuda())
    # tex = TexturesVertex(verts_features=color_tensor)

    # Initialise the mesh with textures
    meshes = Meshes(verts=verts_list, faces=faces_list)
    return meshes


def save_obj_articulation(folder, prefix, meshes, cam_meshes=None, decimal_places=None, blend_flag=False, map_files=None, uv_maps=None):
    os.makedirs(folder, exist_ok=True)

    # pytorch3d does not support map_files
    #map_files = meshes.textures.map_files()
    #assert map_files is not None
    if map_files is None and uv_maps is None:
        raise RuntimeError("either map_files or uv_maps should be set!")

    # generate map_files from uv_map
    if uv_maps is not None and map_files is None:
        map_files = []
        uv_dir = os.path.join(folder, 'uv_maps')
        if not os.path.exists(uv_dir):
            os.mkdir(uv_dir)
        for map_id, uv_map in enumerate(uv_maps):
            uv_path = os.path.join(uv_dir, '{}_uv_plane_{}.png'.format(prefix, map_id))
            #pdb.set_trace()
            imageio.imwrite(uv_path, uv_map)
            map_files.append(uv_path)

    #pdb.set_trace()

    f_mtl = open(os.path.join(folder, prefix+'.mtl'), 'w')
    f = open(os.path.join(folder, prefix+'.obj'), 'w')
    try:
        seen = set()
        uniq_map_files = [m for m in list(map_files) if m not in seen and not seen.add(m)]
        for map_id, map_file in enumerate(uniq_map_files):
            if uv_maps is not None:
                # we do not need to copy map_files,
                # they are already in uv_maps/...
                f_mtl.write(_get_mtl_map(
                    os.path.basename(map_file).split('.')[0], 
                    os.path.join('uv_maps', os.path.basename(map_file))
                ))
                continue

            if not blend_flag:
                shutil.copy(map_file, folder)
                os.chmod(os.path.join(folder, os.path.basename(map_file)), 0o755)
                f_mtl.write(_get_mtl_map(os.path.basename(map_file).split('.')[0], os.path.basename(map_file)))
            else:
                rgb = cv2.imread(map_file, cv2.IMREAD_COLOR)
                if cam_meshes is not None:
                    blend_color = np.array(cam_meshes.textures.verts_features_packed().numpy().tolist()[map_id])*255
                else:
                    blend_color = np.array(create_color_palette()[map_id+10])
                alpha = 0.7
                blend = (rgb*alpha + blend_color[::-1]*(1-alpha)).astype(np.uint8)
                cv2.imwrite(os.path.join(folder, os.path.basename(map_file).split('.')[0]+'_debug.png'), blend)
                f_mtl.write(_get_mtl_map(os.path.basename(map_file).split('.')[0], os.path.basename(map_file).split('.')[0]+'_debug.png'))
        
        f.write(f"mtllib {prefix}.mtl\n\n")
        # we want [list]    verts, vert_uvs, map_files; 
        #         [packed]  faces;
        #         face per mesh
        verts_list = meshes.verts_list()
        verts_uvs_list = meshes.textures.verts_uvs_list()
        faces_list = meshes.faces_packed().split(meshes.num_faces_per_mesh().tolist(), dim=0)
        #pdb.set_trace()
        for idx, (verts, verts_uvs, faces, map_file) in enumerate(zip(verts_list, verts_uvs_list, faces_list, map_files)):
            f.write(f"# mesh {idx}\n")
            trunc_verts_uvs = verts_uvs[:verts.shape[0]]
            _save(f, verts, faces, verts_uv=trunc_verts_uvs, map_file=map_file, idx=idx, decimal_places=decimal_places)
        if cam_meshes:
            face_offset = np.sum([len(v) for v in verts_list])
            cam_verts_list = cam_meshes.verts_list()
            cam_verts_rgbs_list = cam_meshes.textures.verts_features_packed().numpy().tolist()
            cam_faces_list = (cam_meshes.faces_packed()+face_offset).split(cam_meshes.num_faces_per_mesh().tolist(), dim=0)
            assert(len(cam_verts_rgbs_list) == len(cam_verts_list))
            for idx, (verts, faces, rgb) in enumerate(zip(cam_verts_list, cam_faces_list, cam_verts_rgbs_list)):
                f.write(f"# camera {idx}\n")
                f_mtl.write(_get_mtl_rgb(idx, rgb))
                _save(f, verts, faces, rgb=rgb, idx=idx, decimal_places=decimal_places)
    finally:
        f.close()
        f_mtl.close()


def _get_mtl_map(material_name, map_Kd):
        return f"""newmtl {material_name}
map_Kd {map_Kd}
# Test colors
Ka 1.000 1.000 1.000  # white
Kd 1.000 1.000 1.000  # white
Ks 0.000 0.000 0.000  # black
Ns 10.0\n"""


def _get_mtl_rgb(material_idx, rgb):
        return f"""newmtl color_{material_idx}
Kd {rgb[0]} {rgb[1]} {rgb[2]}
Ka 0.000 0.000 0.000\n"""


def _save(f, verts, faces, verts_uv=None, map_file=None, rgb=None, idx=None, double_sided=True, decimal_places: Optional[int] = None):
    if decimal_places is None:
        float_str = "%f"
    else:
        float_str = "%" + ".%df" % decimal_places

    lines = ""
    
    V, D = verts.shape
    for i in range(V):
        vert = [float_str % verts[i, j] for j in range(D)]
        lines += "v %s\n" % " ".join(vert)

    if verts_uv is not None:
        V, D = verts_uv.shape
        for i in range(V):
            vert_uv = [float_str % verts_uv[i, j] for j in range(D)]
            lines += "vt %s\n" % " ".join(vert_uv)

    if map_file is not None:
        lines += f"usemtl {os.path.basename(map_file).split('.')[0]}\n"
    elif rgb is not None:
        lines += f"usemtl color_{idx}\n"    

    if faces != []:
        F, P = faces.shape
        for i in range(F):
            if verts_uv is not None:
                face = ["%d/%d" % (faces[i, j] + 1, faces[i, j] + 1) for j in range(P)]
            else:
                face = ["%d" % (faces[i, j] + 1) for j in range(P)]
            # if i + 1 < F:
            lines += "f %s\n" % " ".join(face)
            if double_sided:
                if verts_uv is not None:
                    face = ["%d/%d" % (faces[i, j] + 1, faces[i, j] + 1) for j in reversed(range(P))]
                else:
                    face = ["%d" % (faces[i, j] + 1) for j in reversed(range(P))]
                lines += "f %s\n" % " ".join(face)
            # elif i + 1 == F:
            #     # No newline at the end of the file.
            #     lines += "f %s" % " ".join(face)
    else:
        print(f"face = []")
    f.write(lines)
