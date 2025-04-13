from __future__ import annotations
import os
os.environ['HF_HOME'] = "/u/hongchix/scratch/"
import sys
sys.path.append("./scripts")
sys.path.append("./scripts/lama")

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
from saicinpainting.training.trainers import load_checkpoint

from collections import namedtuple
from pytorch3d.renderer import TexturesUV, TexturesVertex, RasterizationSettings, MeshRenderer, MeshRendererWithFragments, MeshRasterizer, HardPhongShader, SoftSilhouetteShader, PointLights, AmbientLights, FoVPerspectiveCameras, look_at_view_transform
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType

import torch
import numpy as np
from PIL import Image
import json
from torchvision import transforms
from pathlib import Path

from pytorch3d.renderer import TexturesUV
from pytorch3d.io import (
    load_objs_as_meshes
)
from tex_inpaint_helpers.projection_helper import (
    build_similarity_texture_cache_for_all_views,
    render_one_view_and_build_masks,
    backproject_from_image,
)
from tex_inpaint_helpers.io_helper import (
    save_backproject_obj
)
from omegaconf import OmegaConf
import yaml
from nerfstudio.utils.eval_utils import eval_setup, resume_setup_gs_uc

import time
from scipy.ndimage import binary_dilation, binary_erosion
import networkx as nx
import pickle
from sklearn.neighbors import NearestNeighbors
import nvdiffrast.torch as dr
from tqdm import tqdm

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
    keep_faces = torch.logical_or(keep_0, keep_1)
    keep_faces = torch.logical_or(keep_faces, keep_2)

    return keep_faces

def point_transform(point, transform):
    point_shape = point.shape
    point_pad = np.pad(point.reshape(-1, 3), ((0, 0), (0, 1)), constant_values=(0, 1))
    point_transformed_pad = point_pad @ transform.T
    point_transformed = point_transformed_pad[:, :3] / point_transformed_pad[:, 3:]
    return point_transformed.reshape(point_shape)

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
    }

def get_centroids(mesh):
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    centroids = verts[faces.reshape(-1)].reshape(-1, 3, 3).mean(dim=1)
    return centroids

@torch.no_grad()
def mask_faces_in_texture(v, f, vt, ft, uv_size, glctx, face_indices):

    device = "cuda"

    uv_size_h = uv_size[0]
    uv_size_w = uv_size[1]

    vt = torch.from_numpy(vt.astype(np.float32)).float().to(device)
    ft = torch.from_numpy(ft.astype(np.int64)).int().to(device)

    # render uv maps
    uv = vt * 2.0 - 1.0  # uvs to range [-1, 1]
    uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1)  # [N, 4]

    rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (uv_size_h, uv_size_w))  # [1, h, w, 4]

    v = torch.from_numpy(v).to(device).reshape(-1, 3).float()
    f = torch.from_numpy(f).to(device).reshape(-1, 3).int()

    all_tex_mask = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f)[0].reshape(uv_size_h, uv_size_w)
    all_tex_mask = torch.flip(all_tex_mask, [0])

    mask_tex_faces = np.zeros((f.shape[0]), dtype=np.bool_)
    mask_tex_faces[face_indices] = True

    face_id = (rast[..., -1] - 1).reshape(uv_size_h, uv_size_w).long()
    valid_pixel = (rast[..., -1] > 0).reshape(uv_size_h, uv_size_w)
    face_id = torch.flip(face_id, [0])
    valid_pixel = torch.flip(valid_pixel, [0])

    mask_tex_faces = torch.logical_not(torch.from_numpy(mask_tex_faces).to(device))
    mask_tex_faces = mask_tex_faces[face_id[valid_pixel].reshape(-1)]

    all_tex_mask_flatten = all_tex_mask.reshape(-1).clone()
    valid_pixel_flatten = valid_pixel.reshape(-1).clone()
    all_tex_mask_flatten_valid = all_tex_mask_flatten[valid_pixel_flatten]
    all_tex_mask_flatten_valid[mask_tex_faces] = 0.
    all_tex_mask_flatten[valid_pixel_flatten] = all_tex_mask_flatten_valid
    exist_tex_mask = all_tex_mask_flatten.reshape(uv_size_h, uv_size_w)


    tex_mask = {
        "all": all_tex_mask.cpu().numpy(),
        "exist": exist_tex_mask.cpu().numpy(),
    }

    return tex_mask


seed = 42
np.random.seed(seed)

device = torch.device("cuda")
lama_config_path = "./scripts/lama/configs/prediction/default.yaml"
predict_config = OmegaConf.load(lama_config_path)
predict_config.model.path = "/u/hongchix/codes/ns_revival/scripts/lama/big-lama/"
train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
with open(train_config_path, 'r') as f:
    train_config = OmegaConf.create(yaml.safe_load(f))
train_config.training_model.predict_only = True
train_config.visualizer.kind = 'noop'
out_ext = predict_config.get('out_ext', '.png')
checkpoint_path = os.path.join(predict_config.model.path,
                               'models',
                               predict_config.model.checkpoint)
model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
model.freeze()
if not predict_config.get('refine', False):
    model.to(device)

# room_structure_mesh_path = "/projects/perception/personals/hongchix/codes/psdf/vis/room_structure/texture_mesh/mesh-simplify.obj"
# lama_prepare_dir = "/home/hongchix/codes/psdf/vis/room_structure/lama_prepare_views/"
# log_dir = "/home/hongchix/codes/psdf/vis/room_structure/lama_tex_inpaint"

# room_structure_mesh_path = "/projects/perception/personals/hongchix/codes/psdf/vis/ns_data/room2_table_039_filled/texture_mesh/mesh-simplify.obj"
# lama_prepare_dir = "/projects/perception/personals/hongchix/codes/psdf/vis/ns_data/room2_table_039_filled/lama_prepare"
# log_dir = "/projects/perception/personals/hongchix/codes/psdf/vis/ns_data/room2_table_039_filled/lama_inpaint_mesh"

# load_config = "outputs/meshgauss/241005_cs_kitchen_splatfacto_on_mesh_bakedsdf_sdfstudio_normal_mono_depth_mono/splatfacto_on_mesh_uc/2024-10-25_235605/config.yml"
# bakedsdf_dir = "/home/hongchix/codes/psdf/outputs/cs_kitchen/241003_cs_kitchen_bakedsdf_sdfstudio_normal_mono_1em1_depth_mono_2e0/bakedsdf/2024-10-03_180313/"
# load_config = "outputs/meshgauss/241005_cs_kitchen_splatfacto_on_mesh_bakedsdf_sdfstudio_normal_mono_depth_mono/splatfacto_on_mesh_uc/2024-10-28_184255/config.yml"
# bakedsdf_dir = "/home/hongchix/codes/psdf/outputs/uw_kitchen/241024_cs_kitchen_n_bakedsdf_sdfstudio_normal_mono_5em1_depth_mono_2e0/bakedsdf/2024-10-24_141450"
# load_config = "outputs/meshgauss/241005_cs_kitchen_splatfacto_on_mesh_bakedsdf_sdfstudio_normal_mono_depth_mono/splatfacto_on_mesh_uc/2024-11-08_005659/config.yml"
# bakedsdf_dir = "/home/hongchix/codes/psdf/outputs/cs_kitchen/241013_uw_kitchen_bakedsdf_sdfstudio_normal_mono_5em1_depth_mono_2e0/bakedsdf/2024-10-13_021348/"

# load_config = "outputs/meshgauss/241005_cs_kitchen_splatfacto_on_mesh_bakedsdf_sdfstudio_normal_mono_depth_mono/splatfacto_on_mesh_uc/2024-11-08_234546/config.yml"
# bakedsdf_dir = "/home/hongchix/codes/psdf/outputs/uw_kitchen/241024_cs_kitchen_n_bakedsdf_sdfstudio_normal_mono_5em1_depth_mono_2e0/bakedsdf/2024-10-24_141450"

# load_config = "outputs/meshgauss/241005_cs_kitchen_splatfacto_on_mesh_bakedsdf_sdfstudio_normal_mono_depth_mono/splatfacto_on_mesh_uc/2024-11-09_004246/config.yml"
# bakedsdf_dir = "/home/hongchix/codes/psdf/outputs/cs_kitchen/241003_cs_kitchen_bakedsdf_sdfstudio_normal_mono_1em1_depth_mono_2e0/bakedsdf/2024-10-03_180313/"

# load_config = "outputs/meshgauss/splatfacto_on_mesh_uc_cc_bedroom/splatfacto_on_mesh_uc/2025-01-23_145238/config.yml"
# bakedsdf_dir = "/u/hongchix/codes/psdf/outputs/uw_kitchen/250123_cc_bedroom_bakedsdf_sdfstudio_normal_mono_5em1_depth_mono_2e0/bakedsdf/2025-01-23_010641/"

load_config = "outputs/meshgauss/splatfacto_on_mesh_uc_cc_bedroom/splatfacto_on_mesh_uc/2025-01-24_144839/config.yml"
bakedsdf_dir = "/u/hongchix/codes/psdf/outputs/uw_kitchen/250123_cs_office_bakedsdf_sdfstudio_normal_mono_5em1_depth_mono_2e0/bakedsdf/2025-01-23_134107/"

separate_mesh_dir = os.path.join(bakedsdf_dir, "separate/texture_mesh/")
objects_faces_list_path = os.path.join(separate_mesh_dir, "combined.pkl")
log_dir = os.path.join(bakedsdf_dir, "separate/texture_mesh/lama_inpaint")

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

main_mesh_path = os.path.join(separate_mesh_dir, f"mesh_inpainted.obj")
main_tex_mesh = load_objs_as_meshes([main_mesh_path], device="cuda")

config, pipeline, checkpoint_path, _, optimizer = resume_setup_gs_uc(Path(load_config))

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
glctx = dr.RasterizeGLContext(output_db=False)

room_structure_mesh_dict = {
    "v": main_tex_mesh.verts_packed().reshape(-1, 3).cpu().numpy(),
    "f": main_tex_mesh.faces_packed().reshape(-1, 3).cpu().numpy(),
    "vt": main_tex_mesh.textures.verts_uvs_padded().reshape(-1, 2).cpu().numpy(),
    "ft": main_tex_mesh.textures.faces_uvs_padded().reshape(-1, 3).cpu().numpy(),
    "maps": main_tex_mesh.textures.maps_padded().squeeze(0).cpu().numpy(),
}

tex_mask = mask_faces_in_texture(
    room_structure_mesh_dict["v"],
    room_structure_mesh_dict["f"],
    room_structure_mesh_dict["vt"],
    room_structure_mesh_dict["ft"],
    room_structure_mesh_dict["maps"].shape[:2],
    glctx,
    main_tex_mesh_overlap_faces.cpu().numpy(),
)["exist"]

# data gen
znear = 0.01
zfar = 1e10
render_H = render_W = 512
resolution = [render_H, render_W]
aspect_ratio = 1.0
fov = 40.0
fx = fy = render_W / (2 * np.tan(fov * np.pi / 180. / 2))
cx = render_H / 2
cy = render_W / 2
camera_params_all = {
    "00": {}
}
print("start sampling data")
sampled_data = []
with torch.no_grad():
    main_mesh_sample_data = []
    train_cameras = pipeline.datamanager.train_dataset.cameras
    for camera in tqdm(train_cameras):
        # print("test")
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
        # if inside_faces_cnt >= 0.5 * main_tex_mesh_overlap_faces.shape[0]:
        if inside_faces_cnt >= 0.2 * main_tex_mesh_overlap_faces.shape[0]:
            print("hit")
            mask = torch.zeros(render_H, render_W, device="cuda")
            mask_valid = mask[valid]
            mask_valid[pix_to_face_overlap] = 1.0
            mask[valid] = mask_valid

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

            # main_mesh_sample_data.append({
            #     "mesh_camera": camera_p3d.to("cpu"),
            #     "rgb": image[..., :3].reshape(render_H, render_W, 3).cpu(),
            #     "mask": (mask > 0.).cpu(),
            #     "depth": depth.cpu(),
            #     "gs_camera": target_camera,
            # })

            sampled_data.append({
                "znear": znear,
                "zfar": zfar,
                "W": render_W,
                "H": render_H,
                "fov": fov,
                "R": R,
                "tvec": tvec,
            })

sampled_data = [elem for idx, elem in enumerate(sampled_data) if idx in np.linspace(0, len(sampled_data)-1, 10).astype(np.int32)]
for idx, elem in enumerate(sampled_data):
    camera_params_all["00"][f"{idx + 1:0>2d}"] = elem


room_structure_mesh = main_tex_mesh

Faces = namedtuple("Faces", "verts_idx, textures_idx")
room_structure_mesh_faces = Faces(
    verts_idx=torch.from_numpy(room_structure_mesh_dict["f"]).cuda(),
    textures_idx=torch.from_numpy(room_structure_mesh_dict["ft"]).cuda()
)

init_texture = Image.fromarray(np.clip(room_structure_mesh_dict["maps"] * 255., 0, 255).astype(np.uint8))


save_dir = log_dir
os.makedirs(log_dir, exist_ok=True)
# camera_params_path = os.path.join(lama_prepare_dir, "cameras.json")
#
# with open(camera_params_path, 'r') as f:
#     camera_params_all = json.load(f)


DEVICE = "cuda"
image_size = 512
render_simple_factor = 12
fragment_k = 1
uv_size = 2048
smooth_mask = False
view_threshold = 0.2


split_cnt = len(camera_params_all.keys())
for split_i in range(split_cnt):
    camera_params_split_samples = camera_params_all[f"{split_i:0>2d}"]
    sample_cnt = len(camera_params_split_samples.keys())

    # tex_mask = np.array(Image.open(os.path.join(lama_prepare_dir, f"tex_mask_{split_i:0>2d}.png")), dtype=np.float32) / 255.
    exist_texture = torch.from_numpy(1 - tex_mask).to(DEVICE)
    view_punishments = [1 for _ in range(sample_cnt)]

    camera_params_list = [camera_params_split_samples[f"{ci+1:0>2d}"] for ci in range(sample_cnt)]

    output_dir = os.path.join(log_dir, f"{split_i:0>2d}")

    generate_dir = os.path.join(output_dir, "generate")
    os.makedirs(generate_dir, exist_ok=True)

    update_dir = os.path.join(output_dir, "update")
    os.makedirs(update_dir, exist_ok=True)

    sample_dir = os.path.join(output_dir, "sample")
    os.makedirs(sample_dir, exist_ok=True)

    init_image_dir = os.path.join(generate_dir, "rendering")
    os.makedirs(init_image_dir, exist_ok=True)

    normal_map_dir = os.path.join(generate_dir, "normal")
    os.makedirs(normal_map_dir, exist_ok=True)

    mask_image_dir = os.path.join(generate_dir, "mask")
    os.makedirs(mask_image_dir, exist_ok=True)

    depth_map_dir = os.path.join(generate_dir, "depth")
    os.makedirs(depth_map_dir, exist_ok=True)

    similarity_map_dir = os.path.join(generate_dir, "similarity")
    os.makedirs(similarity_map_dir, exist_ok=True)

    inpainted_image_dir = os.path.join(generate_dir, "inpainted")
    os.makedirs(inpainted_image_dir, exist_ok=True)

    mesh_dir = os.path.join(generate_dir, "mesh")
    os.makedirs(mesh_dir, exist_ok=True)

    interm_dir = os.path.join(generate_dir, "intermediate")
    os.makedirs(interm_dir, exist_ok=True)

    pre_similarity_texture_cache = build_similarity_texture_cache_for_all_views(
        room_structure_mesh, room_structure_mesh_faces,
        room_structure_mesh.textures.verts_uvs_padded().reshape(-1, 2),
        camera_params_list,
        image_size, image_size * render_simple_factor, uv_size, fragment_k,
        DEVICE
    )

    print("=> start generating texture...")
    start_time = time.time()
    for view_idx in range(sample_cnt):
        print("=> processing view {}...".format(view_idx))
        camera_params = camera_params_list[view_idx]

        camera_params["resolution"] = (camera_params["H"], camera_params["W"])

        # 1.1. render and build masks
        (
            view_score,
            renderer, cameras, fragments,
            init_image, normal_map, depth_map,
            init_images_tensor, normal_maps_tensor, depth_maps_tensor, similarity_tensor,
            keep_mask_image, update_mask_image, generate_mask_image,
            keep_mask_tensor, update_mask_tensor, generate_mask_tensor, all_mask_tensor, quad_mask_tensor,
        ) = render_one_view_and_build_masks(camera_params,
                                            view_idx, view_idx, view_punishments,
                                            # => actual view idx and the sequence idx
                                            pre_similarity_texture_cache, exist_texture,
                                            [room_structure_mesh], room_structure_mesh, room_structure_mesh_faces,
                                            room_structure_mesh.textures.verts_uvs_padded().reshape(-1, 2),
                                            image_size, fragment_k,
                                            init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir,
                                            similarity_map_dir,
                                            DEVICE, save_intermediate=True, smooth_mask=smooth_mask,
                                            view_threshold=view_threshold
                                            )

        # 1.2. generate missing region
        # NOTE first view still gets the mask for consistent ablations
        # do some mask dilation
        generate_mask_image_np = np.array(generate_mask_image).astype(np.float32) / 255.
        generate_mask_image_np = binary_dilation(generate_mask_image_np, iterations=30)
        generate_mask_image = Image.fromarray(np.clip(generate_mask_image_np * 255., 0, 255).astype(np.uint8))

        actual_generate_mask_image = generate_mask_image

        batch = {}
        batch["mask"] = torch.from_numpy(np.array(actual_generate_mask_image, dtype=np.float32) / 255.0).reshape(1, 1, image_size, image_size).cuda()
        batch["image"] = torch.from_numpy(np.array(init_image.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1).reshape(1, 3, image_size, image_size).cuda()

        lama_inpaint_image = model(batch)[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
        generate_image = Image.fromarray(np.clip(lama_inpaint_image * 255., 0, 255).astype(np.uint8))
        generate_image.save(os.path.join(inpainted_image_dir, "{}.png".format(view_idx)))

        # 1.2.2 back-project and create texture
        # NOTE projection mask = generate mask
        init_texture, project_mask_image, exist_texture = backproject_from_image(
            room_structure_mesh, room_structure_mesh_faces,
            room_structure_mesh.textures.verts_uvs_padded().reshape(-1, 2), cameras,
            generate_image, generate_mask_image, generate_mask_image, init_texture, exist_texture,
            image_size * render_simple_factor, uv_size, fragment_k,
            DEVICE
        )

        project_mask_image.save(os.path.join(mask_image_dir, "{}_project.png".format(view_idx)))

        # update the mesh
        room_structure_mesh.textures = TexturesUV(
            maps=transforms.ToTensor()(init_texture)[None, ...].permute(0, 2, 3, 1).to(DEVICE),
            faces_uvs=room_structure_mesh.textures.faces_uvs_padded(),
            verts_uvs=room_structure_mesh.textures.verts_uvs_padded(),
        )

        # 1.2.3. re: render
        # NOTE only the rendered image is needed - masks should be re-used
        composed_mesh = [room_structure_mesh]
        (
            view_score,
            renderer, cameras, fragments,
            init_image, *_,
        ) = render_one_view_and_build_masks(camera_params,
                                            view_idx, view_idx, view_punishments,
                                            # => actual view idx and the sequence idx
                                            pre_similarity_texture_cache, exist_texture,
                                            composed_mesh, room_structure_mesh, room_structure_mesh_faces,
                                            room_structure_mesh.textures.verts_uvs_padded().reshape(-1, 2),
                                            image_size, fragment_k,
                                            init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir,
                                            similarity_map_dir,
                                            DEVICE, save_intermediate=False, smooth_mask=smooth_mask,
                                            view_threshold=view_threshold
                                            )

    save_backproject_obj(
        save_dir, "main_tex_mesh_inpainted.obj",
        room_structure_mesh.verts_packed(),
        room_structure_mesh_faces.verts_idx,
        room_structure_mesh.textures.verts_uvs_padded().reshape(-1, 2),
        room_structure_mesh_faces.textures_idx, init_texture,
        DEVICE
    )

        # # save the intermediate view
        # inter_images_tensor, *_ = render(room_structure_mesh, renderer)
        # inter_image = inter_images_tensor[0].cpu()
        # inter_image = inter_image.permute(2, 0, 1)
        # inter_image = transforms.ToPILImage()(inter_image).convert("RGB")
        # inter_image.save(os.path.join(interm_dir, "{}.png".format(view_idx)))
        #
        # # save texture mask
        # exist_texture_image = exist_texture * 255.
        # exist_texture_image = Image.fromarray(exist_texture_image.cpu().numpy().astype(np.uint8)).convert("L")
        # exist_texture_image.save(os.path.join(mesh_dir, "{}_texture_mask.png".format(view_idx)))

    print("=> total generate time: {} s".format(time.time() - start_time))



