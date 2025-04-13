import os
import torch

import cv2
import random

import numpy as np

from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.neighbors import NearestNeighbors
from scipy.special import softmax

from torchvision import transforms

from pytorch3d.renderer import TexturesUV, FoVPerspectiveCameras
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.utils import cameras_from_opencv_projection

from PIL import Image

from tqdm import tqdm

# customized
import sys
sys.path.append(".")

from .render_helper import init_renderer, render
from .shading_helper import (
    BlendParams,
    init_soft_phong_shader,
    init_flat_texel_shader,
)

from .vis_helper import visualize_quad_mask
from .constants import *

def get_all_4_locations(values_y, values_x):
    y_0 = torch.floor(values_y)
    y_1 = torch.ceil(values_y)
    x_0 = torch.floor(values_x)
    x_1 = torch.ceil(values_x)

    return torch.cat([y_0, y_0, y_1, y_1], 0).long(), torch.cat([x_0, x_1, x_0, x_1], 0).long()


@torch.no_grad()
def render_one_view(mesh,
                    camera_params,
                    image_size, faces_per_pixel,
                    device):
    # render the view
    # c2w = camera_params['c2w']
    # w2c = torch.inverse(c2w)
    # intrinsic = camera_params['intrinsic']
    # resolution = camera_params['resolution']
    #
    # cameras = cameras_from_opencv_projection(
    #     R=w2c[..., :3, :3].reshape(1, 3, 3),
    #     tvec=w2c[..., :3, 3].reshape(1, 3),
    #     camera_matrix=intrinsic.reshape(1, 3, 3),
    #     image_size=torch.from_numpy(np.array(resolution).astype(np.int32)).reshape(1, 2)
    # ).to(device)

    znear = camera_params["znear"]
    zfar = camera_params["zfar"]
    W = camera_params["W"]
    H = camera_params["H"]
    aspect_ratio = W / H
    fov = camera_params["fov"]

    # fx = W / (2 * np.tan(fov * np.pi / 180. / 2))
    # fy = H / (2 * np.tan(fov * np.pi / 180. / 2))
    # cx = W / 2
    # cy = H / 2

    # c2w = camera_params['c2w']
    # c2w[:3, 0:2] *= -1
    # w2c = torch.inverse(c2w).reshape(4, 4)
    # R = w2c[:3, :3].T.reshape(1, 3, 3)
    # tvec = w2c[:3, 3].reshape(1, 3)

    R = torch.tensor(camera_params['R']).float().reshape(1, 3, 3)
    tvec = torch.tensor(camera_params['tvec']).float().reshape(1, 3)

    cameras = FoVPerspectiveCameras(
        znear=znear,
        zfar=zfar,
        aspect_ratio=aspect_ratio,
        fov=fov,
        degrees=True,
        R=R,
        T=tvec,
        device='cuda'
    ).to(device)

    renderer = init_renderer(cameras,
                             shader=init_soft_phong_shader(
                                 camera=cameras,
                                 blend_params=BlendParams(),
                                 device=device),
                             image_size=image_size,
                             faces_per_pixel=faces_per_pixel
                             )

    init_images_tensor, normal_maps_tensor, similarity_tensor, depth_maps_tensor, fragments = render(mesh, renderer)

    return (
        cameras, renderer,
        init_images_tensor, normal_maps_tensor, similarity_tensor, depth_maps_tensor, fragments
    )

@torch.no_grad()
def build_backproject_mask(mesh, faces, verts_uvs,
    cameras, reference_image, faces_per_pixel,
    image_size, uv_size, device):
    # construct pixel UVs
    renderer_scaled = init_renderer(cameras,
        shader=init_soft_phong_shader(
            camera=cameras,
            blend_params=BlendParams(),
            device=device),
        image_size=image_size,
        faces_per_pixel=faces_per_pixel
    )
    fragments_scaled = renderer_scaled.rasterizer(mesh)

    # get UV coordinates for each pixel
    faces_verts_uvs = verts_uvs[faces.textures_idx]

    pixel_uvs = interpolate_face_attributes(
        fragments_scaled.pix_to_face, fragments_scaled.bary_coords, faces_verts_uvs
    )  # NxHsxWsxKx2
    pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(-1, 2)

    texture_locations_y, texture_locations_x = get_all_4_locations(
        (1 - pixel_uvs[:, 1]).reshape(-1) * (uv_size - 1),
        pixel_uvs[:, 0].reshape(-1) * (uv_size - 1)
    )

    K = faces_per_pixel

    texture_values = torch.from_numpy(np.array(reference_image.resize((image_size, image_size)))).float() / 255.
    texture_values = texture_values.to(device).unsqueeze(0).expand([4, -1, -1, -1]).unsqueeze(0).expand([K, -1, -1, -1, -1])

    # texture
    texture_tensor = torch.zeros(uv_size, uv_size, 3).to(device)
    texture_tensor[texture_locations_y, texture_locations_x, :] = texture_values.reshape(-1, 3)

    return texture_tensor[:, :, 0]


@torch.no_grad()
def build_similarity_texture_cache_for_all_views(mesh, faces, verts_uvs,
                                                 camera_params_list,
                                                 image_size, image_size_scaled, uv_size, faces_per_pixel,
                                                 device):
    num_candidate_views = len(camera_params_list)
    similarity_texture_cache = torch.zeros(num_candidate_views, uv_size, uv_size).to(device)

    print("=> building similarity texture cache for all views...")
    for i in tqdm(range(num_candidate_views)):
        cameras, _, _, _, similarity_tensor, _, _ = render_one_view(mesh,
                                                                    camera_params_list[i],
                                                                    image_size, faces_per_pixel, device)

        similarity_texture_cache[i] = build_backproject_mask(mesh, faces, verts_uvs,
                                                             cameras, transforms.ToPILImage()(
                similarity_tensor[0, :, :, 0]).convert("RGB"), faces_per_pixel,
                                                             image_size_scaled, uv_size, device)

    return similarity_texture_cache


@torch.no_grad()
def build_diffusion_mask(mesh_stuff,
    aggr_visible_mask,
    renderer, exist_texture, similarity_texture_cache, target_value, device, image_size,
    smooth_mask=False, view_threshold=0.01):

    mesh, faces, verts_uvs = mesh_stuff
    mask_mesh = mesh.clone() # NOTE in-place operation - DANGER!!!

    # visible mask => the whole region
    exist_texture_expand = exist_texture.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3).to(device)
    mask_mesh.textures = TexturesUV(
        maps=torch.ones_like(exist_texture_expand),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...],
        sampling_mode="nearest"
    )
    # visible_mask_tensor, *_ = render(mask_mesh, renderer)
    visible_mask_tensor, _, similarity_map_tensor, *_ = render(mask_mesh, renderer)
    # faces that are too rotated away from the viewpoint will be treated as invisible
    valid_mask_tensor = (similarity_map_tensor >= view_threshold).float()
    visible_mask_tensor *= valid_mask_tensor
    visible_mask_tensor *= aggr_visible_mask

    # nonexist mask <=> new mask erode
    exist_texture_expand = exist_texture.unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3).to(device)
    mask_mesh.textures = TexturesUV(
        maps=1 - exist_texture_expand,
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...],
        sampling_mode="nearest"
    )
    new_mask_tensor, *_ = render(mask_mesh, renderer)
    new_mask_tensor *= valid_mask_tensor
    new_mask_tensor *= aggr_visible_mask

    # exist mask => visible mask - new mask
    exist_mask_tensor = visible_mask_tensor - new_mask_tensor
    exist_mask_tensor[exist_mask_tensor < 0] = 0 # NOTE dilate can lead to overflow

    # all update mask
    mask_mesh.textures = TexturesUV(
        maps=(
            similarity_texture_cache.argmax(0) == target_value
            # # only consider the views that have already appeared before
            # similarity_texture_cache[0:target_value+1].argmax(0) == target_value
        ).float().unsqueeze(0).unsqueeze(-1).expand(-1, -1, -1, 3).to(device),
        faces_uvs=faces.textures_idx[None, ...],
        verts_uvs=verts_uvs[None, ...],
        sampling_mode="nearest"
    )
    all_update_mask_tensor, *_ = render(mask_mesh, renderer)
    all_update_mask_tensor *= aggr_visible_mask

    # current update mask => intersection between all update mask and exist mask
    update_mask_tensor = exist_mask_tensor * all_update_mask_tensor

    # keep mask => exist mask - update mask
    old_mask_tensor = exist_mask_tensor - update_mask_tensor

    # convert
    new_mask = new_mask_tensor[0].cpu().float().permute(2, 0, 1)
    new_mask = transforms.ToPILImage()(new_mask).convert("L")

    update_mask = update_mask_tensor[0].cpu().float().permute(2, 0, 1)
    update_mask = transforms.ToPILImage()(update_mask).convert("L")

    old_mask = old_mask_tensor[0].cpu().float().permute(2, 0, 1)
    old_mask = transforms.ToPILImage()(old_mask).convert("L")

    exist_mask = exist_mask_tensor[0].cpu().float().permute(2, 0, 1)
    exist_mask = transforms.ToPILImage()(exist_mask).convert("L")

    return new_mask, update_mask, old_mask, exist_mask


def compose_quad_mask(new_mask_image, update_mask_image, old_mask_image, device):
    """
        compose quad mask:
            -> 0: background
            -> 1: old
            -> 2: update
            -> 3: new
    """

    new_mask_tensor = transforms.ToTensor()(new_mask_image).to(device)
    update_mask_tensor = transforms.ToTensor()(update_mask_image).to(device)
    old_mask_tensor = transforms.ToTensor()(old_mask_image).to(device)

    all_mask_tensor = new_mask_tensor + update_mask_tensor + old_mask_tensor

    quad_mask_tensor = torch.zeros_like(all_mask_tensor)
    quad_mask_tensor[old_mask_tensor == 1] = 1
    quad_mask_tensor[update_mask_tensor == 1] = 2
    quad_mask_tensor[new_mask_tensor == 1] = 3

    return old_mask_tensor, update_mask_tensor, new_mask_tensor, all_mask_tensor, quad_mask_tensor

def compute_view_heat(similarity_tensor, quad_mask_tensor):
    num_total_pixels = quad_mask_tensor.reshape(-1).shape[0]
    heat = 0
    for idx in QUAD_WEIGHTS:
        heat += (quad_mask_tensor == idx).sum() * QUAD_WEIGHTS[idx] / num_total_pixels

    return heat


def visualize_quad_mask(mask_image_dir, quad_mask_tensor, view_idx, view_score, device):
    quad_mask_tensor = quad_mask_tensor.unsqueeze(-1).repeat(1, 1, 1, 3)
    quad_mask_image_tensor = torch.zeros_like(quad_mask_tensor)

    for idx in PALETTE:
        selected = quad_mask_tensor[quad_mask_tensor == idx].reshape(-1, 3)
        selected = torch.FloatTensor(PALETTE[idx]).to(device).unsqueeze(0).repeat(selected.shape[0], 1)

        quad_mask_image_tensor[quad_mask_tensor == idx] = selected.reshape(-1)

    quad_mask_image_np = quad_mask_image_tensor[0].cpu().numpy().astype(np.uint8)
    quad_mask_image = Image.fromarray(quad_mask_image_np).convert("RGB")
    quad_mask_image.save(os.path.join(mask_image_dir, "{}_quad_{:.5f}.png".format(view_idx, view_score)))


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

    def get_relative_depth_map(depth_buf, pad_value=10):
        absolute_depth = depth_buf  # B, H, W
        no_depth = torch.logical_or(depth_buf > 1e4, depth_buf < 0)
        valid_depth = torch.logical_not(no_depth)

        depth_min, depth_max = absolute_depth[valid_depth].min(), absolute_depth[
            valid_depth].max()
        target_min, target_max = 50, 255

        depth_value = absolute_depth[valid_depth]
        depth_value = depth_max - depth_value  # reverse values

        depth_value /= (depth_max - depth_min)
        depth_value = depth_value * (target_max - target_min) + target_min

        relative_depth = absolute_depth.clone()
        relative_depth[valid_depth] = depth_value
        relative_depth[no_depth] = pad_value  # not completely black

        return relative_depth


    return {
        "rgb": image,
        "depth": depth_buf,
        "idx": idx_buf,
        "pix_to_face": pix_to_face_list,
        "bary_coords": bary_coords_list,
        "depth_map": get_relative_depth_map(depth_buf),
    }

def backproject_from_image(mesh, faces, verts_uvs, cameras,
    reference_image, new_mask_image, update_mask_image,
    init_texture, exist_texture,
    image_size, uv_size, faces_per_pixel,
    device):
    # construct pixel UVs
    renderer_scaled = init_renderer(cameras,
                                    shader=init_soft_phong_shader(
                                        camera=cameras,
                                        blend_params=BlendParams(),
                                        device=device),
                                    image_size=image_size,
                                    faces_per_pixel=faces_per_pixel
                                    )
    fragments_scaled = renderer_scaled.rasterizer(mesh)

    # get UV coordinates for each pixel
    faces_verts_uvs = verts_uvs[faces.textures_idx]

    pixel_uvs = interpolate_face_attributes(
        fragments_scaled.pix_to_face, fragments_scaled.bary_coords, faces_verts_uvs
    )  # NxHsxWsxKx2
    pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(pixel_uvs.shape[-2], pixel_uvs.shape[1], pixel_uvs.shape[2], 2)

    # the update mask has to be on top of the diffusion mask
    new_mask_image_tensor = transforms.ToTensor()(new_mask_image).to(device).unsqueeze(-1)
    update_mask_image_tensor = transforms.ToTensor()(update_mask_image).to(device).unsqueeze(-1)

    project_mask_image_tensor = torch.logical_or(update_mask_image_tensor, new_mask_image_tensor).float()
    project_mask_image = project_mask_image_tensor * 255.
    project_mask_image = Image.fromarray(project_mask_image[0, :, :, 0].cpu().numpy().astype(np.uint8))

    project_mask_image_scaled = project_mask_image.resize(
        (image_size, image_size),
        Image.Resampling.NEAREST
    )
    project_mask_image_tensor_scaled = transforms.ToTensor()(project_mask_image_scaled).to(device)

    pixel_uvs_masked = pixel_uvs[project_mask_image_tensor_scaled == 1]

    texture_locations_y, texture_locations_x = get_all_4_locations(
        (1 - pixel_uvs_masked[:, 1]).reshape(-1) * (uv_size - 1),
        pixel_uvs_masked[:, 0].reshape(-1) * (uv_size - 1)
    )

    K = pixel_uvs.shape[0]
    project_mask_image_tensor_scaled = project_mask_image_tensor_scaled[:, None, :, :, None].repeat(1, 4, 1, 1, 3)

    texture_values = torch.from_numpy(np.array(reference_image.resize((image_size, image_size))))
    texture_values = texture_values.to(device).unsqueeze(0).expand([4, -1, -1, -1]).unsqueeze(0).expand(
        [K, -1, -1, -1, -1])

    texture_values_masked = texture_values.reshape(-1, 3)[project_mask_image_tensor_scaled.reshape(-1, 3) == 1].reshape(
        -1, 3)

    # texture
    texture_tensor = torch.from_numpy(np.array(init_texture)).to(device)
    texture_tensor[texture_locations_y, texture_locations_x, :] = texture_values_masked

    init_texture = Image.fromarray(texture_tensor.cpu().numpy().astype(np.uint8))

    # update texture cache
    exist_texture[texture_locations_y, texture_locations_x] = 1

    return init_texture, project_mask_image, exist_texture



@torch.no_grad()
def render_one_view_and_build_masks(camera_params,
                                    selected_view_idx, view_idx, view_punishments,
                                    similarity_texture_cache, exist_texture,
                                    composed_mesh, target_mesh, faces, verts_uvs,
                                    image_size, faces_per_pixel,
                                    init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
                                    device, save_intermediate=False, smooth_mask=False, view_threshold=0.01):
    # render the view
    # first: for every mesh in composed_mesh, render rgb, depth and normal for it

    # output = depth_blend(renderer, composed_mesh, resolution)

    init_image_list = []
    normal_map_list = []
    # depth_map_list = []
    for single_mesh in composed_mesh:

        (
            cameras, renderer,
            init_images_tensor, normal_maps_tensor, similarity_tensor, depth_maps_tensor, fragments
        ) = render_one_view(single_mesh,
                            camera_params,
                            image_size, faces_per_pixel,
                            device
                            )

        init_image = init_images_tensor[0].cpu()
        init_image = init_image.permute(2, 0, 1)

        normal_map = normal_maps_tensor[0].cpu()
        normal_map = normal_map.permute(2, 0, 1)

        # depth_map = depth_maps_tensor[0].cpu()

        init_image_list.append(init_image)
        normal_map_list.append(normal_map)
        # depth_map_list.append(depth_map)

    # second, depth blend
    resolution = camera_params['resolution']
    H, W = resolution
    output = depth_blend(renderer, composed_mesh, resolution)
    blend_idx = output["idx"].cpu()
    depth_map = output["depth_map"].cpu()

    for mesh_i in range(len(composed_mesh)):
        if mesh_i == 0:
            init_image = init_image_list[0]
            normal_map = normal_map_list[0]
            # depth_map = depth_map_list[0]
        else:
            init_image[:, blend_idx == mesh_i + 1] = init_image_list[mesh_i][:, blend_idx == mesh_i + 1]
            normal_map[:, blend_idx == mesh_i + 1] = normal_map_list[mesh_i][:, blend_idx == mesh_i + 1]
            # depth_map[blend_idx == mesh_i + 1] = depth_map_list[mesh_i][blend_idx == mesh_i + 1]

    # set output
    init_images_tensor = init_image.permute(1, 2, 0).cuda().unsqueeze(0)
    normal_maps_tensor = normal_map.permute(1, 2, 0).cuda().unsqueeze(0)
    depth_maps_tensor = depth_map.reshape(1, H, W).cuda()

    init_image = transforms.ToPILImage()(init_image).convert("RGB")
    normal_map = transforms.ToPILImage()(normal_map).convert("RGB")
    depth_map = Image.fromarray(depth_map.numpy()).convert("L")

    # finally, render for target mesh and get similarity map
    (
        cameras, renderer,
        _, _, similarity_tensor, _, fragments
    ) = render_one_view(target_mesh,
                        camera_params,
                        image_size, faces_per_pixel,
                        device
                        )

    similarity_map = similarity_tensor[0, :, :, 0].cpu()
    similarity_map = transforms.ToPILImage()(similarity_map).convert("L")



    flat_renderer = init_renderer(cameras,
                                  shader=init_flat_texel_shader(
                                      camera=cameras,
                                      device=device),
                                  image_size=image_size,
                                  faces_per_pixel=faces_per_pixel
                                  )
    if 'mask' not in camera_params:
        aggr_visible_mask = torch.ones((H, W)).cuda().unsqueeze(0).unsqueeze(-1)
    else:
        aggr_visible_mask = torch.from_numpy(camera_params['mask']).cuda().unsqueeze(0).unsqueeze(-1)
    new_mask_image, update_mask_image, old_mask_image, exist_mask_image = build_diffusion_mask(
        (target_mesh, faces, verts_uvs),
        aggr_visible_mask,
        flat_renderer, exist_texture, similarity_texture_cache, selected_view_idx, device, image_size,
        smooth_mask=smooth_mask, view_threshold=view_threshold
    )
    # NOTE the view idx is the absolute idx in the sample space (i.e. `selected_view_idx`)
    # it should match with `similarity_texture_cache`

    (
        old_mask_tensor,
        update_mask_tensor,
        new_mask_tensor,
        all_mask_tensor,
        quad_mask_tensor
    ) = compose_quad_mask(new_mask_image, update_mask_image, old_mask_image, device)

    view_heat = compute_view_heat(similarity_tensor, quad_mask_tensor)
    view_heat *= view_punishments[selected_view_idx]

    # save intermediate results
    if save_intermediate:
        init_image.save(os.path.join(init_image_dir, "{}.png".format(view_idx)))
        normal_map.save(os.path.join(normal_map_dir, "{}.png".format(view_idx)))
        depth_map.save(os.path.join(depth_map_dir, "{}.png".format(view_idx)))
        similarity_map.save(os.path.join(similarity_map_dir, "{}.png".format(view_idx)))

        new_mask_image.save(os.path.join(mask_image_dir, "{}_new.png".format(view_idx)))
        update_mask_image.save(os.path.join(mask_image_dir, "{}_update.png".format(view_idx)))
        old_mask_image.save(os.path.join(mask_image_dir, "{}_old.png".format(view_idx)))
        exist_mask_image.save(os.path.join(mask_image_dir, "{}_exist.png".format(view_idx)))

        visualize_quad_mask(mask_image_dir, quad_mask_tensor, view_idx, view_heat, device)

    return (
        view_heat,
        renderer, cameras, fragments,
        init_image, normal_map, depth_map,
        init_images_tensor, normal_maps_tensor, depth_maps_tensor, similarity_tensor,
        old_mask_image, update_mask_image, new_mask_image,
        old_mask_tensor, update_mask_tensor, new_mask_tensor, all_mask_tensor, quad_mask_tensor
    )


def select_viewpoint(selected_view_ids, view_punishments,
                     mode, camera_params_list, view_idx,
                     similarity_texture_cache, exist_texture,
                     composed_mesh, target_mesh, faces, verts_uvs,
                     image_size, faces_per_pixel,
                     init_image_dir, mask_image_dir, normal_map_dir, depth_map_dir, similarity_map_dir,
                     device
                     ):

    if mode == "sequential":

        num_views = len(camera_params_list)

        camera_params = camera_params_list[view_idx % num_views]

        selected_view_ids.append(view_idx % num_views)

    elif mode == "heuristic":

        selected_view_idx = None
        max_heat = 0

        print("=> selecting next view...")
        view_heat_list = []
        for sample_idx in tqdm(range(len(camera_params_list))):

            view_heat, *_ = render_one_view_and_build_masks(
                camera_params_list[sample_idx],
                sample_idx, sample_idx, view_punishments,
                similarity_texture_cache, exist_texture,
                composed_mesh, target_mesh, faces, verts_uvs,
                image_size, faces_per_pixel,
                init_image_dir, mask_image_dir, normal_map_dir,
                depth_map_dir, similarity_map_dir,
                device)

            if view_heat > max_heat:
                selected_view_idx = sample_idx
                max_heat = view_heat

            view_heat_list.append(view_heat.item())

        print(view_heat_list)
        print("select view {} with heat {}".format(selected_view_idx, max_heat))

        camera_params = camera_params_list[selected_view_idx]

        selected_view_ids.append(selected_view_idx)

        view_punishments[selected_view_idx] *= 0.01

    elif mode == "random":

        selected_view_idx = random.choice(range(len(camera_params_list)))

        camera_params = camera_params_list[selected_view_idx]

        selected_view_ids.append(selected_view_idx)

    else:
        raise NotImplementedError()

    return camera_params, selected_view_ids, view_punishments