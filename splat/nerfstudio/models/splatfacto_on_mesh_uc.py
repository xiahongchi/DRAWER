# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import trimesh
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
import cv2

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
from gsplat.cuda_legacy._wrapper import num_sh_bases

from pytorch_msssim import SSIM
from torch.nn import Parameter
# from typing_extensions import Literal
from typing import Literal

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

# need following import for background color override
from nerfstudio.model_components import renderers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.misc import torch_compile

from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    axis_angle_to_quaternion,
    quaternion_invert,
    quaternion_multiply,
    euler_angles_to_matrix,
)
from pytorch3d.io import load_objs_as_meshes, save_obj
from collections import namedtuple
from pytorch3d.renderer import TexturesUV, TexturesVertex, RasterizationSettings, MeshRenderer, MeshRendererWithFragments, MeshRasterizer, HardPhongShader, PointLights
from torchvision import transforms
from PIL import Image
import os
import torchvision.transforms.functional as F_V

import torch._dynamo
torch._dynamo.config.suppress_errors = True

def area(triangles):
    # Extract the vertices of the triangles
    A = triangles[:, 0, :]
    B = triangles[:, 1, :]
    C = triangles[:, 2, :]

    # Compute the lengths of the sides of the triangles
    a = torch.norm(B - C, dim=1)
    b = torch.norm(C - A, dim=1)
    c = torch.norm(A - B, dim=1)

    # Compute the semi-perimeter of each triangle
    s = (a + b + c) / 2

    # Compute the area of each triangle using Heron's formula
    area = torch.sqrt(s * (s - a) * (s - b) * (s - c))

    return area

def circumcircle_radius(triangles):
    # Extract the vertices of the triangles
    A = triangles[:, 0, :]
    B = triangles[:, 1, :]
    C = triangles[:, 2, :]

    # Compute the lengths of the sides of the triangles
    a = torch.norm(B - C, dim=1)
    b = torch.norm(C - A, dim=1)
    c = torch.norm(A - B, dim=1)

    # Compute the semi-perimeter of each triangle
    s = (a + b + c) / 2

    # Compute the area of each triangle using Heron's formula
    area = torch.sqrt(s * (s - a) * (s - b) * (s - c))

    # Compute the circumcircle radius
    R = (a * b * c) / (4 * area)

    return R

def incircle_radius(triangles):
    # Extract the vertices of the triangles
    A = triangles[:, 0, :]
    B = triangles[:, 1, :]
    C = triangles[:, 2, :]

    # Compute the lengths of the sides of the triangles
    a = torch.norm(B - C, dim=1)
    b = torch.norm(C - A, dim=1)
    c = torch.norm(A - B, dim=1)

    # Compute the semi-perimeter of each triangle
    s = (a + b + c) / 2

    # Compute the area of each triangle using Heron's formula
    area = torch.sqrt(s * (s - a) * (s - b) * (s - c))

    # Compute the circumcircle radius
    R = area / s

    return R


def compute_min_distance(v, v1, v2, v3):
    """
    Computes the minimum distance from point v to the edges of the triangle formed by v1, v2, and v3.
    Supports batch inference.

    Args:
    v (Tensor): Tensor of shape (batch_size, 3), the point from which distances are computed.
    v1, v2, v3 (Tensor): Tensors of shape (batch_size, 3), representing the vertices of the triangle.

    Returns:
    Tensor: Minimum distance from v to the triangle edges for each batch element.
    """

    def distance_to_edge(v, v_start, v_end):
        # Compute edge vector e and vector from start to v (w)
        e = v_end - v_start
        w = v - v_start

        # Projection scalar t
        e_dot_e = torch.sum(e * e, dim=-1, keepdim=True)
        t = torch.sum(w * e, dim=-1, keepdim=True) / e_dot_e

        # Clamp t to [0, 1] to handle closest point on the edge segment
        t_clamped = torch.clamp(t, 0, 1)

        # Closest point on the edge
        closest_point = v_start + t_clamped * e

        # Compute distance from v to closest point
        distance = torch.norm(v - closest_point, dim=-1)

        return distance

    # Compute distances to each edge
    d1 = distance_to_edge(v, v1, v2)  # Edge from v1 to v2
    d2 = distance_to_edge(v, v2, v3)  # Edge from v2 to v3
    d3 = distance_to_edge(v, v3, v1)  # Edge from v3 to v1

    # Return the minimum distance across the edges
    min_distance = torch.min(torch.stack([d1, d2, d3], dim=-1), dim=-1).values

    return min_distance


def compute_triangle_vertices(a, b, c):
    """
    Compute the coordinates of triangle vertices A, B, and C
    given side lengths a, b, and c in a batched manner.

    Args:
        a: Tensor of side lengths |BC|.
        b: Tensor of side lengths |CA|.
        c: Tensor of side lengths |AB|.

    Returns:
        A (0, 0), B (c, 0), C (x_C, y_C): Coordinates of the triangle vertices.
    """
    # Vertices A and B
    A_x, A_y = torch.zeros_like(a), torch.zeros_like(a)  # A is at (0, 0)
    B_x, B_y = c, torch.zeros_like(c)  # B is at (c, 0)

    # Compute C coordinates
    x_C = (c ** 2 - a ** 2 + b ** 2) / (2 * c)  # x-coordinate of C
    y_C = torch.sqrt(b ** 2 - x_C ** 2)  # y-coordinate of C

    A = torch.stack((A_x, A_y), dim=-1)
    B = torch.stack((B_x, B_y), dim=-1)
    C = torch.stack((x_C, y_C), dim=-1)

    return A, B, C


def barycentric_coordinates(P, A, B, C):
    """
    Compute the barycentric coordinates for a point P relative to triangle ABC.

    Args:
        P_x: Tensor of x-coordinates of point P.
        P_y: Tensor of y-coordinates of point P.
        A, B, C: Tuples of tensors representing the coordinates of points A, B, and C.

    Returns:
        alpha, beta, gamma: Barycentric coordinates of the point P.
    """
    A_x, A_y = A[:, 0], A[:, 1]
    B_x, B_y = B[:, 0], B[:, 1]
    C_x, C_y = C[:, 0], C[:, 1]
    P_x, P_y = P[:, 0], P[:, 1]

    # Compute the denominator (area of the triangle ABC)
    By_Cy = B_y - C_y
    Ax_Cx = A_x - C_x
    Cx_Bx = C_x - B_x
    Ay_Cy = A_y - C_y
    Px_Cx = P_x - C_x
    Py_Cy = P_y - C_y

    denominator = By_Cy * Ax_Cx + Cx_Bx * Ay_Cy

    # Compute the barycentric coordinates
    alpha = (By_Cy * Px_Cx + Cx_Bx * Py_Cy) / denominator
    beta = (-Ay_Cy * Px_Cx + Ax_Cx * Py_Cy) / denominator
    gamma = 1 - alpha - beta

    return torch.stack([alpha, beta, gamma], dim=-1)

def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    height, width = image.shape[:2]
    # print("height: ", height)
    # print("width: ", width)

    scaling_factor = 1.0 / d
    # weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    # return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)
    height_new = int(torch.floor(0.5 + (torch.tensor([float(height)]) * scaling_factor)).to(torch.int64))
    width_new = int(torch.floor(0.5 + (torch.tensor([float(width)]) * scaling_factor)).to(torch.int64))

    # print("height_new: ", height_new)
    # print("width_new: ", width_new)

    image_resized = F_V.resize(image.permute(2, 0, 1), [height_new, width_new])
    image_resized = image_resized.permute(1, 2, 0)

    return image_resized

@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


@dataclass
class SplatfactoOnMeshUCModelConfig(ModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: SplatfactoOnMeshUCModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = True
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""
    mesh_area_to_subdivide: float = 1e-5
    upper_scale: float = 0.5
    unconstrained_scale: bool = True
    unconstrained_elevate: bool = True
    face_flat_coef: float = 0.005
    elevate_coef: float = 0.15
    cone_coef: float = 10.0 * np.pi / 180.0
    acm_lambda: float = 4.0
    mesh_depth_lambda: float = 0.1
    gaussian_save_extra_info_path: str = None


class SplatfactoOnMeshUCModel(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: SplatfactoOnMeshUCModelConfig

    def __init__(
        self,
        *args,
        seed_mesh: Optional[Dict] = None,
        **kwargs,
    ):
        self.seed_mesh = seed_mesh
        assert self.seed_mesh is not None, "splatfacto on mesh needs a mesh to run"
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        means = self.seed_mesh["means"]
        num_points = means.shape[0]

        normal_elevates = torch.nn.Parameter(torch.zeros(num_points).float())

        self.xys_grad_norm = None
        self.max_2Dsize = None
        # distances, _ = self.k_nearest_sklearn(means.data, 3)
        # distances = torch.from_numpy(distances)
        # # find the average of the three nearest neighbors for each point and use that as the scale
        # avg_dist = distances.mean(dim=-1, keepdim=True)
        self.radius = torch.abs(self.seed_mesh["radius"]).reshape(-1, 1).clone().cuda()
        self.xyz_radius = self.radius.clone().repeat(1, 3)
        self.xyz_radius[:, 2] *= self.config.face_flat_coef

        self.mesh_verts = mesh_verts = self.seed_mesh["mesh_verts"].clone().cuda()
        self.mesh_faces = mesh_faces = self.seed_mesh["mesh_faces"].clone().cuda()
        self.mesh_faces_verts = mesh_verts[mesh_faces.reshape(-1)].reshape(-1, 3, 3).cuda()

        v_a = self.mesh_faces_verts[:, 0]
        v_b = self.mesh_faces_verts[:, 1]
        v_c = self.mesh_faces_verts[:, 2]

        self.mesh_triangles_edge_ab = v_b - v_a
        self.mesh_triangles_edge_bc = v_c - v_b
        self.mesh_triangles_edge_ca = v_a - v_c

        self.mesh_triangles_edge_len_a = torch.linalg.norm(self.mesh_triangles_edge_bc, ord=2, dim=-1)
        self.mesh_triangles_edge_len_b = torch.linalg.norm(self.mesh_triangles_edge_ca, ord=2, dim=-1)
        self.mesh_triangles_edge_len_c = torch.linalg.norm(self.mesh_triangles_edge_ab, ord=2, dim=-1)

        self.mesh_triangles_2d_coords_a, self.mesh_triangles_2d_coords_b, self.mesh_triangles_2d_coords_c = compute_triangle_vertices(self.mesh_triangles_edge_len_a, self.mesh_triangles_edge_len_b, self.mesh_triangles_edge_len_c)
        means_2d_coords = (self.mesh_triangles_2d_coords_a + self.mesh_triangles_2d_coords_b + self.mesh_triangles_2d_coords_c) / 3
        means_2d = torch.nn.Parameter(means_2d_coords)


        # scales = torch.nn.Parameter(torch.log(self.xyz_radius.clone() + 1e-10))
        if self.config.unconstrained_scale:
            means_data = torch.mean(self.mesh_faces_verts, dim=1).reshape(-1, 3)
            distances, _ = self.k_nearest_sklearn(means_data, 3)
            distances = torch.from_numpy(distances)
            # find the average of the three nearest neighbors for each point and use that as the scale
            avg_dist = distances.mean(dim=-1, keepdim=True)
            scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        else:
            scales = torch.nn.Parameter(torch.zeros(num_points, 3).float())


        self.normals = self.seed_mesh["normals"].clone().cuda()
        self.normals = torch.nn.functional.normalize(self.normals, dim=-1, p=2)
        self.faces_axis_x = torch.nn.functional.normalize(self.mesh_triangles_edge_ab, dim=-1, p=2)
        self.faces_axis_y = torch.cross(self.normals, self.faces_axis_x, dim=-1).reshape(-1, 3)
        self.faces_axis_y = torch.nn.functional.normalize(self.faces_axis_y, dim=-1, p=2)

        # axis_x_norm_value = torch.sum(self.faces_axis_x * self.faces_axis_x, dim=-1)
        # print("self.faces_axis_x: ", axis_x_norm_value.mean(), axis_x_norm_value.max(), axis_x_norm_value.min())
        # axis_y_norm_value = torch.sum(self.faces_axis_y * self.faces_axis_y, dim=-1)
        # print("self.faces_axis_y: ", axis_y_norm_value.mean(), axis_y_norm_value.max(), axis_y_norm_value.min())
        # assert False

        rot_mat = torch.stack([self.faces_axis_x, self.faces_axis_y, self.normals], dim=2).cuda()
        self.faces_quats = matrix_to_quaternion(rot_mat)



        # quats = torch.nn.Parameter(self.faces_quats.clone())
        quats = torch.nn.Parameter(torch.zeros(num_points, 3).float())
        self.dim_sh = dim_sh = num_sh_bases(self.config.sh_degree)

        shs = torch.zeros((self.seed_mesh["features_dc"].shape[0], dim_sh, 3)).float().cuda()
        if self.config.sh_degree > 0:
            shs[:, 0, :3] = RGB2SH(self.seed_mesh["features_dc"].clone())
            shs[:, 1:, 3:] = 0.0
        else:
            CONSOLE.log("use color only optimization with sigmoid activation")
            shs[:, 0, :3] = torch.logit(self.seed_mesh["features_dc"].clone(), eps=1e-10)
        features_dc = torch.nn.Parameter(shs[:, 0, :])
        self.features_dc_dim = features_dc.shape[-1]
        features_rest = torch.nn.Parameter(shs[:, 1:, :])
        self.features_rest_dims = features_rest.shape[-2:]


        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict(
            {
                "means_2d": means_2d,
                "normal_elevates": normal_elevates,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        self.gaussians_to_mesh_indices = torch.arange(num_points, device="cuda")

        if self.config.gaussian_save_extra_info_path is None:
            self.save_extra_info_path = os.path.join(self.seed_mesh["mesh_dir"], "gaussian_on_mesh_extra_info.pt")
        else:
            self.save_extra_info_path = self.config.gaussian_save_extra_info_path
            os.makedirs(os.path.dirname(self.config.gaussian_save_extra_info_path), exist_ok=True)

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity('alex', normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

        self.articulate_transform = None
        self.visible_gs_indices = None

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        return self.features_dc

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):

        # if self.articulate_means:
        #     return self.external_means.to(self.gauss_params["bary_coords"].device)

        means_2d = self.gauss_params["means_2d"]

        triangles_2d_coords_a = self.mesh_triangles_2d_coords_a[self.gaussians_to_mesh_indices]
        triangles_2d_coords_b = self.mesh_triangles_2d_coords_b[self.gaussians_to_mesh_indices]
        triangles_2d_coords_c = self.mesh_triangles_2d_coords_c[self.gaussians_to_mesh_indices]

        means_2d_bary_coords = barycentric_coordinates(
            means_2d,
            triangles_2d_coords_a,
            triangles_2d_coords_b,
            triangles_2d_coords_c
        )
        means_2d_bary_coords = torch.clip(means_2d_bary_coords, 0, 1)
        means_2d_bary_coords = means_2d_bary_coords / torch.sum(means_2d_bary_coords, dim=-1).reshape(-1, 1)

        means_2d_limited = torch.sum(means_2d_bary_coords.reshape(-1, 3, 1) * torch.stack([triangles_2d_coords_a, triangles_2d_coords_b, triangles_2d_coords_c], dim=1), dim=1)
        means_2d_limited = means_2d_limited.reshape(-1, 2)

        means_2d = means_2d + means_2d_limited.detach() - means_2d.detach()

        means_2d_transformed = torch.sum(
            means_2d.unsqueeze(-1) * torch.stack([self.faces_axis_x[self.gaussians_to_mesh_indices], self.faces_axis_y[self.gaussians_to_mesh_indices]], dim=1), dim=1).reshape(-1, 3)
        means = means_2d_transformed + self.mesh_faces_verts[self.gaussians_to_mesh_indices][:, 0]

        normals = self.normals[self.gaussians_to_mesh_indices]
        radius = self.radius[self.gaussians_to_mesh_indices]

        if self.config.unconstrained_elevate:
            elevate = self.gauss_params["normal_elevates"].reshape(-1, 1)
            maximum_elevate = radius.reshape(-1, 1) * self.config.elevate_coef
            minimum_elevate = -radius.reshape(-1, 1) * self.config.elevate_coef
            elevate_limited = torch.where(elevate < maximum_elevate, elevate, maximum_elevate)
            elevate_limited = torch.where(elevate_limited > minimum_elevate, elevate_limited, minimum_elevate)
            elevate = elevate_limited.detach() + elevate - elevate.detach()
            elevate = normals * elevate

        else:
            normal_elevates = torch.sigmoid(self.gauss_params["normal_elevates"]) - 0.5
            elevate = normals * normal_elevates.reshape(-1, 1) * radius.reshape(-1, 1)

        means += elevate

        if self.articulate_transform is not None:
            transform_indices_list = self.articulate_transform['transform_indices_list']
            transform_matrix_list = self.articulate_transform['transform_matrix_means_list']

            for transform_indices, transform_matrix in zip(transform_indices_list, transform_matrix_list):
                transform_indices = transform_indices.to(means.device)
                transform_matrix = transform_matrix.to(means.device)

                means_to_transform = means[transform_indices]
                means_to_transform = torch.nn.functional.pad(means_to_transform, (0, 1), "constant", 1.0)
                means_to_transform = means_to_transform @ transform_matrix.T
                means_to_transform = means_to_transform[:, :3] / means_to_transform[:, 3:]

                means = means.reshape(-1).masked_scatter(transform_indices.reshape(-1, 1).repeat(1, 3).reshape(-1), means_to_transform.reshape(-1)).reshape(-1, 3)

        return means

    @property
    def scales(self):
        if self.config.unconstrained_scale:
            real_scales = torch.exp(self.gauss_params["scales"])
            xyz_radius = self.xyz_radius[self.gaussians_to_mesh_indices]
            scale_limit = self.config.upper_scale * xyz_radius
            real_scales_limited = torch.where(real_scales <= scale_limit, real_scales, scale_limit)
            real_scales = real_scales_limited.detach() + real_scales - real_scales.detach()
            return torch.log(real_scales + 1e-20)


        else:
            upper_scale = self.config.upper_scale
            local_scales = torch.sigmoid(self.gauss_params["scales"])
            xyz_radius = self.xyz_radius[self.gaussians_to_mesh_indices]
            scales = torch.log(local_scales * xyz_radius * upper_scale + 1e-20)

            return scales

        # return self.gauss_params["scales"]

    @property
    def quats(self):

        thetas = self.gauss_params["quats"][:, 0].reshape(-1, 1)
        xy_rotation_quats = axis_angle_to_quaternion(torch.nn.functional.pad(thetas, (2, 0), "constant", 0.0))

        alphas = self.gauss_params["quats"][:, 1]
        phis = self.gauss_params["quats"][:, 2]

        phis_limited = torch.clip(phis, 0.0, self.config.cone_coef)
        phis = phis + phis_limited.detach() - phis.detach()

        z_rotation_axis = torch.nn.functional.pad(torch.stack([torch.cos(alphas), torch.sin(alphas)], dim=-1), (0, 1), "constant", 0.0)

        z_rotation_quats = axis_angle_to_quaternion(z_rotation_axis * phis.reshape(-1, 1))

        faces_quats = self.faces_quats[self.gaussians_to_mesh_indices]
        quats = quaternion_multiply(faces_quats, quaternion_multiply(z_rotation_quats, xy_rotation_quats))

        if self.articulate_transform is not None:
            transform_indices_list = self.articulate_transform['transform_indices_list']
            transform_matrix_list = self.articulate_transform['transform_matrix_quats_list']

            for transform_indices, transform_matrix in zip(transform_indices_list, transform_matrix_list):
                transform_indices = transform_indices.to(quats.device)
                transform_matrix = transform_matrix.to(quats.device)

                quats_to_transform = quats[transform_indices]
                quats_to_transform_mat = quaternion_to_matrix(quats_to_transform)
                quats_to_transform_mat = transform_matrix @ quats_to_transform_mat
                quats_to_transform = matrix_to_quaternion(quats_to_transform_mat)

                quats = quats.reshape(-1).masked_scatter(transform_indices.reshape(-1, 1).repeat(1, 4).reshape(-1), quats_to_transform.reshape(-1)).reshape(-1, 4)

        return quats

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        if self.visible_gs_indices is not None:
            opacities = self.gauss_params["opacities"]
            opacities_mask = torch.zeros((opacities.shape[0], 1), device="cuda", dtype=torch.float32)
            opacities_mask[self.visible_gs_indices] = 1.0
            opacities = opacities * opacities_mask + torch.logit(torch.tensor([1e-6]).reshape(1, 1)).to("cuda").float() * (1 - opacities_mask)
            return opacities
        else:
            return self.gauss_params["opacities"]

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000

        newp = dict["gauss_params.scales"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))

        if os.path.exists(self.save_extra_info_path):
            print("extra info is loaded from: ", self.save_extra_info_path)
            extra_info = torch.load(self.save_extra_info_path)
            self.gaussians_to_mesh_indices = extra_info["gaussians_to_mesh_indices"].cuda()
        super().load_state_dict(dict, **kwargs)

    def append_from_mesh(self, texture_mesh_path):

        mesh = load_objs_as_meshes([texture_mesh_path], device='cpu')
        mesh_verts = mesh.verts_packed().clone().reshape(-1, 3)
        mesh_faces = mesh.faces_packed().clone().reshape(-1, 3)

        normals = mesh.faces_normals_packed().clone().reshape(-1, 3)

        N_Gaussians = mesh_faces.shape[0]
        triangles = mesh_verts[mesh_faces.reshape(-1)].reshape(-1, 3, 3)
        means = torch.mean(triangles, dim=1)
        radius = circumcircle_radius(triangles)

        pix_to_face = torch.arange(N_Gaussians)
        bary_coords = torch.ones(N_Gaussians, 3) / 3

        Mesh_Fragments = namedtuple("Mesh_Fragments", ['pix_to_face', 'bary_coords'])
        mesh_fragments = Mesh_Fragments(
            pix_to_face=pix_to_face.reshape(1, 1, N_Gaussians, 1),
            bary_coords=bary_coords.reshape(1, 1, N_Gaussians, 1, 3),
        )

        area_to_subdivide = self.config.mesh_area_to_subdivide
        n_subdivision_max_iter = 4
        subdivision_iter = 0
        while True:
            areas = area(triangles)
            if torch.all(areas <= area_to_subdivide) or subdivision_iter >= n_subdivision_max_iter:
                break
            face_to_subdivide = (areas > area_to_subdivide)

            mesh_faces_subdivided = mesh_faces[face_to_subdivide]

            triangles_subdivided = mesh_verts[mesh_faces_subdivided.reshape(-1)].reshape(-1, 3, 3)

            mesh_verts_added = torch.cat([
                (triangles_subdivided[:, 0] + triangles_subdivided[:, 1]) / 2,
                (triangles_subdivided[:, 0] + triangles_subdivided[:, 2]) / 2,
                (triangles_subdivided[:, 1] + triangles_subdivided[:, 2]) / 2,
            ], dim=0)

            num_verts_before = mesh_verts.shape[0]
            num_subdivided_faces = triangles_subdivided.shape[0]

            verts_a_idxs = num_verts_before + torch.arange(num_subdivided_faces)
            verts_b_idxs = num_verts_before + num_subdivided_faces + torch.arange(num_subdivided_faces)
            verts_c_idxs = num_verts_before + num_subdivided_faces * 2 + torch.arange(num_subdivided_faces)

            verts_0_idxs = mesh_faces_subdivided[:, 0]
            verts_1_idxs = mesh_faces_subdivided[:, 1]
            verts_2_idxs = mesh_faces_subdivided[:, 2]

            faces_0ab = torch.stack([verts_0_idxs, verts_a_idxs, verts_b_idxs], dim=-1)
            faces_1ca = torch.stack([verts_1_idxs, verts_c_idxs, verts_a_idxs], dim=-1)
            faces_2bc = torch.stack([verts_2_idxs, verts_b_idxs, verts_c_idxs], dim=-1)
            faces_acb = torch.stack([verts_a_idxs, verts_c_idxs, verts_b_idxs], dim=-1)

            bary_coords_to_subdivide = bary_coords[face_to_subdivide]
            weight_0 = bary_coords_to_subdivide[:, 0]
            weight_1 = bary_coords_to_subdivide[:, 1]
            weight_2 = bary_coords_to_subdivide[:, 2]
            bary_coords_0ab = torch.stack([weight_0 + 0.5 * (weight_1 + weight_2), 0.5 * weight_1, 0.5 * weight_2], dim=-1)
            bary_coords_1ca = torch.stack([0.5 * weight_0, weight_1 + 0.5 * (weight_0 + weight_2), 0.5 * weight_2], dim=-1)
            bary_coords_2bc = torch.stack([0.5 * weight_0, 0.5 * weight_1, weight_2 + 0.5 * (weight_0 + weight_1)], dim=-1)

            mesh_faces[face_to_subdivide] = faces_acb
            mesh_faces = torch.cat([
                mesh_faces,
                faces_0ab, faces_1ca, faces_2bc
            ], dim=0)

            pix_to_face = torch.cat([pix_to_face] + [pix_to_face[face_to_subdivide]] * 3, dim=0)
            bary_coords = torch.cat([
                bary_coords,
                bary_coords_0ab, bary_coords_1ca, bary_coords_2bc
            ], dim=0)

            mesh_verts = torch.cat([
                mesh_verts,
                mesh_verts_added
            ], dim=0)

            triangles = mesh_verts[mesh_faces.reshape(-1)].reshape(-1, 3, 3)

            radius = circumcircle_radius(triangles)
            N_Gaussians = mesh_faces.shape[0]
            means = torch.mean(triangles, dim=1)
            normals = torch.cat([normals] + [normals[face_to_subdivide]] * 3, dim=0)
            # features_dc = torch.cat([features_dc] + [features_dc[face_to_subdivide]] * 3, dim=0)

            subdivision_iter += 1

        Mesh_Fragments = namedtuple("Mesh_Fragments", ['pix_to_face', 'bary_coords'])
        mesh_fragments = Mesh_Fragments(
            pix_to_face=pix_to_face.reshape(1, 1, N_Gaussians, 1),
            bary_coords=bary_coords.reshape(1, 1, N_Gaussians, 1, 3),
        )
        features_dc = mesh.textures.sample_textures(mesh_fragments).reshape(N_Gaussians, 3)

        append_num_gaussians = mesh_faces.shape[0]
        num_existing_gaussians = self.gauss_params["means_2d"].shape[0]

        original_face_cnt = self.mesh_faces.shape[0]
        append_indices = torch.arange(append_num_gaussians) + original_face_cnt
        append_gaussians_indices = torch.arange(append_num_gaussians) + num_existing_gaussians

        radius = torch.abs(radius).reshape(-1, 1).clone().cuda()
        mesh_verts = mesh_verts.cuda()
        first_edge = mesh_verts[mesh_faces[:, :2].reshape(-1)].reshape(-1, 2, 3)
        first_edge_vec = (first_edge[:, 0] - first_edge[:, 1]).reshape(-1, 3)
        bottom_length = torch.sum(first_edge_vec ** 2, dim=-1) ** 0.5
        cross_height = (area(triangles).cuda() * 2) / bottom_length
        # xyz_radius = torch.stack(
        #     [bottom_length.reshape(-1) * 3.0, cross_height.reshape(-1) * 3.0, radius.reshape(-1) * 0.05], dim=-1)
        # radius *= 50
        xyz_radius = radius.clone().repeat(1, 3)
        xyz_radius[:, 2] *= self.config.face_flat_coef

        self.radius = torch.cat([self.radius, radius.clone().reshape(-1, 1).to(self.radius.device)], dim=0)
        self.xyz_radius = torch.cat([self.xyz_radius, xyz_radius.clone().to(self.xyz_radius.device)], dim=0)

        original_num_verts = self.mesh_verts.shape[0]

        self.mesh_verts = torch.cat([self.mesh_verts, mesh_verts.to(self.mesh_verts.device)], dim=0)
        self.mesh_faces = torch.cat([self.mesh_faces, mesh_faces.to(self.mesh_faces.device) + original_num_verts],
                                    dim=0)
        self.mesh_faces_verts = self.mesh_verts[self.mesh_faces.reshape(-1)].reshape(-1, 3, 3).cuda()
        self.normals = torch.cat([self.normals, normals.to(self.normals.device)])

        mesh_faces_verts = mesh_verts[mesh_faces.reshape(-1)].reshape(-1, 3, 3).cuda()

        v_a = mesh_faces_verts[:, 0]
        v_b = mesh_faces_verts[:, 1]
        v_c = mesh_faces_verts[:, 2]

        mesh_triangles_edge_ab = v_b - v_a
        mesh_triangles_edge_bc = v_c - v_b
        mesh_triangles_edge_ca = v_a - v_c

        mesh_triangles_edge_len_a = torch.linalg.norm(mesh_triangles_edge_bc, ord=2, dim=-1)
        mesh_triangles_edge_len_b = torch.linalg.norm(mesh_triangles_edge_ca, ord=2, dim=-1)
        mesh_triangles_edge_len_c = torch.linalg.norm(mesh_triangles_edge_ab, ord=2, dim=-1)

        mesh_triangles_2d_coords_a, mesh_triangles_2d_coords_b, mesh_triangles_2d_coords_c = compute_triangle_vertices(
            mesh_triangles_edge_len_a, mesh_triangles_edge_len_b, mesh_triangles_edge_len_c)
        means_2d_coords = (mesh_triangles_2d_coords_a + mesh_triangles_2d_coords_b + mesh_triangles_2d_coords_c) / 3
        means_2d = means_2d_coords

        self.mesh_triangles_edge_ab = torch.cat([self.mesh_triangles_edge_ab, mesh_triangles_edge_ab.to(self.mesh_triangles_edge_ab.device)], dim=0)
        self.mesh_triangles_edge_bc = torch.cat([self.mesh_triangles_edge_bc, mesh_triangles_edge_bc.to(self.mesh_triangles_edge_bc.device)], dim=0)
        self.mesh_triangles_edge_ca = torch.cat([self.mesh_triangles_edge_ca, mesh_triangles_edge_ca.to(self.mesh_triangles_edge_ca.device)], dim=0)

        self.mesh_triangles_edge_len_a = torch.cat([self.mesh_triangles_edge_len_a, mesh_triangles_edge_len_a.to(self.mesh_triangles_edge_len_a.device)], dim=0)
        self.mesh_triangles_edge_len_b = torch.cat([self.mesh_triangles_edge_len_b, mesh_triangles_edge_len_b.to(self.mesh_triangles_edge_len_b.device)], dim=0)
        self.mesh_triangles_edge_len_c = torch.cat([self.mesh_triangles_edge_len_c, mesh_triangles_edge_len_c.to(self.mesh_triangles_edge_len_c.device)], dim=0)

        self.mesh_triangles_2d_coords_a = torch.cat([self.mesh_triangles_2d_coords_a, mesh_triangles_2d_coords_a.to(self.mesh_triangles_2d_coords_a.device)], dim=0)
        self.mesh_triangles_2d_coords_b = torch.cat([self.mesh_triangles_2d_coords_b, mesh_triangles_2d_coords_b.to(self.mesh_triangles_2d_coords_b.device)], dim=0)
        self.mesh_triangles_2d_coords_c = torch.cat([self.mesh_triangles_2d_coords_c, mesh_triangles_2d_coords_c.to(self.mesh_triangles_2d_coords_c.device)], dim=0)


        if self.config.unconstrained_scale:
            means_data = torch.mean(mesh_faces_verts, dim=1).reshape(-1, 3)
            distances, _ = self.k_nearest_sklearn(means_data, 3)
            distances = torch.from_numpy(distances)
            # find the average of the three nearest neighbors for each point and use that as the scale
            avg_dist = distances.mean(dim=-1, keepdim=True)
            scales = torch.log(avg_dist.repeat(1, 3))
        else:
            scales = torch.zeros(append_num_gaussians, 3).float()

        normals = torch.nn.functional.normalize(normals, dim=-1, p=2).cuda()
        faces_axis_x = torch.nn.functional.normalize(mesh_triangles_edge_ab, dim=-1, p=2).cuda()
        faces_axis_y = torch.cross(normals, faces_axis_x, dim=-1).reshape(-1, 3)
        faces_axis_y = torch.nn.functional.normalize(faces_axis_y, dim=-1, p=2)

        self.faces_axis_x = torch.cat([self.faces_axis_x, faces_axis_x.to(self.faces_axis_x.device)], dim=0)
        self.faces_axis_y = torch.cat([self.faces_axis_y, faces_axis_y.to(self.faces_axis_y.device)], dim=0)

        rot_mat = torch.stack([faces_axis_x, faces_axis_y, normals], dim=2).cuda()
        faces_quats = matrix_to_quaternion(rot_mat)
        self.faces_quats = torch.cat([self.faces_quats, faces_quats.to(self.faces_quats.device)], dim=0)

        quats = torch.nn.Parameter(torch.zeros(append_num_gaussians, 3).float())

        self.gaussians_to_mesh_indices = torch.cat([self.gaussians_to_mesh_indices, torch.arange(append_num_gaussians, device="cuda") + original_face_cnt], dim=0)

        self.gauss_params["means_2d"] = torch.nn.Parameter(torch.cat([
            self.gauss_params["means_2d"].detach(),
            means_2d.float().cuda()
        ], dim=0))

        self.gauss_params["normal_elevates"] = torch.nn.Parameter(torch.cat([
            self.gauss_params["normal_elevates"].detach(),
            torch.zeros(append_num_gaussians).float().cuda()
        ], dim=0))

        self.gauss_params["scales"] = torch.nn.Parameter(torch.cat([
            self.gauss_params["scales"].detach(),
            scales.float().cuda()
        ], dim=0))

        self.gauss_params["quats"] = torch.nn.Parameter(torch.cat([
            self.gauss_params["quats"].detach(),
            quats.float().cuda()
        ], dim=0))

        self.gauss_params["features_dc"] = torch.nn.Parameter(torch.cat([
            self.gauss_params["features_dc"].detach(),
            RGB2SH(features_dc.reshape(-1, 3)).cuda()
        ], dim=0))

        self.gauss_params["features_rest"] = torch.nn.Parameter(torch.cat([
            self.gauss_params["features_rest"].detach(),
            torch.zeros(append_num_gaussians, self.dim_sh - 1, 3).float().cuda()
        ], dim=0))

        self.gauss_params["opacities"] = torch.nn.Parameter(torch.cat([
            self.gauss_params["opacities"].detach(),
            torch.logit(0.1 * torch.ones(append_num_gaussians, 1)).float().cuda()
        ], dim=0))

        return append_indices, append_gaussians_indices


    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.config.stop_split_at:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            grads = self.xys.absgrad[0][visible_mask].norm(dim=-1)  # type: ignore
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=self.device, dtype=torch.float32)
                self.vis_counts = torch.ones(self.num_points, device=self.device, dtype=torch.float32)

            assert self.vis_counts is not None
            self.vis_counts[visible_mask] += 1
            self.xys_grad_norm[visible_mask] += grads

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                self.step < self.config.stop_split_at
                and self.step % reset_interval > min(self.num_train_data, 1000) + self.config.refine_every
            )
            if do_densification:
                # then we densify
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.config.n_split_samples
                split_params = self.split_gaussians(splits, nsamps)
                splits_idxs = self.gaussians_to_mesh_indices[splits].repeat(nsamps)

                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                dups &= high_grads
                dups_idxs = self.gaussians_to_mesh_indices[dups]

                dup_params = self.dup_gaussians(dups)

                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    )
                    # print(name, self.gauss_params[name].shape)
                self.gaussians_to_mesh_indices = torch.cat([
                    self.gaussians_to_mesh_indices,
                    splits_idxs,
                    dups_idxs
                ], dim=0)
                # print("gaussians_to_mesh_indices: ", self.gaussians_to_mesh_indices.shape)
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales"][:, 0]),
                        torch.zeros_like(dup_params["scales"][:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )

                deleted_mask = self.cull_gaussians(splits_mask)
            elif self.step >= self.config.stop_split_at and self.config.continue_cull_post_densification:
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None

            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)


            if self.step < self.config.stop_split_at and self.step % reset_interval == self.config.refine_every:
                # Reset value is set to be twice of the cull_alpha_thresh
                reset_value = self.config.cull_alpha_thresh * 2.0
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacities"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                if self.max_2Dsize is not None:
                    toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()

        faces_gaussian_cnt = torch.bincount(self.gaussians_to_mesh_indices)
        assert faces_gaussian_cnt.shape[0] == self.mesh_faces.shape[0]
        assert (not torch.any(faces_gaussian_cnt == 0)), torch.count_nonzero(faces_gaussian_cnt == 0)

        gaussians_to_mesh_indices_after_culls = self.gaussians_to_mesh_indices[~culls]
        faces_gaussian_cnt_after_culls = torch.bincount(gaussians_to_mesh_indices_after_culls)
        if faces_gaussian_cnt_after_culls.shape[0] < self.mesh_faces.shape[0]:
            faces_gaussian_cnt_after_culls = torch.nn.functional.pad(faces_gaussian_cnt_after_culls, (0, self.mesh_faces.shape[0] - faces_gaussian_cnt_after_culls.shape[0]), "constant", 0)

        pruned_faces_gs = torch.isin(self.gaussians_to_mesh_indices, torch.arange(self.mesh_faces.shape[0], device="cuda")[faces_gaussian_cnt_after_culls == 0])
        culls = torch.logical_and(culls, ~pruned_faces_gs)

        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])
            # print(name, self.gauss_params[name].shape)
        self.gaussians_to_mesh_indices = self.gaussians_to_mesh_indices[~culls]
        # print("self.gaussians_to_mesh_indices: ", self.gaussians_to_mesh_indices.shape)
        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")

        # def sample_pts_in_triangle(num_samples):
        #     uv = np.random.rand(num_samples, 2)
        #     cond = np.sum(uv, axis=-1) > 1
        #     uv[cond] = 1 - uv[cond]
        #     uvw = np.concatenate([uv, 1 - np.sum(uv, axis=-1, keepdims=True)], axis=-1)
        #     return torch.from_numpy(uvw).float()
        #
        # new_bary_coords = torch.log(sample_pts_in_triangle(n_splits * samps).reshape(samps * n_splits, 3).cuda() + 1e-20)
        # new_normal_elevates = self.gauss_params["normal_elevates"][split_mask].repeat(samps)

        # start with sample new scales
        size_fac = 1.6
        if self.config.unconstrained_scale:
            real_scales = torch.exp(self.scales[split_mask])
            new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
            self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)

        else:
            upper_scale = self.config.upper_scale
            real_scales = torch.exp(self.scales[split_mask])
            new_real_scales = real_scales / size_fac
            xyz_radius_split = self.xyz_radius[self.gaussians_to_mesh_indices][split_mask]
            new_scales = torch.logit((new_real_scales / upper_scale) / xyz_radius_split).repeat(samps, 1)

        # then sample new means
        real_scale_min_xy = torch.min(real_scales[:, :2], dim=-1)[0].reshape(-1, 1).repeat(samps, 1)
        radius_splits = torch.abs(torch.randn(n_splits * samps, 1, device="cuda", dtype=torch.float32)) * 0.5
        theta_splits = torch.rand(n_splits * samps, 1, device="cuda", dtype=torch.float32) * torch.pi * 2.0

        # bary_coords = torch.nn.functional.softmax(self.gauss_params["bary_coords"], dim=-1)[split_mask]
        mesh_faces_verts = self.mesh_faces_verts[self.gaussians_to_mesh_indices][split_mask]
        # # normals = self.normals[self.gaussians_to_mesh_indices][split_mask]
        # # radius = self.radius[self.gaussians_to_mesh_indices][split_mask]
        #
        # v_proj_splits = torch.sum(mesh_faces_verts * bary_coords.reshape(-1, 3, 1), dim=1).reshape(-1, 3)

        means_2d = self.gauss_params["means_2d"][split_mask]
        faces_axis_x = self.faces_axis_x[self.gaussians_to_mesh_indices][split_mask]
        faces_axis_y = self.faces_axis_y[self.gaussians_to_mesh_indices][split_mask]
        means_2d_transformed = torch.sum(
            means_2d.unsqueeze(-1) * torch.stack([faces_axis_x, faces_axis_y], dim=1), dim=1).reshape(-1, 3)
        v_proj_splits = means_2d_transformed + mesh_faces_verts[:, 0]

        largest_radius_splits = compute_min_distance(v_proj_splits, mesh_faces_verts[:, 0], mesh_faces_verts[:, 1], mesh_faces_verts[:, 2])
        largest_radius_splits = largest_radius_splits.unsqueeze(-1).repeat(samps, 1)
        v_proj_splits = v_proj_splits.repeat(samps, 1)

        x_axis_splits = faces_axis_x.repeat(samps, 1)
        y_axis_splits = faces_axis_y.repeat(samps, 1)

        real_radius_splits = radius_splits * real_scale_min_xy
        real_radius_splits = torch.where(real_radius_splits > largest_radius_splits * 0.99, largest_radius_splits * 0.99, real_radius_splits)

        direction_splits = torch.cos(theta_splits) * x_axis_splits + torch.sin(theta_splits) * y_axis_splits

        # norm_value = torch.sum(direction_splits * direction_splits, dim=-1)
        # print("direction_splits: ", norm_value.mean(), norm_value.max(), norm_value.min())
        # assert False

        v_sample_proj_splits = v_proj_splits + real_radius_splits * direction_splits
        v_sample_proj_splits_2d_coords = v_sample_proj_splits - mesh_faces_verts[:, 0].repeat(samps, 1)

        new_means_2d = torch.stack([
            torch.sum(v_sample_proj_splits_2d_coords * x_axis_splits, dim=-1).reshape(-1),
            torch.sum(v_sample_proj_splits_2d_coords * y_axis_splits, dim=-1).reshape(-1),
        ], dim=-1)

        new_normal_elevates = self.gauss_params["normal_elevates"][split_mask].repeat(samps)

        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 5, sample new quats
        new_quats = self.gauss_params["quats"][split_mask].repeat(samps, 1)
        out = {
            "means_2d": new_means_2d,
            "normal_elevates": new_normal_elevates,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        return new_dups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        torch.save({
            "gaussians_to_mesh_indices": self.gaussians_to_mesh_indices.cpu()
        }, self.save_extra_info_path)
        print("extra info is saved to: ", self.save_extra_info_path)
        return {
            name: [self.gauss_params[name]]
            for name in ["means_2d", "normal_elevates", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        self.camera_optimizer.get_param_groups(param_groups=gps)
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        # get the background color
        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac, "round")
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)

        camera.rescale_output_resolution(camera_scale_fac, "round")  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )
        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"]  # [1, N, 2]
        self.radii = info["radii"][0]  # [N]
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        # print("predicted_rgb: ", predicted_rgb.shape)
        # print("gt_rgb: ", gt_rgb.shape)
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        accumulation = outputs["accumulation"]
        loss_acm = torch.mean(torch.nn.functional.relu(0.95 - accumulation)) * self.config.acm_lambda
        # loss_acm = torch.mean(torch.nn.functional.relu(0.8 - accumulation)) * 0.5

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        if "mesh_depth" in batch:
            if batch["mesh_depth"].shape[0] != gt_img.shape[0] and batch["mesh_depth"].shape[1] != gt_img.shape[1]:
                downscale_factor = int(torch.floor(torch.tensor([float(batch["mesh_depth"].shape[0]) / float(gt_img.shape[0]) + 0.5])))
            else:
                downscale_factor = 1
            # print("0 mesh_depth: ", batch["mesh_depth"].shape)
            # print("0 gt_img: ", gt_img.shape)
            # print("downscale_factor: ", downscale_factor)
            mesh_depth = batch["mesh_depth"].unsqueeze(-1)
            if downscale_factor > 1:
                mesh_depth = resize_image(mesh_depth, downscale_factor)
            mesh_depth = mesh_depth.to(self.device)
            # print("1 mesh_depth: ", mesh_depth.shape)
            # print("1 gt_img: ", gt_img.shape)
            assert mesh_depth.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            predicted_depth = outputs["depth"]
            L1_depth = torch.nn.functional.relu(torch.abs(mesh_depth.reshape(predicted_depth.shape) - predicted_depth) - 0.).mean() * self.config.mesh_depth_lambda
        else:
            L1_depth = 0

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales[:, :2])
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        # gamma = 1
        # scale_exp = torch.exp(self.scales)
        # face_scale_loss = torch.clip(scale_exp - gamma * self.xyz_radius.to(scale_exp.device), min=0)
        # valid = torch.logical_not(torch.isnan(face_scale_loss))
        # face_scale_loss = 10 * face_scale_loss[valid].mean()

        # faces_quats_normalized = torch.nn.functional.normalize(self.faces_quats.to(self.quats.device))
        # quats_normalized = torch.nn.functional.normalize(self.quats)
        # face_quat_loss = (1 - torch.abs(torch.sum(faces_quats_normalized * quats_normalized, dim=-1))).mean()

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss + loss_acm + L1_depth,
            "scale_reg": scale_reg,
            # "face_scale_loss": face_scale_loss,
            # "face_quat_loss": face_quat_loss,
        }

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]
        predicted_depth = outputs["depth"]
        accumulation = outputs["accumulation"]

        depth_im = cv2.applyColorMap(np.clip((predicted_depth / predicted_depth.max()).detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        depth_im = torch.from_numpy(np.array(depth_im, dtype=np.float32)[..., ::-1] / 255.)

        accu_im = cv2.applyColorMap(np.clip(accumulation.cpu().numpy() * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        accu_im = torch.from_numpy(np.array(accu_im, dtype=np.float32)[..., ::-1] / 255.)

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb, "depth_im": depth_im, "accu_im": accu_im}

        if "mesh_depth" in batch:
            mesh_depth_im = cv2.applyColorMap(np.clip((batch["mesh_depth"].to(predicted_depth.device) / predicted_depth.max()).detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
            mesh_depth_im = torch.from_numpy(np.array(mesh_depth_im, dtype=np.float32)[..., ::-1] / 255.)
            images_dict["mesh_depth_im"] = mesh_depth_im


        return metrics_dict, images_dict

    def export_splatfacto_on_mesh(self):
        export_dict = {
            "means_2d": self.gauss_params["means_2d"].detach().cpu(),
            "normal_elevates": self.gauss_params["normal_elevates"].detach().cpu(),
            "scales": self.gauss_params["scales"].detach().cpu(),
            "quats": self.gauss_params["quats"].detach().cpu(),
            "features_dc": self.gauss_params["features_dc"].detach().cpu(),
            "features_rest": self.gauss_params["features_rest"].detach().cpu(),
            "opacities": self.gauss_params["opacities"].detach().cpu(),

            "mesh_faces_verts": self.mesh_faces_verts.cpu(),
            "normals": self.normals.cpu(),
            "radius": self.radius.cpu(),
            "xyz_radius": self.xyz_radius.cpu(),
            "faces_quats": self.faces_quats.cpu(),
            "mesh_verts": self.mesh_verts.cpu(),
            "mesh_faces": self.mesh_faces.cpu(),

            "faces_axis_x": self.faces_axis_x.cpu(),
            "faces_axis_y": self.faces_axis_y.cpu(),

            "mesh_triangles_edge_ab": self.mesh_triangles_edge_ab.cpu(),
            "mesh_triangles_edge_bc": self.mesh_triangles_edge_bc.cpu(),
            "mesh_triangles_edge_ca": self.mesh_triangles_edge_ca.cpu(),

            "mesh_triangles_edge_len_a": self.mesh_triangles_edge_len_a.cpu(),
            "mesh_triangles_edge_len_b": self.mesh_triangles_edge_len_b.cpu(),
            "mesh_triangles_edge_len_c": self.mesh_triangles_edge_len_c.cpu(),

            "mesh_triangles_2d_coords_a": self.mesh_triangles_2d_coords_a.cpu(),
            "mesh_triangles_2d_coords_b": self.mesh_triangles_2d_coords_b.cpu(),
            "mesh_triangles_2d_coords_c": self.mesh_triangles_2d_coords_c.cpu(),

            "gaussians_to_mesh_indices": self.gaussians_to_mesh_indices.cpu(),
        }
        return export_dict

    def load_splatfacto_on_mesh(self, load_dict):
        self.gauss_params["means_2d"] = torch.nn.Parameter(load_dict["means_2d"].cuda())
        self.gauss_params["normal_elevates"] = torch.nn.Parameter(load_dict["normal_elevates"].cuda())
        self.gauss_params["scales"] = torch.nn.Parameter(load_dict["scales"].cuda())
        self.gauss_params["quats"] = torch.nn.Parameter(load_dict["quats"].cuda())
        self.gauss_params["features_dc"] = torch.nn.Parameter(load_dict["features_dc"].cuda())
        self.gauss_params["features_rest"] = torch.nn.Parameter(load_dict["features_rest"].cuda())
        self.gauss_params["opacities"] = torch.nn.Parameter(load_dict["opacities"].cuda())

        self.mesh_faces_verts = load_dict["mesh_faces_verts"].cuda()
        self.normals = load_dict["normals"].cuda()
        self.radius = load_dict["radius"].cuda()
        self.xyz_radius = load_dict["xyz_radius"].cuda()
        self.faces_quats = load_dict["faces_quats"].cuda()
        self.mesh_verts = load_dict["mesh_verts"].cuda()
        self.mesh_faces = load_dict["mesh_faces"].cuda()

        self.faces_axis_x = load_dict["faces_axis_x"].cuda()
        self.faces_axis_y = load_dict["faces_axis_y"].cuda()
        self.mesh_triangles_edge_ab = load_dict["mesh_triangles_edge_ab"].cuda()
        self.mesh_triangles_edge_bc = load_dict["mesh_triangles_edge_bc"].cuda()
        self.mesh_triangles_edge_ca = load_dict["mesh_triangles_edge_ca"].cuda()
        self.mesh_triangles_edge_len_a = load_dict["mesh_triangles_edge_len_a"].cuda()
        self.mesh_triangles_edge_len_b = load_dict["mesh_triangles_edge_len_b"].cuda()
        self.mesh_triangles_edge_len_c = load_dict["mesh_triangles_edge_len_c"].cuda()
        self.mesh_triangles_2d_coords_a = load_dict["mesh_triangles_2d_coords_a"].cuda()
        self.mesh_triangles_2d_coords_b = load_dict["mesh_triangles_2d_coords_b"].cuda()
        self.mesh_triangles_2d_coords_c = load_dict["mesh_triangles_2d_coords_c"].cuda()
        self.gaussians_to_mesh_indices = load_dict["gaussians_to_mesh_indices"].cuda()


