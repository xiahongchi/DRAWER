# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
""" Data parser for nerfstudio datasets. """

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Optional, Type

import numpy as np
import torch
import torchvision.transforms.functional
import trimesh
from PIL import Image
from rich.console import Console
# from typing_extensions import Literal
from typing import Literal

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json
from tqdm import tqdm
import os, pickle

from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.ops import sample_points_from_meshes, SubdivideMeshes
from collections import namedtuple

import nvdiffrast.torch as dr
import torch.nn.functional as F

from scipy.spatial import KDTree
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 1024


def get_mesh_depths(image_idx: int, mesh_depths):
    """function to process additional depths and normal information

    Args:
        image_idx: specific image index to work with
        semantics: semantics data
    """

    # depth
    # normal
    mesh_depth = mesh_depths[image_idx]

    return {"mesh_depth": mesh_depth}


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


def get_train_eval_split_fraction(image_filenames: List, train_split_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split fraction based on the number of images and the train split fraction.

    Args:
        image_filenames: list of image filenames
        train_split_fraction: fraction of images to use for training
    """

    # filter image_filenames and poses based on train/eval split percentage
    num_images = len(image_filenames)
    num_train_images = math.ceil(num_images * train_split_fraction)
    num_eval_images = num_images - num_train_images
    i_all = np.arange(num_images)
    i_train = np.linspace(
        0, num_images - 1, num_train_images, dtype=int
    )  # equally spaced training images starting and ending at 0 and num_images-1
    i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
    assert len(i_eval) == num_eval_images

    return i_train, i_eval


def get_train_eval_split_filename(image_filenames: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split based on the filename of the images.

    Args:
        image_filenames: list of image filenames
    """

    num_images = len(image_filenames)
    basenames = [os.path.basename(image_filename) for image_filename in image_filenames]
    i_all = np.arange(num_images)
    i_train = []
    i_eval = []
    for idx, basename in zip(i_all, basenames):
        # check the frame index
        if "train" in basename:
            i_train.append(idx)
        elif "eval" in basename:
            i_eval.append(idx)
        else:
            raise ValueError("frame should contain train/eval in its name to use this eval-frame-index eval mode")

    return np.array(i_train), np.array(i_eval)


def get_train_eval_split_interval(image_filenames: List, eval_interval: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split based on the interval of the images.

    Args:
        image_filenames: list of image filenames
        eval_interval: interval of images to use for eval
    """

    num_images = len(image_filenames)
    all_indices = np.arange(num_images)
    train_indices = all_indices[all_indices % eval_interval != 0]
    eval_indices = all_indices[all_indices % eval_interval == 0]
    i_train = train_indices
    i_eval = eval_indices

    return i_train, i_eval


def get_train_eval_split_all(image_filenames: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split where all indices are used for both train and eval.

    Args:
        image_filenames: list of image filenames
    """
    num_images = len(image_filenames)
    i_all = np.arange(num_images)
    i_train = i_all
    i_eval = i_all
    return i_train, i_eval


# def get_normals(image_idx: int, normals):
#     """function to process additional depths and normal information

#     Args:
#         image_idx: specific image index to work with
#         semantics: semantics data
#     """

#     # depth
#     # normal
#     normal = normals[image_idx]

#     return {"normal": normal}

# def get_depths(image_idx: int, depths):
#     """function to process additional depths and normal information

#     Args:
#         image_idx: specific image index to work with
#         semantics: semantics data
#     """

#     # depth
#     # normal
#     depth = depths[image_idx]

#     return {"depth": depth}

# def get_panoptics(image_idx: int, segments, semantics, instances, invalid_masks, probabilities, confidences):
#     """function to process additional depths and normal information

#     Args:
#         image_idx: specific image index to work with
#         semantics: semantics data
#     """

#     segment = segments[image_idx]
#     semantic = semantics[image_idx]
#     instance = instances[image_idx]
#     invalid_mask = invalid_masks[image_idx]
#     probability = probabilities[image_idx]
#     confidence = confidences[image_idx]

#     return {
#         "segment": segment,
#         "semantic": semantic,
#         "instance": instance,
#         "invalid_mask": invalid_mask,
#         "probability": probability,
#         "confidence": confidence,
#     }

def filter_list(list_to_filter, indices):
    """Returns a copy list with only selected indices"""
    if list_to_filter:
        return [list_to_filter[i] for i in indices]
    else:
        return []

def create_segmentation_data_panopli(seg_data):
    seg_data_dict = {
        'fg_classes': sorted(seg_data['fg_classes']),
        'bg_classes': sorted(seg_data['bg_classes']),
        'instance_to_semantics': seg_data["instance_to_semantic"],
        'num_semantic_classes': len(seg_data['fg_classes'] + seg_data['bg_classes']),
        'num_instances': len(seg_data['fg_classes'])
    }
    return seg_data_dict

@dataclass
class PanopticDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: Panoptic)
    """target class to instantiate"""
    data: Path = Path("data/nerfstudio/poster")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "none"] = "up"
    """The method to use for orientation."""
    center_poses: bool = True
    """Whether to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_percentage: float = 0.9
    """The percent of images to use for training. The remaining images are for eval."""
    use_all_train_images: bool = False
    """Whether to use all images for training. If True, all images are used for training."""
    # panoptic_data: bool = False
    # mono_normal_data: bool = False
    # mono_depth_data: bool = False
    # panoptic_segment: bool = False
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "fraction"
    """
    The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval.
    All uses all the images for any split.
    """
    train_split_fraction: float = 0.9
    mesh_gauss_path: Optional[str] = None
    mesh_area_to_subdivide: float = 1e-5
    drawer_transform_dir: Optional[str] = None
    mesh_depth: bool = False
    num_max_image: int = 5000

@dataclass
class Panoptic(DataParser):
    """Nerfstudio DatasetParser"""

    config: PanopticDataParserConfig
    downscale_factor: Optional[int] = None

    def _generate_dataparser_outputs(self, split="train"):
        # pylint: disable=too-many-statements

        meta = load_from_json(self.config.data / "transforms.json")
        image_filenames = []
        mask_filenames = []
        poses = []
        num_skipped_image_filenames = 0

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        # if self.config.mono_normal_data:
        #     normal_images = []
        # if self.config.mono_depth_data:
        #     depth_images = []

        # if self.config.panoptic_data:
        #     segments = []
        #     semantics = []
        #     instances = []
        #     invalid_masks = []
        #     probabilities = []
        #     confidences = []
        
        if self.config.num_max_image < len(meta["frames"]):
            selected_frames = np.linspace(0, len(meta["frames"]) - 1, self.config.num_max_image, dtype=int)
            meta["frames"] = [meta["frames"][i] for i in selected_frames]

        for frame in tqdm(meta["frames"]):
            filepath = PurePath(frame["file_path"])
            fname = self._get_fname(filepath)
            if not fname.exists():
                num_skipped_image_filenames += 1
                continue

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    camera_utils.get_distortion_params(
                        k1=float(meta["k1"]) if "k1" in meta else 0.0,
                        k2=float(meta["k2"]) if "k2" in meta else 0.0,
                        k3=float(meta["k3"]) if "k3" in meta else 0.0,
                        k4=float(meta["k4"]) if "k4" in meta else 0.0,
                        p1=float(meta["p1"]) if "p1" in meta else 0.0,
                        p2=float(meta["p2"]) if "p2" in meta else 0.0,
                    )
                )

            image_filenames.append(fname)
            pose = np.array(frame["transform_matrix"])
            poses.append(pose)
            if "mask_path" in frame:
                mask_filepath = PurePath(frame["mask_path"])
                mask_fname = self._get_fname(mask_filepath, downsample_folder_prefix="masks_")
                mask_filenames.append(mask_fname)

            # if self.config.mono_depth_data:
            #     dpath = fname.parent.parent / "depth" / (os.path.splitext(fname.name)[0] + ".npy")
            #     # depth = np.load(dpath)
            #     depth_images.append(dpath)
            #     # depth_images.append(torch.from_numpy(depth).float())

            # if self.config.mono_normal_data:
            #     npath = fname.parent.parent / "normal" / (os.path.splitext(fname.name)[0]+".png")
            #     normal_images.append(npath)

                # normal = np.array(Image.open(npath)) / 255.0
                # normal = normal * 2.0 - 1.0  # omnidata output is normalized so we convert it back to normal here
                # normal = torch.from_numpy(normal).float()
                # normal[..., 1:3] *= -1
                # normal_images.append(normal)

            # if self.config.panoptic_data:
            #     assert height_fixed and width_fixed
            #     pstem = os.path.splitext(fname.name)[0]

            #     segment_mask = torch.from_numpy(np.load(self.config.data / "segments" / (pstem+".npy")).astype(np.int32))
            #     semantic = torch.from_numpy(np.load(self.config.data / "semantics" / (pstem+".npy")).astype(np.int32))
            #     instance = torch.from_numpy(np.load(self.config.data / "instance" / (pstem+".npy")).astype(np.int32))
            #     invalid_mask = torch.from_numpy(np.load(self.config.data / "invalid" / (pstem+".npy")))
            #     probability = torch.from_numpy(np.load(self.config.data / "probabilities" / (pstem+".npy")))
            #     confidence = torch.from_numpy(np.load(self.config.data / "confidences" / (pstem+".npy")))

            #     segments.append(segment_mask.long())
            #     semantics.append(semantic.long())
            #     instances.append(instance.long())
            #     invalid_masks.append(invalid_mask.bool())
            #     probabilities.append(probability.float())
            #     confidences.append(confidence.float())

        # if self.config.panoptic_data:

        #     with open(os.path.join(self.config.data / 'segmentation_data.pkl'), 'rb') as f:
        #         segment_data = pickle.load(f)

        #     self.segment_data = create_segmentation_data_panopli(segment_data)
        #     self.total_classes = len(self.segment_data["bg_classes"]) + len(self.segment_data["fg_classes"])

        if num_skipped_image_filenames >= 0:
            CONSOLE.log(f"Skipping {num_skipped_image_filenames} files in dataset split {split}.")
        assert (
            len(image_filenames) != 0
        ), """
        No image files found. 
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """
        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """

        # filter image_filenames and poses based on train/eval split percentage
        has_split_files_spec = any(f"{split}_filenames" in meta for split in ("train", "val", "test"))
        if self.config.use_all_train_images and split == "train":
            num_images = len(image_filenames)
            i_all = np.arange(num_images)
            indices = i_all
        elif f"{split}_filenames" in meta:
            # Validate split first
            split_filenames = set(self._get_fname(Path(x), data_dir) for x in meta[f"{split}_filenames"])
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(f"Some filenames for split {split} were not found: {unmatched_filenames}.")

            indices = [i for i, path in enumerate(image_filenames) if path in split_filenames]
            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
        else:
            # find train and eval indices based on the eval_mode specified
            if self.config.eval_mode == "fraction":
                i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
            elif self.config.eval_mode == "filename":
                i_train, i_eval = get_train_eval_split_filename(image_filenames)
            elif self.config.eval_mode == "interval":
                i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
            elif self.config.eval_mode == "all":
                CONSOLE.log(
                    "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
                )
                i_train, i_eval = get_train_eval_split_all(image_filenames)
            else:
                raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")

        # indices = indices[::10]

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses_deprecated(
            poses,
            method=orientation_method,
            center_poses=self.config.center_poses,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        image_filenames = [image_filenames[i] for i in indices]
        mask_filenames = [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        poses = poses[indices]


        # we should also transform normal accordingly
        # if self.config.mono_normal_data:
        #     normal_paths = filter_list(normal_images, indices)
        #     normal_images = []
        #     for npath in tqdm(normal_paths):
        #         normal = np.array(Image.open(npath)) / 255.0
        #         normal = normal * 2.0 - 1.0  # omnidata output is normalized so we convert it back to normal here
        #         normal = torch.from_numpy(normal).float()
        #         normal[..., 1:3] *= -1
        #         normal_images.append(normal)

        #     normal_images_aligned = []
        #     for norm_i, normal_image in tqdm(enumerate(normal_images), total=len(normal_images)):
        #         h, w, _ = normal_image.shape
        #         normal_image = normal_image.reshape(-1, 3) @ torch.inverse(poses[norm_i, :3, :3])
        #         normal_image = normal_image.reshape(h, w, 3)
        #         normal_images_aligned.append(normal_image)
        #     normal_images = normal_images_aligned

        # if self.config.mono_depth_data:
        #     depth_paths = filter_list(depth_images, indices)
        #     depth_images = []

        #     for dpath in depth_paths:
        #         depth_frame = torch.from_numpy(np.load(dpath)).float()
        #         depth_images.append(depth_frame)

        # additional_inputs_dict = {}
        metadata = {}
        # if self.config.mono_normal_data:
        #     additional_inputs_dict["normals_cues"] = {
        #         "func": get_normals,
        #         "kwargs": {
        #             "normals": normal_images,
        #         },
        #     }
        # if self.config.mono_depth_data:
        #     additional_inputs_dict["cues"] = {
        #         "func": get_depths,
        #         "kwargs": {
        #             "depths": depth_images,
        #         },
        #     }
        # if self.config.panoptic_data:

        #     segments = filter_list(segments, indices)
        #     semantics = filter_list(semantics, indices)
        #     instances = filter_list(instances, indices)
        #     invalid_masks = filter_list(invalid_masks, indices)
        #     probabilities = filter_list(probabilities, indices)
        #     confidences = filter_list(confidences, indices)

        #     additional_inputs_dict["panoptic"] = {
        #         "func": get_panoptics,
        #         "kwargs": {
        #             "segments": segments,
        #             "semantics": semantics,
        #             "instances": instances,
        #             "invalid_masks": invalid_masks,
        #             "probabilities": probabilities,
        #             "confidences": confidences,
        #         },
        #     }
        #     if split == "train" and self.config.panoptic_segment:
        #         all_pixels = torch.from_numpy(np.stack(np.meshgrid(
        #             np.linspace(0, meta["h"] - 1, meta["h"]).astype(np.int32),
        #             np.linspace(0, meta["w"] - 1, meta["w"]).astype(np.int32),
        #             indexing='ij'
        #         ), axis=-1))
        #         metadata = {
        #             "segments_rays": [],
        #             "segments_confs": [],
        #             "segments_ones": [],
        #         }
        #         for segment_i in range(len(segments)):
        #             segment = segments[segment_i]
        #             ray_indices = torch.cat(
        #                 [torch.ones_like(all_pixels[..., :1]) * segment_i, all_pixels], dim=-1
        #             ).long()
        #             for s in torch.unique(segment):
        #                 if s.item() != 0:
        #                     metadata["segments_rays"].append(ray_indices[segment == s].reshape(-1, 3))
        #                     metadata["segments_confs"].append(confidences[segment_i][segment == s].reshape(-1))
        #                     metadata["segments_ones"].append(torch.ones(metadata["segments_confs"][-1].shape[0]).long())

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        fx = float(meta["fl_x"]) if fx_fixed else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = float(meta["fl_y"]) if fy_fixed else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = float(meta["cx"]) if cx_fixed else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = float(meta["cy"]) if cy_fixed else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = int(meta["h"]) if height_fixed else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = int(meta["w"]) if width_fixed else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        if distort_fixed:
            distortion_params = camera_utils.get_distortion_params(
                k1=float(meta["k1"]) if "k1" in meta else 0.0,
                k2=float(meta["k2"]) if "k2" in meta else 0.0,
                k3=float(meta["k3"]) if "k3" in meta else 0.0,
                k4=float(meta["k4"]) if "k4" in meta else 0.0,
                p1=float(meta["p1"]) if "p1" in meta else 0.0,
                p2=float(meta["p2"]) if "p2" in meta else 0.0,
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        n = 0.01
        f = 1e10  # infinite

        n00 = 2.0 * fx / width
        n11 = 2.0 * fy / height
        n02 = 2.0 * cx / width - 1.0
        n12 = 2.0 * cy / height - 1.0
        n32 = 1.0
        n22 = (f + n) / (f - n)
        n23 = (2 * f * n) / (n - f)
        camera_projmat = np.array([[n00, 0, n02, 0],
                                   [0, n11, n12, 0],
                                   [0, 0, n22, n23],
                                   [0, 0, n32, 0]], dtype=np.float32)

        camera_projmat = torch.from_numpy(camera_projmat)
        bottom = torch.tensor([0, 0, 0, 1]).reshape(1, 1, 4).expand((poses.shape[0], -1, -1))
        i_pose = poses.clone()[:, :3, :4]
        i_pose[..., :3, 1:3] *= -1
        square_pose = torch.cat((i_pose, bottom), dim=1)
        mvps = camera_projmat.unsqueeze(0) @ torch.inverse(square_pose)
        mvps = mvps.reshape(-1, 4, 4).float()
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
            mvps=mvps
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        additional_inputs_dict = {}

        if self.config.mesh_gauss_path is not None:
            sparse_points = self._load_3D_points_on_mesh(self.config.mesh_gauss_path)
            metadata.update(sparse_points)

            if self.config.mesh_depth:
                glctx = dr.RasterizeCudaContext()

                vertices = sparse_points["mesh_verts"].cuda().float()
                triangles = sparse_points["mesh_faces"].cuda().long()
                vertices_pad = F.pad(vertices, pad=(0, 1), mode='constant', value=1.0)

                frame_depth_list = []

                for i in tqdm(range(mvps.shape[0])):
                    mvp = mvps[i]
                    pose = square_pose[i]
                    w2c = torch.inverse(pose)
                    vertices_clip = torch.matmul(vertices_pad, torch.transpose(mvp.cuda(), 0, 1)).float().unsqueeze(0)
                    vertices_cam = torch.matmul(vertices_pad, torch.transpose(w2c.cuda(), 0, 1)).float()
                    vertices_cam = vertices_cam[:, :3] / vertices_cam[:, 3:]
                    vertices_depth_cam = vertices_cam[:, -1].reshape(-1, 1)

                    rast, _ = dr.rasterize(glctx, vertices_clip, triangles.int(), (height, width))
                    # rast = rast.flip([1]).cuda()
                    bary = torch.stack([rast[..., 0], rast[..., 1], 1 - rast[..., 0] - rast[..., 1]], dim=-1).reshape(height, width, 3)
                    pix_to_face = rast[..., -1].reshape(height, width)
                    valid_pix = pix_to_face > 0
                    pix_to_face = (pix_to_face - 1).long()

                    pix_valid_depth = vertices_depth_cam[triangles[pix_to_face[valid_pix]].reshape(-1)].reshape(-1, 3)
                    pix_valid_bary = bary[valid_pix].reshape(-1).reshape(-1, 3)
                    # print("0 pix_valid_depth: ", pix_valid_depth.shape, pix_valid_depth.max(), pix_valid_depth.min())

                    pix_valid_inverse_depth = 1 / (pix_valid_depth + 1e-10)
                    pix_valid_depth = 1 / (torch.sum(pix_valid_inverse_depth * pix_valid_bary, dim=-1) + 1e-10)
                    # print("1 pix_valid_depth: ", pix_valid_depth.shape, pix_valid_depth.max(), pix_valid_depth.min())

                    frame_depth = torch.zeros((height, width), device="cuda")
                    frame_depth[valid_pix] = pix_valid_depth.reshape(-1)
                    frame_depth = torch.abs(frame_depth)

                    invalid_indices = torch.nonzero(torch.logical_not(valid_pix)).reshape(-1, 2)
                    for dx, dy in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                        invalid_indices_offset = invalid_indices + torch.tensor([dx, dy]).cuda().reshape(1, 2).int()
                        invalid_indices_offset[:, 0] = torch.clip(invalid_indices_offset[:, 0], 0, height - 1)
                        invalid_indices_offset[:, 1] = torch.clip(invalid_indices_offset[:, 1], 0, width - 1)
                        frame_depth[torch.logical_not(valid_pix)] += frame_depth[invalid_indices_offset[:, 0], invalid_indices_offset[:, 1]] * 0.25

                    frame_depth_list.append(frame_depth.cpu())

                metadata.update({"mesh_depths": frame_depth_list})


        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
        )
        return dataparser_outputs

    def _get_fname(self, filepath: PurePath, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxillary image data, e.g. masks
        """

        if self.downscale_factor is None:
            if self.config.downscale_factor is None:
                test_img = Image.open(self.config.data / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (self.config.data / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                        break
                    df += 1

                self.downscale_factor = 2**df
                CONSOLE.log(f"Auto image downscale factor of {self.downscale_factor}")
            else:
                self.downscale_factor = self.config.downscale_factor

        if self.downscale_factor > 1:
            return self.config.data / f"{downsample_folder_prefix}{self.downscale_factor}" / filepath.name
        return self.config.data / filepath
    
    def _load_3D_points(self, texture_mesh_dir: str, transform_matrix: torch.Tensor, scale_factor: float):
        """Loads point clouds positions and colors from .ply

        Args:
            ply_file_path: Path to .ply file
            transform_matrix: Matrix to transform world coordinates
            scale_factor: How much to scale the camera origins by.

        Returns:
            A dictionary of points: points3D_xyz and colors: points3D_rgb
        """
        # import open3d as o3d  # Importing open3d is slow, so we only do it if we need it.

        # pcd = o3d.io.read_point_cloud(str(ply_file_path))

        # # if no points found don't read in an initial point cloud
        # if len(pcd.points) == 0:
        #     return None

        # points3D = torch.from_numpy(np.asarray(pcd.points, dtype=np.float32))

        obj_filename = os.path.join(texture_mesh_dir, "mesh.obj")
        pts_filename = os.path.join(texture_mesh_dir, "sampled_pts.pkl")

        if os.path.exists(pts_filename):
            with open(pts_filename, "rb") as f:
                points_dict = pickle.load(f)
            points3D = torch.from_numpy(points_dict["points3D_xyz"])
            points3D_rgb = torch.from_numpy(points_dict["points3D_rgb"])

        else:
            mesh = load_objs_as_meshes([obj_filename], device='cpu')

            points3D, points3D_rgb = sample_points_from_meshes(mesh, num_samples=300000, return_textures=True)

            points3D = points3D.reshape(-1, 3)
            points3D_rgb = points3D_rgb.reshape(-1, 3)
            with open(pts_filename, "wb") as f:
                pickle.dump({
                    "points3D_xyz": points3D.cpu().numpy(),
                    "points3D_rgb": points3D_rgb.cpu().numpy(),
                }, f)


        # points3D = (
        #     torch.cat(
        #         (
        #             points3D,
        #             torch.ones_like(points3D[..., :1]),
        #         ),
        #         -1,
        #     )
        #     @ transform_matrix.T
        # )
        # points3D *= scale_factor
        points3D_rgb = torch.from_numpy((np.asarray(points3D_rgb) * 255).astype(np.uint8))

        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
        }
        return out


    def _load_3D_points_on_mesh(self, texture_mesh_path):
        obj_filename = texture_mesh_path

        mesh = load_objs_as_meshes([obj_filename], device='cpu')
        # while mesh.faces_packed().shape[0] < target_pts_num:
        #     mesh = SubdivideMeshes()(mesh)
        #     print("face number: ", mesh.faces_packed().shape[0])

        mesh_verts = mesh.verts_packed().clone().reshape(-1, 3)
        mesh_faces = mesh.faces_packed().clone().reshape(-1, 3)

        N_Gaussians = mesh_faces.shape[0]
        triangles = mesh_verts[mesh_faces.reshape(-1)].reshape(-1, 3, 3)
        means = torch.mean(triangles, dim=1)
        radius = circumcircle_radius(triangles)

        Mesh_Fragments = namedtuple("Mesh_Fragments", ['pix_to_face', 'bary_coords'])
        mesh_fragments = Mesh_Fragments(
            pix_to_face=torch.arange(N_Gaussians).reshape(1, 1, N_Gaussians, 1),
            bary_coords=(torch.ones(1, 1, N_Gaussians, 1, 3) / 3)
        )
        features_dc = mesh.textures.sample_textures(mesh_fragments).reshape(N_Gaussians, 3)

        normals = mesh.faces_normals_packed().clone().reshape(-1, 3)

        area_to_subdivide = self.config.mesh_area_to_subdivide

        print("before subdivision: ")
        print("num of gs points: ", N_Gaussians)
        while True:
            areas = area(triangles)
            if torch.all(areas[torch.isfinite(areas)] <= area_to_subdivide):
                break
            face_to_subdivide = (areas > area_to_subdivide)

            mesh_faces_subdivided = mesh_faces[face_to_subdivide]

            verts_0_idxs = mesh_faces_subdivided[:, 0]
            verts_1_idxs = mesh_faces_subdivided[:, 1]
            verts_2_idxs = mesh_faces_subdivided[:, 2]

            # edges_faces_subdivided = mesh_faces_subdivided[:, 0]
            edges_faces_subdivided_01 = torch.stack([verts_0_idxs, verts_1_idxs], dim=-1)
            edges_faces_subdivided_02 = torch.stack([verts_0_idxs, verts_2_idxs], dim=-1)
            edges_faces_subdivided_12 = torch.stack([verts_1_idxs, verts_2_idxs], dim=-1)

            edges_faces_subdivided_01 = torch.sort(edges_faces_subdivided_01, dim=-1)[0]
            edges_faces_subdivided_02 = torch.sort(edges_faces_subdivided_02, dim=-1)[0]
            edges_faces_subdivided_12 = torch.sort(edges_faces_subdivided_12, dim=-1)[0]


            edges_faces_subdivided = torch.stack([
                edges_faces_subdivided_01,
                edges_faces_subdivided_02,
                edges_faces_subdivided_12,
            ], dim=1)
            edges_faces_subdivided_flatten = edges_faces_subdivided.reshape(-1, 2)
            edges_faces_subdivided_unique, edges_faces_subdivided_unique_inverse_idx = torch.unique(edges_faces_subdivided_flatten, return_inverse=True, dim=0)

            edges_faces_subdivided_unique_verts = mesh_verts[edges_faces_subdivided_unique.reshape(-1)].reshape(-1, 2, 3)
            mesh_verts_added = (edges_faces_subdivided_unique_verts[:, 0] + edges_faces_subdivided_unique_verts[:, 1]) / 2

            num_verts_before = mesh_verts.shape[0]
            mesh_verts_added_idxs = num_verts_before + torch.arange(mesh_verts_added.shape[0])

            verts_abc_idxs = mesh_verts_added_idxs[edges_faces_subdivided_unique_inverse_idx]
            verts_abc_idxs = verts_abc_idxs.reshape(-1, 3)

            verts_a_idxs = verts_abc_idxs[:, 0]
            verts_b_idxs = verts_abc_idxs[:, 1]
            verts_c_idxs = verts_abc_idxs[:, 2]

            faces_0ab = torch.stack([verts_0_idxs, verts_a_idxs, verts_b_idxs], dim=-1)
            faces_1ca = torch.stack([verts_1_idxs, verts_c_idxs, verts_a_idxs], dim=-1)
            faces_2bc = torch.stack([verts_2_idxs, verts_b_idxs, verts_c_idxs], dim=-1)
            faces_acb = torch.stack([verts_a_idxs, verts_c_idxs, verts_b_idxs], dim=-1)

            mesh_faces[face_to_subdivide] = faces_acb
            mesh_faces = torch.cat([
                mesh_faces,
                faces_0ab, faces_1ca, faces_2bc
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
            features_dc = torch.cat([features_dc] + [features_dc[face_to_subdivide]] * 3, dim=0)
        print("after basic subdivision: ")
        print("num of gs points: ", N_Gaussians)
        # trimesh.exchange.export.export_mesh(
        #     trimesh.Trimesh(mesh_verts.cpu().numpy(), mesh_faces.cpu().numpy(), process=False),
        #     os.path.join("./vis/cs_kitchen/subdivided.ply")
        # )

        if self.config.drawer_transform_dir is not None:
            drawer_transform_dir = self.config.drawer_transform_dir
            num_drawers = len([elem for elem in os.listdir(drawer_transform_dir) if elem.endswith(".pkl")])
            prim_transform_list = []
            for prim_i in range(num_drawers):
                with open(os.path.join(drawer_transform_dir, f"drawer_{prim_i}.pkl"), 'rb') as f:
                    prim_transform = pickle.load(f)
                    prim_transform = torch.from_numpy(prim_transform["transform"]).float()
                    prim_transform_list.append(prim_transform)

            area_to_subdivide = self.config.mesh_area_to_subdivide * 0.01
            door_depth = 0.05

            while True:
                areas = area(triangles)
                faces_to_subdivide_large_area = (areas > area_to_subdivide).reshape(-1)

                full_mesh_verts_pad = torch.nn.functional.pad(mesh_verts, (0, 1), "constant", 1.0)
                num_verts = mesh_verts.shape[0]
                verts_need_subdivide = torch.zeros((num_verts), dtype=torch.bool)

                for prim_transform in prim_transform_list:

                    means_pad_transformed = full_mesh_verts_pad @ torch.inverse(prim_transform).T
                    means_transformed = means_pad_transformed[:, :3] / means_pad_transformed[:, 3:]
                    scale_limit = np.array([1e4 * door_depth, 1, 1]).reshape(1, 3)
                    means_transformed = means_transformed / scale_limit
                    prim_i_means_indices = torch.all((torch.abs(means_transformed) < 0.5), dim=1).reshape(-1)
                    verts_need_subdivide[prim_i_means_indices] = True

                faces_to_subdivide_inside_drawers = torch.any(torch.isin(mesh_faces, torch.arange(num_verts)[verts_need_subdivide]), dim=-1).reshape(-1)

                face_to_subdivide = torch.logical_and(faces_to_subdivide_large_area, faces_to_subdivide_inside_drawers)
                if not torch.any(face_to_subdivide):
                    break

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

                mesh_faces[face_to_subdivide] = faces_acb
                mesh_faces = torch.cat([
                    mesh_faces,
                    faces_0ab, faces_1ca, faces_2bc
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
                features_dc = torch.cat([features_dc] + [features_dc[face_to_subdivide]] * 3, dim=0)

        print("after further subdivision: ")
        print("num of gs points: ", N_Gaussians)

        out = {
            "means": means,
            "radius": radius,
            "features_dc": features_dc,
            "normals": normals,
            "mesh_verts": mesh_verts,
            "mesh_faces": mesh_faces,
            "mesh_dir": os.path.dirname(texture_mesh_path),
        }

        return out

