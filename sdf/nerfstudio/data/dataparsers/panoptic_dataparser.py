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
from PIL import Image
from rich.console import Console
from typing_extensions import Literal

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
CONSOLE = Console(width=120)
MAX_AUTO_RESOLUTION = 4000



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


def get_normals(image_idx: int, normals):
    """function to process additional depths and normal information

    Args:
        image_idx: specific image index to work with
        semantics: semantics data
    """

    # depth
    # normal
    normal = normals[image_idx]

    return {"normal": normal}

def get_depths(image_idx: int, depths):
    """function to process additional depths and normal information

    Args:
        image_idx: specific image index to work with
        semantics: semantics data
    """

    # depth
    # normal
    depth = depths[image_idx]

    return {"depth": depth}

def get_masks(image_idx: int, masks):
    """function to process additional depths and normal information

    Args:
        image_idx: specific image index to work with
        semantics: semantics data
    """

    # depth
    # normal
    mask = masks[image_idx]

    return {"fg_mask": mask}
def get_panoptics(image_idx: int, segments, semantics, instances, invalid_masks, probabilities, confidences):
    """function to process additional depths and normal information

    Args:
        image_idx: specific image index to work with
        semantics: semantics data
    """

    segment = segments[image_idx]
    semantic = semantics[image_idx]
    instance = instances[image_idx]
    invalid_mask = invalid_masks[image_idx]
    probability = probabilities[image_idx]
    confidence = confidences[image_idx]

    return {
        "segment": segment,
        "semantic": semantic,
        "instance": instance,
        "invalid_mask": invalid_mask,
        "probability": probability,
        "confidence": confidence,
    }

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
    panoptic_data: bool = False
    mono_normal_data: bool = False
    mono_depth_data: bool = False
    mask_data: bool = False
    panoptic_segment: bool = False
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "fraction"
    """
    The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval.
    All uses all the images for any split.
    """
    train_split_fraction: float = 0.9
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

        if self.config.mono_normal_data:
            normal_images = []
        if self.config.mono_depth_data:
            depth_images = []
        if self.config.mask_data:
            mask_images = []

        if self.config.panoptic_data:
            segments = []
            semantics = []
            instances = []
            invalid_masks = []
            probabilities = []
            confidences = []

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

            if self.config.mono_depth_data:
                dpath = fname.parent.parent / "depth" / (os.path.splitext(fname.name)[0] + ".npy")
                # depth = np.load(dpath)
                depth_images.append(dpath)
                # depth_images.append(torch.from_numpy(depth).float())

            if self.config.mono_normal_data:
                npath = fname.parent.parent / "normal" / (os.path.splitext(fname.name)[0]+".png")
                normal_images.append(npath)

                # normal = np.array(Image.open(npath)) / 255.0
                # normal = normal * 2.0 - 1.0  # omnidata output is normalized so we convert it back to normal here
                # normal = torch.from_numpy(normal).float()
                # normal[..., 1:3] *= -1
                # normal_images.append(normal)

            if self.config.mask_data:
                mask_path = fname.parent.parent / "mask" / (os.path.splitext(fname.name)[0]+".png")
                mask_images.append(mask_path)

            if self.config.panoptic_data:
                assert height_fixed and width_fixed
                pstem = os.path.splitext(fname.name)[0]

                segment_mask = torch.from_numpy(np.load(self.config.data / "segments" / (pstem+".npy")).astype(np.int32))
                semantic = torch.from_numpy(np.load(self.config.data / "semantics" / (pstem+".npy")).astype(np.int32))
                instance = torch.from_numpy(np.load(self.config.data / "instance" / (pstem+".npy")).astype(np.int32))
                invalid_mask = torch.from_numpy(np.load(self.config.data / "invalid" / (pstem+".npy")))
                probability = torch.from_numpy(np.load(self.config.data / "probabilities" / (pstem+".npy")))
                confidence = torch.from_numpy(np.load(self.config.data / "confidences" / (pstem+".npy")))

                segments.append(segment_mask.long())
                semantics.append(semantic.long())
                instances.append(instance.long())
                invalid_masks.append(invalid_mask.bool())
                probabilities.append(probability.float())
                confidences.append(confidence.float())

        if self.config.panoptic_data:

            with open(os.path.join(self.config.data / 'segmentation_data.pkl'), 'rb') as f:
                segment_data = pickle.load(f)

            self.segment_data = create_segmentation_data_panopli(segment_data)
            self.total_classes = len(self.segment_data["bg_classes"]) + len(self.segment_data["fg_classes"])

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

        # if len(indices) > 2000:
        #     indices = indices[::2]

        if "orientation_override" in meta:
            orientation_method = meta["orientation_override"]
            CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
        else:
            orientation_method = self.config.orientation_method

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
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
        if self.config.mono_normal_data:
            normal_paths = filter_list(normal_images, indices)
            normal_images = []
            for npath in tqdm(normal_paths):
                normal = np.array(Image.open(npath)) / 255.0
                normal = normal * 2.0 - 1.0  # omnidata output is normalized so we convert it back to normal here
                normal = torch.from_numpy(normal).float()
                normal[..., 1:3] *= -1
                normal_images.append(normal)

            normal_images_aligned = []
            for norm_i, normal_image in tqdm(enumerate(normal_images), total=len(normal_images)):
                h, w, _ = normal_image.shape
                normal_image = normal_image.reshape(-1, 3) @ torch.inverse(poses[norm_i, :3, :3])
                normal_image = normal_image.reshape(h, w, 3)
                normal_images_aligned.append(normal_image)
            normal_images = normal_images_aligned

        if self.config.mono_depth_data:
            depth_paths = filter_list(depth_images, indices)
            depth_images = []

            for dpath in depth_paths:
                depth_frame = torch.from_numpy(np.load(dpath)).float()
                depth_images.append(depth_frame)

        if self.config.mask_data:
            mask_paths = filter_list(mask_images, indices)
            mask_images = []
            for mpath in mask_paths:
                mask_frame = torch.from_numpy(np.array(Image.open(mpath)).astype(np.float32) / 255.0)
                mask_images.append(mask_frame)

        additional_inputs_dict = {}
        metadata = {}

        metadata["transform_matrix"] = transform_matrix
        metadata["scale_factor"] = scale_factor

        scale_factor_matrix = torch.eye(4)
        scale_factor_matrix[torch.arange(3), torch.arange(3)] *= scale_factor

        trans_h, trans_w = transform_matrix.shape
        total_transform = torch.eye(4)
        total_transform[:trans_h, :trans_w] = transform_matrix

        total_transform = scale_factor_matrix @ total_transform
        metadata["total_transform"] = total_transform
        metadata["total_transform_inv"] = torch.inverse(total_transform)

        if self.config.mono_normal_data:
            additional_inputs_dict["normals_cues"] = {
                "func": get_normals,
                "kwargs": {
                    "normals": normal_images,
                },
            }
        if self.config.mono_depth_data:
            additional_inputs_dict["cues"] = {
                "func": get_depths,
                "kwargs": {
                    "depths": depth_images,
                },
            }
        if self.config.mask_data:
            additional_inputs_dict["mask"] = {
                "func": get_masks,
                "kwargs": {
                    "masks": mask_images,
                },
            }
        if self.config.panoptic_data:

            segments = filter_list(segments, indices)
            semantics = filter_list(semantics, indices)
            instances = filter_list(instances, indices)
            invalid_masks = filter_list(invalid_masks, indices)
            probabilities = filter_list(probabilities, indices)
            confidences = filter_list(confidences, indices)

            additional_inputs_dict["panoptic"] = {
                "func": get_panoptics,
                "kwargs": {
                    "segments": segments,
                    "semantics": semantics,
                    "instances": instances,
                    "invalid_masks": invalid_masks,
                    "probabilities": probabilities,
                    "confidences": confidences,
                },
            }
            if split == "train" and self.config.panoptic_segment:
                all_pixels = torch.from_numpy(np.stack(np.meshgrid(
                    np.linspace(0, meta["h"] - 1, meta["h"]).astype(np.int32),
                    np.linspace(0, meta["w"] - 1, meta["w"]).astype(np.int32),
                    indexing='ij'
                ), axis=-1))
                metadata = {
                    "segments_rays": [],
                    "segments_confs": [],
                    "segments_ones": [],
                }
                for segment_i in range(len(segments)):
                    segment = segments[segment_i]
                    ray_indices = torch.cat(
                        [torch.ones_like(all_pixels[..., :1]) * segment_i, all_pixels], dim=-1
                    ).long()
                    for s in torch.unique(segment):
                        if s.item() != 0:
                            metadata["segments_rays"].append(ray_indices[segment == s].reshape(-1, 3))
                            metadata["segments_confs"].append(confidences[segment_i][segment == s].reshape(-1))
                            metadata["segments_ones"].append(torch.ones(metadata["segments_confs"][-1].shape[0]).long())

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            ),
            near=0.01,
            far=10.0
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

        n = 0.001
        f = 10  # infinite

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
        i_pose[:, :3, 1:3] *= -1
        square_pose = torch.cat((i_pose, bottom), dim=1)
        projections = camera_projmat.unsqueeze(0) @ torch.inverse(square_pose)

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
            projections=projections,
            camprojs=camera_projmat.unsqueeze(0).expand(poses.shape[0], -1, -1),
        )

        assert self.downscale_factor is not None
        cameras.rescale_output_resolution(scaling_factor=1.0 / self.downscale_factor)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            additional_inputs=additional_inputs_dict,
            metadata=metadata,
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
