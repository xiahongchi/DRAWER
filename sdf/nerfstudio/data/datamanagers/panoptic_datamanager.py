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

"""
Semantic datamanager.
"""

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.datasets.panoptic_dataset import PanopticDataset, PanopticSegmentDataset
from nerfstudio.cameras.rays import RayBundle
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import torch
from rich.progress import Console
CONSOLE = Console(width=120)
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
)
from nerfstudio.model_components.ray_generators import RayGenerator
from torch.utils.data import DataLoader

@dataclass
class PanopticDataManagerConfig(VanillaDataManagerConfig):
    """A semantic datamanager - required to use with .setup()"""

    _target: Type = field(default_factory=lambda: PanopticDataManager)
    batch_size_segments: int = 32
    segment_loss_iterations: int = 10000


class PanopticDataManager(VanillaDataManager):  # pylint: disable=abstract-method
    """Data manager implementation for data that also requires processing semantic data.

    Args:
        config: the DataManagerConfig used to instantiate class
    """
    config: PanopticDataManagerConfig

    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )
        # for loading full images
        self.fixed_indices_train_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.train_dataset,
            device=self.device,
            num_workers=self.world_size * 2,
            shuffle=False,
        )
        if self.train_dataset.segment_enabled:
            self.train_seg_dataloader = DataLoader(self.train_segment_set, self.config.batch_size_segments, shuffle=True, drop_last=True,
                       collate_fn=self.train_segment_set.collate_fn, num_workers=self.world_size * 4)
            self.iter_train_seg_dataloader = iter(self.train_seg_dataloader)
            self.iter_train_seg_dataloader_cnt = 0

    def create_train_dataset(self) -> PanopticDataset:
        parser_output = self.dataparser.get_dataparser_outputs(split="train")

        train_dataset = PanopticDataset(
            dataparser_outputs=parser_output,
            scale_factor=self.config.camera_res_scale_factor,
        )

        if train_dataset.segment_enabled:
            self.train_segment_set = PanopticSegmentDataset(parser_output)

        return train_dataset

    def create_eval_dataset(self) -> PanopticDataset:
        return PanopticDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        if self.train_dataset.segment_enabled and self.train_count > self.config.segment_loss_iterations:
            if self.iter_train_seg_dataloader_cnt < len(self.train_seg_dataloader):
                segment_batch = next(self.iter_train_seg_dataloader)
                batch["segments_rays"] = segment_batch["rays"].cuda()
                batch["segments_confs"] = segment_batch["confidences"].cuda()
                batch["segments_groups"] = segment_batch["group"].cuda()
                batch["segments_rays"] = self.train_ray_generator(batch["segments_rays"])
                self.iter_train_seg_dataloader_cnt += 1
            else:
                self.iter_train_seg_dataloader = iter(self.train_seg_dataloader)
                self.iter_train_seg_dataloader_cnt = 0
        return ray_bundle, batch
