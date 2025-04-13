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
A pipeline that dynamically chooses the number of rays to sample.
"""

from dataclasses import dataclass, field
from typing import Type

import torch
from typing_extensions import Literal

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig


@dataclass
class PanopticPipelineConfig(VanillaPipelineConfig):
    """Dynamic Batch Pipeline Config"""

    _target: Type = field(default_factory=lambda: PanopticPipeline)


class PanopticPipeline(VanillaPipeline):
    """Pipeline with logic for changing the number of rays per batch."""

    # pylint: disable=abstract-method

    config: PanopticPipelineConfig
    datamanager: VanillaDataManager
    dynamic_num_rays_per_batch: int

    def __init__(
        self,
        config: PanopticPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        assert isinstance(
            self.datamanager, VanillaDataManager
        ), "PanopticPipelineConfig only works with VanillaDataManager."

    def get_train_loss_dict(self, step: int):
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
        if camera_opt_param_group in self.datamanager.get_param_groups():
            # Report the camera optimization metrics
            metrics_dict["camera_opt_translation"] = (
                self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
            )
            metrics_dict["camera_opt_rotation"] = (
                self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
            )

        if "segments_rays" in batch:
            model_outputs_segment = self._model.get_semantics(batch["segments_rays"])
            model_outputs["segment"] = model_outputs_segment

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

