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
Semantic dataset.
"""

from typing import Dict

import torch
import numpy as np
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from torch.utils.data import Dataset


class PanopticDataset(InputDataset):
    """Dataset that returns images and semantics and masks.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        self.segment_enabled = False
        if "segments_rays" in dataparser_outputs.metadata:
            self.segment_enabled = True

    def get_metadata(self, data: Dict) -> Dict:
        # handle mask
        supp_data = {}

        return supp_data



class PanopticSegmentDataset(Dataset):

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__()
        self.segments_rays = dataparser_outputs.metadata["segments_rays"]
        self.segments_confs = dataparser_outputs.metadata["segments_confs"]
        self.segments_ones = dataparser_outputs.metadata["segments_ones"]
        dataparser_outputs.metadata = {}
        self.segment_max_rays = 512

    def __len__(self):
        return len(self.segments_rays)

    def __getitem__(self, idx):

        if self.segments_rays[idx].shape[0] > self.segment_max_rays:
            indices = np.random.choice(self.segments_rays[idx].shape[0], self.segment_max_rays, replace=False)
            segments_rays = self.segments_rays[idx][indices]
            segments_confs = self.segments_confs[idx][indices]
            segments_ones = self.segments_ones[idx][indices]
        else:
            segments_rays = self.segments_rays[idx]
            segments_confs = self.segments_confs[idx]
            segments_ones = self.segments_ones[idx]

        supp_data = {}

        supp_data["rays"] = segments_rays
        supp_data["confidences"] = segments_confs
        supp_data["group"] = segments_ones
        return supp_data

    @staticmethod
    def collate_fn(batch):
        return {
            "rays": torch.cat([x["rays"] for x in batch], dim=0),
            "confidences": torch.cat([x["confidences"] for x in batch], dim=0),
            "group": torch.cat([batch[i]['group'] * i for i in range(len(batch))], dim=0)
        }