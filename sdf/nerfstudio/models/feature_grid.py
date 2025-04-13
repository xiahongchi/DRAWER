import math
from dataclasses import dataclass, field
from typing import Optional, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import (
    NeRFEncoding,
    PeriodicVolumeEncoding,
    TensorVMEncoding,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, FieldConfig

import tinycudann as tcnn

@dataclass
class FeatureGridConfig(FieldConfig):
    _target: Type = field(default_factory=lambda: FeatureGrid)
    num_layers: int = 8
    """Number of layers for geometric network"""
    hidden_dim: int = 512
    """Number of hidden dimension of geometric network"""
    feat_dim: int = 256
    """Dimension of geometric feature"""
    num_levels: int = 16
    """number of levels for multi-resolution hash grids"""
    max_res: int = 2048
    """max resolution for multi-resolution hash grids"""
    base_res: int = 16
    """base resolution for multi-resolution hash grids"""
    log2_hashmap_size: int = 19
    """log2 hash map size for multi-resolution hash grids"""
    hash_features_per_level: int = 2
    """number of features per level for multi-resolution hash grids"""
    hash_smoothstep: bool = True
    """whether to use smoothstep for multi-resolution hash grids"""
    position_encoding_max_degree: int = 8
    """positional encoding max degree"""
    use_position_encoding: bool = True
    """whether to use positional encoding as input for geometric network"""
    off_axis: bool = True
    """whether to use off axis encoding from mipnerf360"""
    weight_norm: bool = True
    """Whether to use weight norm for linear laer"""

class FeatureGrid(nn.Module):
    config: FeatureGridConfig

    def __init__(self,
                 config: FeatureGridConfig,
                 aabb,
                 spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion

        self.num_levels = self.config.num_levels
        self.max_res = self.config.max_res
        self.base_res = self.config.base_res
        self.log2_hashmap_size = self.config.log2_hashmap_size
        self.features_per_level = self.config.hash_features_per_level
        smoothstep = self.config.hash_smoothstep
        self.growth_factor = np.exp((np.log(self.max_res) - np.log(self.base_res)) / (self.num_levels - 1))

        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.num_levels,
                "n_features_per_level": self.features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": self.base_res,
                "per_level_scale": self.growth_factor,
                "interpolation": "Smoothstep" if smoothstep else "Linear",
            },
        )
        self.hash_encoding_mask = torch.ones(
            self.num_levels * self.features_per_level,
            dtype=torch.float32,
        )

        # we concat inputs position ourselves
        self.position_encoding = NeRFEncoding(
            in_dim=3,
            num_frequencies=self.config.position_encoding_max_degree,
            min_freq_exp=0.0,
            max_freq_exp=self.config.position_encoding_max_degree - 1,
            include_input=False,
            off_axis=self.config.off_axis,
        )

        # TODO move it to field components
        # MLP with geometric initialization
        dims = [self.config.hidden_dim for _ in range(self.config.num_layers)]
        in_dim = 3 + self.position_encoding.get_out_dim() + self.encoding.n_output_dims
        dims = [in_dim] + dims + [self.config.feat_dim]
        self.num_layers = len(dims)
        # TODO check how to merge skip_in to config
        self.skip_in = [4]

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if self.config.weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "glin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.relu = nn.ReLU()

    def forward_feat_network(self, inputs):
        """forward the geonetwork"""
        # TODO normalize inputs depending on the whether we model the background or not
        positions = (inputs + 2.0) / 4.0
        # positions = (inputs + 1.0) / 2.0
        feature = self.encoding(positions)
        # mask feature
        feature = feature * self.hash_encoding_mask.to(feature.device)

        pe = self.position_encoding(inputs)
        if not self.config.use_position_encoding:
            pe = torch.zeros_like(pe)

        inputs = torch.cat((inputs, pe, feature), dim=-1)

        x = inputs

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "glin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
        return x

    def get_outputs(self, ray_samples: RaySamples):
        """compute output of ray samples"""
        outputs = {}

        inputs = ray_samples.frustums.get_start_positions()
        inputs = inputs.view(-1, 3)

        if self.spatial_distortion is not None:
            inputs = self.spatial_distortion(inputs)
        points_norm = inputs.norm(dim=-1)
        # compute gradient in constracted space
        inputs.requires_grad_(True)

        feat = self.forward_feat_network(inputs)

        feat = feat.view(*ray_samples.frustums.directions.shape[:-1], -1)

        outputs.update(
            {
                "feat": feat,
            }
        )

        return outputs

    def get_features(self, points: torch.Tensor):
        inputs = points.reshape(-1, 3)
        if self.spatial_distortion is not None:
            inputs = self.spatial_distortion(inputs)

        feat = self.forward_feat_network(inputs)
        return feat

    def forward(self, ray_samples: RaySamples):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        field_outputs = self.get_outputs(ray_samples)
        return field_outputs