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
Implementation of BakedSDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
import torch.nn.functional as F

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import interlevel_loss
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.models.volsdf import VolSDFModel, VolSDFModelConfig
from nerfstudio.utils import colormaps
from nerfstudio.model_components.losses import (
    monosdf_normal_loss,
)

@dataclass
class BakedSDFModelConfig(VolSDFModelConfig):
    """BakedSDF Model Config"""

    _target: Type = field(default_factory=lambda: BakedSDFFactoModel)
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for the proposal network."""
    num_neus_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    proposal_update_every: int = 5
    """Sample every n steps after the warmup"""
    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""
    num_proposal_iterations: int = 2
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 64},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256},
        ]
    )
    """Arguments for the proposal density fields."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
    use_anneal_beta: bool = True
    """whether to use anneal beta"""
    beta_anneal_max_num_iters: int = 250000
    """Max num iterations for the annealing of beta in laplacian density."""
    beta_anneal_init: float = 0.1
    """Initial value of beta for the annealing of beta in laplacian density."""
    beta_anneal_end: float = 0.001
    """End value of beta for the annealing of beta in laplacian density."""
    use_anneal_eikonal_weight: bool = False
    """whether to use annealing for eikonal loss weight"""
    eikonal_anneal_max_num_iters: int = 250000
    """Max num iterations for the annealing of beta in laplacian density."""
    use_spatial_varying_eikonal_loss: bool = False
    """whether to use different weight of eikonal loss based the points norm, farway points have large weights"""
    eikonal_loss_mult_start: float = 0.01
    eikonal_loss_mult_end: float = 0.1
    eikonal_loss_mult_slop: float = 2.0

def compute_scale_and_shift(prediction, target):
    dr = prediction.reshape(-1, 1)
    dr = torch.cat((dr, torch.ones_like(dr).to(dr.device)), -1).unsqueeze(-1) # (N, 2, 1)
    left_part = torch.inverse(torch.sum(dr @ dr.transpose(1, 2), dim=0)).reshape(2, 2) # (2, 2)
    right_part = torch.sum(dr*target.reshape(-1, 1, 1), dim=0).reshape(2, 1)
    rs = left_part @ right_part
    rs = rs.reshape(2)
    return rs[0], rs[1]

def compute_scale_and_shift_batch(prediction, target):
    # prediction: (B, N), N = 32
    # target: (B, N)
    B, N = prediction.shape
    dr = prediction.unsqueeze(-1) # (B, N, 1)
    dr = torch.cat((dr, torch.ones_like(dr).to(dr.device)), dim=-1).reshape(-1, 2, 1)  # (BxN, 2, 1)
    dr_sq = torch.sum((dr @ dr.transpose(1, 2)).reshape(B, N, 2, 2), dim=1) # (B, 2, 2)
    left_part = torch.inverse(dr_sq).reshape(B, 2, 2) # (B, 2, 2)
    right_part = torch.sum((dr.reshape(B, N, 2, 1))*(target.reshape(B, N, 1, 1)), dim=1).reshape(B, 2, 1)
    rs = left_part @ right_part # (B, 2, 1)
    rs = rs.reshape(B, 2)
    return rs[:, 0], rs[:, 1]

class BakedSDFFactoModel(VolSDFModel):
    """BakedSDF model

    Args:
        config: BakedSDF configuration to instantiate model
    """

    config: BakedSDFModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb, spatial_distortion=self.scene_contraction, **prop_net_args
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=self.scene_contraction,
                    **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # update proposal network every iterations
        update_schedule = lambda step: -1

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_neus_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            use_uniform_sampler=False,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.config.use_anneal_beta:
            # don't optimize beta in laplace density if use annealing beta
            param_groups["fields"] = [
                n_p[1] for n_p in filter(lambda n_p: "laplace_density" not in n_p[0], self.field.named_parameters())
            ]
        else:
            param_groups["fields"] = list(self.field.parameters())

        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())

        if self.config.background_model != "none":
            param_groups["field_background"] = list(self.field_background.parameters())
        else:
            param_groups["field_background"] = list(self.field_background)
            
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )

        if self.config.use_anneal_beta:
            # anneal the beta of volsdf before each training iterations
            M = self.config.beta_anneal_max_num_iters
            beta_init = self.config.beta_anneal_init
            beta_end = self.config.beta_anneal_end

            def set_beta(step):
                # bakedsdf's beta schedule
                train_frac = np.clip(step / M, 0, 1)
                beta = beta_init / (1 + (beta_init - beta_end) / beta_end * (train_frac**0.8))
                self.field.laplace_density.beta.data[...] = beta

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_beta,
                )
            )

        if self.config.use_anneal_eikonal_weight:
            # anneal the beta of volsdf before each training iterations
            K = self.config.eikonal_anneal_max_num_iters
            weight_init = 0.01
            weight_end = 0.1

            def set_weight(step):
                # bakedsdf's beta schedule
                train_frac = np.clip(step / K, 0, 1)
                mult = weight_end / (1 + (weight_end - weight_init) / weight_init * ((1.0 - train_frac) ** 10))
                self.config.eikonal_loss_mult = mult

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_weight,
                )
            )

        return callbacks

    def sample_and_forward_field(self, ray_bundle: RayBundle):
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        # TODO only forward the points that are inside the sphere
        field_outputs = self.field(ray_samples)
        field_outputs[FieldHeadNames.ALPHA] = ray_samples.get_alphas(field_outputs[FieldHeadNames.DENSITY])

        if self.config.background_model != "none":
            field_outputs = self.forward_background_field_and_merge(ray_samples, field_outputs)

        weights = ray_samples.get_weights_from_alphas(field_outputs[FieldHeadNames.ALPHA])

        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "weights_list": weights_list,
            "ray_samples_list": ray_samples_list,
        }

        return samples_and_field_outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])

        if self.training:
            # eikonal loss
            grad_theta = outputs["eik_grad"]
            # s3im loss
            if self.config.s3im_loss_mult > 0:
                loss_dict["s3im_loss"] = self.s3im_loss(image, outputs["rgb"]) * self.config.s3im_loss_mult
            if self.config.use_spatial_varying_eikonal_loss:

                points_norm = outputs["points_norm"][..., 0]
                points_weights = torch.where(points_norm <= 1, torch.ones_like(points_norm), points_norm)

                # shortcut
                weight_init = self.config.eikonal_loss_mult_start
                weight_end = self.config.eikonal_loss_mult_end
                slop = self.config.eikonal_loss_mult_slop

                points_weights = weight_end / (
                    1 + (weight_end - weight_init) / weight_init * ((2.0 - points_weights) ** slop)
                )

                loss_dict["eikonal_loss"] = (((grad_theta.norm(2, dim=-1) - 1) ** 2) * points_weights).mean()
            else:
                loss_dict["eikonal_loss"] = (
                    (grad_theta.norm(2, dim=-1) - 1) ** 2
                ).mean() * self.config.eikonal_loss_mult

            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )

            # monocular normal loss
            if "normal" in batch and self.config.mono_normal_loss_mult > 0.0:
                normal_gt = batch["normal"].to(self.device)
                normal_pred = outputs["normal"]
                loss_dict["normal_loss"] = (
                        monosdf_normal_loss(normal_pred, normal_gt) * self.config.mono_normal_loss_mult
                )

            # monocular depth loss
            if "depth" in batch and self.config.mono_depth_loss_mult > 0.0:
                # TODO check it's true that's we sample from only a single image
                # TODO only supervised pixel that hit the surface and remove hard-coded scaling for depth
                depth_gt = batch["depth"].to(self.device)[..., None]
                depth_pred = outputs["depth"]

                patch_size = 128 # N

                depth_gt = depth_gt.reshape(-1, patch_size) # (B, N)
                depth_pred = depth_pred.reshape(-1, patch_size) # (B, N)
                w, q = compute_scale_and_shift_batch(depth_pred, depth_gt)

                w = w.reshape(-1, 1)
                q = q.reshape(-1, 1)

                # w: (B, 1); q: (B, 1)
                diff = ((w * depth_pred + q) - depth_gt) ** 2
                diff = torch.clip(diff, max=1)
                loss_dict["depth_loss"] = (
                        diff.mean() * self.config.mono_depth_loss_mult
                )

                # mask = torch.ones_like(depth_gt).reshape(1, 32, -1).bool()
                # depth_gt = depth_gt[mask]
                # depth_pred = depth_pred[mask]
                #
                #

                # loss_dict["depth_loss"] = (
                #         self.depth_loss(depth_pred.reshape(1, 32, -1), (depth_gt * 50 + 0.5).reshape(1, 32, -1), mask)
                #         * self.config.mono_depth_loss_mult
                # )

            if "fg_mask" in batch and self.config.fg_mask_loss_mult > 0.0:
                fg_label = batch["fg_mask"].float().to(self.device)
                accu = outputs["accumulation"].sum(dim=1).clip(1e-3, 1.0 - 1e-3)
                loss_dict["fg_mask_loss"] = (
                    torch.nn.functional.mse_loss(accu, fg_label.reshape(accu.shape)) * self.config.fg_mask_loss_mult
                )

        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    def get_metrics_dict(self, outputs, batch) -> Dict:
        metric_dict = super().get_metrics_dict(outputs, batch)
        metric_dict["eikonal_loss_mult"] = self.config.eikonal_loss_mult
        return metric_dict
