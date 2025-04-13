# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
import pdb
import os
import torchvision

from typing import Any, Dict, List, Tuple

from .sam.image_encoder import ImageEncoderViT
from .sam.mask_decoder import MaskDecoder
from .sam.prompt_encoder import PromptEncoder
from .detr import box_ops
from .detr.segmentation import dice_loss, sigmoid_focal_loss
from .detr.misc import nested_tensor_from_tensor_list, interpolate
from . import axis_ops, ilnr_loss #, pwnp_loss
from .vnl_loss import VNL_Loss
from .midas_loss import MidasLoss


class SamTransformer(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        affordance_decoder: MaskDecoder,
        depth_decoder: MaskDecoder,
        transformer_hidden_dim: int,
        backbone_name: str,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        sam_pretrained: bool = False,
        image_size: List[int] = [768, 1024],
        num_queries: int = 15,
        affordance_focal_alpha: float = 0.95,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.affordance_decoder = affordance_decoder
        
        # depth head
        self.depth_decoder = depth_decoder
        self.depth_query = nn.Embedding(2, transformer_hidden_dim)
        fov = torch.tensor(1.0)
        image_size = (768, 1024)
        focal_length = (image_size[1] / 2 / torch.tan(fov / 2)).item()
        self.vnl_loss = VNL_Loss(focal_length, focal_length, image_size)
        self.midas_loss = MidasLoss(alpha=0.1)

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        if backbone_name == 'vit_h':
            checkpoint_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'checkpoints', 'sam_vit_h_4b8939.pth')
        elif backbone_name == 'vit_l':
            checkpoint_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'checkpoints', 'sam_vit_l_0b3195.pth')
        elif backbone_name == 'vit_b':
            checkpoint_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'checkpoints', 'sam_vit_b_01ec64.pth')
        else:
            raise ValueError

        if sam_pretrained:
            with open(checkpoint_path, "rb") as f:
                state_dict = torch.load(f)
            self.load_state_dict(state_dict, strict=False)

        self.num_queries = num_queries
        self._affordance_focal_alpha = affordance_focal_alpha
        self._ignore_index = -100

    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    def freeze_layers(self, names):
        """
        Freeze layers in 'names'.
        """
        for name, param in self.named_parameters():
            for freeze_name in names:
                if freeze_name in name:
                    #print(name + ' ' + freeze_name)
                    param.requires_grad = False
    
    def forward(
        self,
        image: torch.Tensor,
        valid: torch.Tensor,
        keypoints: torch.Tensor,
        bbox: torch.Tensor,
        masks: torch.Tensor,
        movable: torch.Tensor,
        rigid: torch.Tensor,
        kinematic: torch.Tensor,
        action: torch.Tensor,
        affordance: torch.Tensor,
        affordance_map: torch.FloatTensor,
        depth: torch.Tensor,
        axis: torch.Tensor,
        fov: torch.Tensor,
        backward: bool = True,
        **kwargs,
    ):
        device = image.device
        multimask_output = False
        
        # image encoder
        # pad image to square
        h, w = image.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(image, (0, padw, 0, padh))
        image_embeddings = self.image_encoder(x)

        outputs_seg_masks = []
        outputs_movable = []
        outputs_rigid = []
        outputs_kinematic = []
        outputs_action = []
        outputs_axis = []
        outputs_boxes = []
        outputs_aff_masks = []
        outputs_depth = []
        for idx, curr_embedding in enumerate(image_embeddings):
            point_coords = keypoints[idx].unsqueeze(1)
            point_labels = torch.ones_like(point_coords[:, :, 0])
            points = (point_coords, point_labels)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )
            
            # mask decoder
            low_res_masks, iou_predictions, output_movable, output_rigid, output_kinematic, output_action, output_axis = self.mask_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            output_mask = self.postprocess_masks(
                low_res_masks,
                input_size=image.shape[-2:],
                original_size=(768, 1024),
            )
            outputs_seg_masks.append(output_mask[:, 0])
            outputs_movable.append(output_movable[:, 0])
            outputs_rigid.append(output_rigid[:, 0])
            outputs_kinematic.append(output_kinematic[:, 0])
            outputs_action.append(output_action[:, 0])
            outputs_axis.append(output_axis[:, 0])

            # convert masks to boxes for evaluation
            pred_mask_bbox = (output_mask[:, 0].clone() > 0.0).long() 
            empty_mask = pred_mask_bbox.sum(dim=-1).sum(dim=-1)
            pred_mask_bbox[empty_mask == 0] += 1
            pred_boxes = torchvision.ops.masks_to_boxes(pred_mask_bbox)
            #pred_boxes = box_ops.rescale_bboxes(pred_boxes, [1 / self._image_size[1], 1 / self._image_size[0]])
            pred_boxes = box_ops.rescale_bboxes(pred_boxes, [1 / 768, 1 / 1024])
            outputs_boxes.append(pred_boxes)

            # affordance decoder
            low_res_masks, iou_predictions = self.affordance_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            output_aff_masks = self.postprocess_masks(
                low_res_masks,
                input_size=image.shape[-2:],
                original_size=(192, 256),
            )
            outputs_aff_masks.append(output_aff_masks[:, 0])

            # depth decoder
            bs = keypoints.shape[0]
            #depth_sparse_embeddings = self.depth_query.weight.unsqueeze(0).repeat(bs, 1, 1)
            depth_sparse_embeddings = self.depth_query.weight.unsqueeze(0)
            #depth_dense_embeddings = torch.zeros((bs, 256, 64, 64)).to(dense_embeddings.device)
            depth_dense_embeddings = torch.zeros((1, 256, 64, 64)).to(dense_embeddings.device)
            low_res_masks, iou_predictions = self.depth_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=depth_sparse_embeddings,
                dense_prompt_embeddings=depth_dense_embeddings,
                multimask_output=multimask_output,
            )
            output_depth = self.postprocess_masks(
                low_res_masks,
                input_size=image.shape[-2:],
                original_size=(768, 1024),
            )
            outputs_depth.append(output_depth[:, 0])
        
        outputs_seg_masks = torch.stack(outputs_seg_masks)
        outputs_movable = torch.stack(outputs_movable)
        outputs_rigid = torch.stack(outputs_rigid)
        outputs_kinematic = torch.stack(outputs_kinematic)
        outputs_action = torch.stack(outputs_action)
        outputs_axis = torch.stack(outputs_axis)
        outputs_boxes = torch.stack(outputs_boxes)
        outputs_aff_masks = torch.stack(outputs_aff_masks)
        outputs_depth = torch.stack(outputs_depth)

 
        out = {
            'pred_boxes': outputs_boxes,
            'pred_movable': outputs_movable,
            'pred_rigid': outputs_rigid,
            'pred_kinematic': outputs_kinematic,
            'pred_action': outputs_action,
            'pred_masks': outputs_seg_masks,
            'pred_axis': outputs_axis,
            'pred_depth': outputs_depth,
            # 'pred_depth': outputs_seg_masks[:, :1].sigmoid(),
            'pred_affordance': outputs_aff_masks,
        }

        if not backward:
            return out

        # backward
        src_boxes = outputs_boxes
        target_boxes = bbox
        target_boxes = box_ops.box_xyxy_to_cxcywh(target_boxes)
        bbox_valid = bbox[:, :, 0] > -0.5
        num_boxes = bbox_valid.sum()

        out['loss_bbox'] = torch.tensor(0.0, requires_grad=True).to(device)
        out['loss_giou'] = torch.tensor(0.0, requires_grad=True).to(device)


        # affordance
        # out['loss_affordance'] = torch.tensor(0.0, requires_grad=True).to(device)
        affordance_valid = affordance[:, :, 0] > -0.5
        if affordance_valid.sum() == 0:
            out['loss_affordance'] = torch.tensor(0.0, requires_grad=True).to(device)
        else:
            src_aff_masks = outputs_aff_masks[affordance_valid]
            tgt_aff_masks = affordance_map[affordance_valid]
            src_aff_masks = src_aff_masks.flatten(1)
            tgt_aff_masks = tgt_aff_masks.flatten(1)
            loss_aff = sigmoid_focal_loss(
                src_aff_masks,
                tgt_aff_masks,
                affordance_valid.sum(),
                alpha=self._affordance_focal_alpha,
            )
            out['loss_affordance'] = loss_aff

        # axis
        axis_valid = axis[:, :, 0] > 0.0
        num_axis = axis_valid.sum()
        if num_axis == 0:
            out['loss_axis_angle'] = torch.tensor(0.0, requires_grad=True).to(device)
            out['loss_axis_offset'] = torch.tensor(0.0, requires_grad=True).to(device)
            out['loss_eascore'] = torch.tensor(0.0, requires_grad=True).to(device)
        else:
            # regress angle
            src_axis_angle = outputs_axis[axis_valid]
            src_axis_angle_norm = F.normalize(src_axis_angle[:, :2])
            src_axis_angle = torch.cat((src_axis_angle_norm, src_axis_angle[:, 2:]), dim=-1)
            target_axis_xyxy = axis[axis_valid]
            
            axis_center = target_boxes[axis_valid].clone()
            axis_center[:, 2:] = axis_center[:, :2] 
            target_axis_angle = axis_ops.line_xyxy_to_angle(target_axis_xyxy, center=axis_center)

            loss_axis_angle = F.l1_loss(src_axis_angle[:, :2], target_axis_angle[:, :2], reduction='sum') / num_axis
            loss_axis_offset = F.l1_loss(src_axis_angle[:, 2:], target_axis_angle[:, 2:], reduction='sum') / num_axis
            out['loss_axis_angle'] = loss_axis_angle
            out['loss_axis_offset'] = loss_axis_offset
            
            src_axis_xyxy = axis_ops.line_angle_to_xyxy(src_axis_angle, center=axis_center)
            target_axis_xyxy = axis_ops.line_angle_to_xyxy(target_axis_angle, center=axis_center)

            axis_eascore, _, _ = axis_ops.ea_score(src_axis_xyxy, target_axis_xyxy)
            loss_eascore = 1 - axis_eascore
            out['loss_eascore'] = loss_eascore.mean()


        loss_movable = F.cross_entropy(outputs_movable.permute(0, 2, 1), movable, ignore_index=self._ignore_index)
        if torch.isnan(loss_movable):
            loss_movable = torch.tensor(0.0, requires_grad=True).to(device)
        out['loss_movable'] = loss_movable

        loss_rigid = F.cross_entropy(outputs_rigid.permute(0, 2, 1), rigid, ignore_index=self._ignore_index)
        if torch.isnan(loss_rigid):
            loss_rigid = torch.tensor(0.0, requires_grad=True).to(device)
        out['loss_rigid'] = loss_rigid

        loss_kinematic = F.cross_entropy(outputs_kinematic.permute(0, 2, 1), kinematic, ignore_index=self._ignore_index)
        if torch.isnan(loss_kinematic):
            loss_kinematic = torch.tensor(0.0, requires_grad=True).to(device)
        out['loss_kinematic'] = loss_kinematic

        loss_action = F.cross_entropy(outputs_action.permute(0, 2, 1), action, ignore_index=self._ignore_index)
        if torch.isnan(loss_action):
            loss_action = torch.tensor(0.0, requires_grad=True).to(device)
        out['loss_action'] = loss_action

        # depth backward
        out['loss_depth'] = torch.tensor(0.0, requires_grad=True).to(device)
        out['loss_vnl'] = torch.tensor(0.0, requires_grad=True).to(device)
        # (bs, 1, H, W)
        src_depths = interpolate(outputs_depth, size=depth.shape[-2:], mode='bilinear', align_corners=False)
        src_depths = src_depths.clamp(min=0.0, max=1.0)
        tgt_depths = depth.unsqueeze(1)  # (bs, H, W)
        valid_depth = depth[:, 0, 0] > 0
        if valid_depth.any():
            src_depths = src_depths[valid_depth]
            tgt_depths = tgt_depths[valid_depth]
            depth_mask = tgt_depths > 1e-8
            midas_loss, ssi_loss, reg_loss = self.midas_loss(src_depths, tgt_depths, depth_mask)
            loss_vnl = self.vnl_loss(tgt_depths, src_depths)
            out['loss_depth'] = midas_loss
            out['loss_vnl'] = loss_vnl
        else:
            out['loss_depth'] = torch.tensor(0.0, requires_grad=True).to(device)
            out['loss_vnl'] = torch.tensor(0.0, requires_grad=True).to(device)
 
        # mask backward
        tgt_masks = masks
        src_masks = interpolate(outputs_seg_masks, size=tgt_masks.shape[-2:], mode='bilinear', align_corners=False)
        valid_mask = tgt_masks.sum(dim=-1).sum(dim=-1) > 10
        if valid_mask.sum() == 0:
            out['loss_mask'] = torch.tensor(0.0, requires_grad=True).to(device)
            out['loss_dice'] = torch.tensor(0.0, requires_grad=True).to(device)
        else:
            num_masks = valid_mask.sum()
            src_masks = src_masks[valid_mask]
            tgt_masks = tgt_masks[valid_mask]
            src_masks = src_masks.flatten(1)
            tgt_masks = tgt_masks.flatten(1)
            tgt_masks = tgt_masks.view(src_masks.shape)
            out['loss_mask'] = sigmoid_focal_loss(src_masks, tgt_masks.float(), num_masks)
            out['loss_dice'] = dice_loss(src_masks, tgt_masks, num_masks)

        return out


    def predict(
        self,
        image: torch.Tensor,
        keypoints_p,
        keypoints_n,
        original_masks,
    ):
        # device = image.device
        multimask_output = False
        
        # image encoder
        # pad image to square
        h, w = image.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        

        outputs_seg_masks = []
        outputs_movable = []
        outputs_rigid = []
        outputs_kinematic = []
        outputs_action = []
        outputs_axis = []
        outputs_boxes = []
        outputs_aff_masks = []
        outputs_depth = []
        for idx in range(image.shape[0]):
            image_idx = image[idx:idx+1].cuda()
            x = F.pad(image_idx, (0, padw, 0, padh))
            curr_embedding = self.image_encoder(x)[0]

            point_coords_p = keypoints_p[idx].unsqueeze(0)
            point_labels_p = torch.ones_like(point_coords_p[:, :, 0])
            # point_coords_n = keypoints_n[idx].unsqueeze(0)
            # point_labels_n = torch.zeros_like(point_coords_n[:, :, 0])
            # point_coords = torch.cat([point_coords_p, point_coords_n], dim=1)
            # point_labels = torch.cat([point_labels_p, point_labels_n], dim=1)
            # points = (point_coords_p, point_labels_p)
            max_iou = 0
            original_mask = original_masks[idx]
            image_size = (image.shape[2], image.shape[3])
            for positive_i in range(point_coords_p.shape[1]):
                points = (point_coords_p[:, positive_i:positive_i+1], point_labels_p[:, positive_i:positive_i+1])
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points,
                    boxes=None,
                    masks=None,
                )

                # mask decoder
                low_res_masks, iou_predictions, _output_movable, _output_rigid, _output_kinematic, _output_action, _output_axis = self.mask_decoder(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                _output_mask = self.postprocess_masks(
                    low_res_masks,
                    input_size=image.shape[-2:],
                    original_size=image.shape[-2:],
                )

                _output_mask_cpu = interpolate(_output_mask.detach().cpu(), size=image_size, mode='bilinear', align_corners=False)
                _output_mask_cpu = _output_mask_cpu.sigmoid() > 0.5

                _output_mask_cpu = _output_mask_cpu.reshape(original_mask.shape)
                inter = torch.count_nonzero(torch.logical_and(_output_mask_cpu, original_mask))
                union = torch.count_nonzero(torch.logical_or(_output_mask_cpu, original_mask))
                iou = inter / union 

                if iou > max_iou:
                    max_iou = iou

                    output_mask = _output_mask
                    output_movable = _output_movable
                    output_rigid = _output_rigid
                    output_kinematic = _output_kinematic
                    output_action = _output_action
                    output_axis = _output_axis

            outputs_seg_masks.append(output_mask[:, 0])
            outputs_movable.append(output_movable[:, 0])
            outputs_rigid.append(output_rigid[:, 0])
            outputs_kinematic.append(output_kinematic[:, 0])
            outputs_action.append(output_action[:, 0])
            outputs_axis.append(output_axis[:, 0])

            # convert masks to boxes for evaluation
            pred_mask_bbox = (output_mask[:, 0].clone() > 0.0).long() 
            empty_mask = pred_mask_bbox.sum(dim=-1).sum(dim=-1)
            pred_mask_bbox[empty_mask == 0] += 1
            pred_boxes = torchvision.ops.masks_to_boxes(pred_mask_bbox)
            #pred_boxes = box_ops.rescale_bboxes(pred_boxes, [1 / self._image_size[1], 1 / self._image_size[0]])
            pred_boxes = box_ops.rescale_bboxes(pred_boxes, [1 / 768, 1 / 1024])
            outputs_boxes.append(pred_boxes)

            # affordance decoder
            low_res_masks, iou_predictions = self.affordance_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            output_aff_masks = self.postprocess_masks(
                low_res_masks,
                input_size=image.shape[-2:],
                original_size=(int(image.shape[-2]*0.25), int(image.shape[-1]*0.25)),
            )
            outputs_aff_masks.append(output_aff_masks[:, 0])

            # depth decoder
            # bs = keypoints.shape[0]
            #depth_sparse_embeddings = self.depth_query.weight.unsqueeze(0).repeat(bs, 1, 1)
            depth_sparse_embeddings = self.depth_query.weight.unsqueeze(0)
            #depth_dense_embeddings = torch.zeros((bs, 256, 64, 64)).to(dense_embeddings.device)
            depth_dense_embeddings = torch.zeros((1, 256, 64, 64)).to(dense_embeddings.device)
            low_res_masks, iou_predictions = self.depth_decoder(
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=depth_sparse_embeddings,
                dense_prompt_embeddings=depth_dense_embeddings,
                multimask_output=multimask_output,
            )
            output_depth = self.postprocess_masks(
                low_res_masks,
                input_size=image.shape[-2:],
                original_size=image.shape[-2:],
            )
            outputs_depth.append(output_depth[:, 0])
        
        outputs_seg_masks = torch.stack(outputs_seg_masks)
        outputs_movable = torch.stack(outputs_movable)
        outputs_rigid = torch.stack(outputs_rigid)
        outputs_kinematic = torch.stack(outputs_kinematic)
        outputs_action = torch.stack(outputs_action)
        outputs_axis = torch.stack(outputs_axis)
        outputs_boxes = torch.stack(outputs_boxes)
        outputs_aff_masks = torch.stack(outputs_aff_masks)
        outputs_depth = torch.stack(outputs_depth)

        out = {
            'pred_boxes': outputs_boxes,
            'pred_movable': outputs_movable,
            'pred_rigid': outputs_rigid,
            'pred_kinematic': outputs_kinematic,
            'pred_action': outputs_action,
            'pred_masks': outputs_seg_masks,
            'pred_axis': outputs_axis,
            'pred_depth': outputs_depth,
            # 'pred_depth': outputs_seg_masks[:, :1].sigmoid(),
            'pred_affordance': outputs_aff_masks,
        }

        return out

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
