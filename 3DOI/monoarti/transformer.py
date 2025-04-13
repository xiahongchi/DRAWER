from typing import List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pdb
from torchvision.ops import roi_pool

from . import axis_ops, ilnr_loss #, pwnp_loss
from .vnl_loss import VNL_Loss
from .midas_loss import MidasLoss
from .detr.detr import MLP
from .detr.transformer import Transformer
from .detr.backbone import Backbone, Joiner
from .detr.position_encoding import PositionEmbeddingSine
from .detr.misc import nested_tensor_from_tensor_list, interpolate
from .detr import box_ops
from .detr.segmentation import (
    MHAttentionMap, MaskHeadSmallConv, dice_loss, sigmoid_focal_loss
)


class AxisHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

    def forward(self, features, hs, boxes, bbox_valid):
        roi_input = features[-2].tensors
        bs = roi_input.shape[0]
        num_queries = boxes.shape[1]

        box_ids = torch.arange(bs).unsqueeze(-1).repeat((1, num_queries)).unsqueeze(-1)
        box_ids = box_ids.to(boxes.device)
        box_tensor = torch.cat((box_ids, boxes), dim=2).reshape(-1, 5)

        rois = roi_pool(roi_input, box_tensor, (12, 16))

        axis_outputs = self.layers(rois)
        axis_outputs = axis_outputs.reshape(bs, num_queries, 3)
        return axis_outputs

class INTR(torch.nn.Module):
    """
    Implement Interaction 3D Transformer.
    """

    def __init__(
        self,
        backbone_name = 'resnet50',
        image_size = [192, 256],
        ignore_index = -100,
        num_classes = 1,
        num_queries = 15,
        freeze_backbone = False,
        transformer_hidden_dim = 256,
        transformer_dropout = 0.1,
        transformer_nhead = 8,
        transformer_dim_feedforward = 2048,
        transformer_num_encoder_layers = 6,
        transformer_num_decoder_layers = 6,
        transformer_normalize_before = False,
        transformer_return_intermediate_dec = True,
        layers_movable = 3,
        layers_rigid = 3,
        layers_kinematic = 3,
        layers_action = 3,
        layers_axis = 2,
        layers_affordance = 3,
        affordance_focal_alpha = 0.95,
        axis_bins = 30,
        depth_on = True,
    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()

        self._ignore_index = ignore_index
        self._image_size = image_size
        self._axis_bins = axis_bins
        self._affordance_focal_alpha = affordance_focal_alpha

        # backbone
        backbone_base = Backbone(backbone_name, not freeze_backbone, True, False)
        N_steps = transformer_hidden_dim // 2
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
        backbone = Joiner(backbone_base, position_embedding)
        backbone.num_channels = backbone_base.num_channels
        self.backbone = backbone

        self.transformer = Transformer(
            d_model=transformer_hidden_dim,
            dropout=transformer_dropout,
            nhead=transformer_nhead,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_num_encoder_layers,
            num_decoder_layers=transformer_num_decoder_layers,
            normalize_before=transformer_normalize_before,
            return_intermediate_dec=transformer_return_intermediate_dec,
        )
        hidden_dim = self.transformer.d_model
        self.hidden_dim = hidden_dim
        nheads = self.transformer.nhead

        self.num_queries = num_queries

        # before transformer, input_proj maps 2048 channel resnet50 output to 512-channel
        # transformer input
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)

        # query mlp maps 2d keypoint coordinates to 256-dim positional encoding
        self.query_mlp = MLP(2, hidden_dim, hidden_dim, 2)

        # bbox MLP        
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        if layers_movable > 1:
            self.movable_embed = MLP(hidden_dim, hidden_dim, 3, layers_movable)
        elif layers_movable == 1:
            self.movable_embed = nn.Linear(hidden_dim, 3)
        else:
            raise ValueError("not supported")

        if layers_rigid > 1:
            self.rigid_embed = MLP(hidden_dim, hidden_dim, 2, layers_rigid)
        elif layers_rigid == 1:
            self.rigid_embed = nn.Linear(hidden_dim, 3)
        else:
            raise ValueError("not supported")

        if layers_kinematic > 1:
            self.kinematic_embed = MLP(hidden_dim, hidden_dim, 3, layers_kinematic)
        elif layers_kinematic == 1:
            self.kinematic_embed = nn.Linear(hidden_dim, 3)
        else:
            raise ValueError("not supported")

        if layers_action > 1:
            self.action_embed = MLP(hidden_dim, hidden_dim, 3, layers_action)
        elif layers_action == 1:
            self.action_embed = nn.Linear(hidden_dim, 3)
        else:
            raise ValueError("not supported")

        if layers_axis > 1:
            self.axis_embed = MLP(hidden_dim, hidden_dim, 3, layers_axis)
        elif layers_axis == 1:
            self.axis_embed = nn.Linear(hidden_dim, 3)
        else:
            raise ValueError("not supported")

        # affordance
        if layers_affordance > 1:
            self.aff_embed = MLP(hidden_dim, hidden_dim, 2, layers_affordance)
        elif layers_affordance == 1:
            self.aff_embed = nn.Linear(hidden_dim, 2)
        else:
            raise ValueError("not supported")
        
        # affordance head
        self.aff_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.aff_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim, nheads)

        # mask head
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim, nheads)

        # depth head
        self._depth_on = depth_on
        if self._depth_on:
            self.depth_query = nn.Embedding(1, hidden_dim)
            self.depth_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
            self.depth_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim, nheads)
            self.depth_loss = ilnr_loss.MEADSTD_TANH_NORM_Loss()
            fov = torch.tensor(1.0)
            focal_length = (image_size[1] / 2 / torch.tan(fov / 2)).item()
            self.vnl_loss = VNL_Loss(focal_length, focal_length, image_size)
            self.midas_loss = MidasLoss(alpha=0.1)

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
        """
        Model forward. Set backward = False if the model is inference only.
        """
        device = image.device

        # number of queries can be different in runtime
        num_queries = keypoints.shape[1]

        # DETR forward
        samples = image
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        bs = features[-1].tensors.shape[0]
        src, mask = features[-1].decompose()
        assert mask is not None

        # sample keypoint queries from the positional embedding
        use_sine = False
        if use_sine:
            anchors = keypoints.float()
            anchors_float = anchors.clone()
            anchors_float = anchors_float.reshape(-1, 2)
            anchors_float[:, 0] = ((anchors_float[:, 0] / self._image_size[1]) - 0.5) * 2
            anchors_float[:, 1] = ((anchors_float[:, 1] / self._image_size[0]) - 0.5) * 2
            anchors_float = anchors_float.unsqueeze(1).unsqueeze(1)
            # 4x256x1x1
            keypoint_queries = F.grid_sample(
                #pos[0].repeat(self.num_queries, 1, 1, 1), 
                pos[-1].repeat(self.num_queries, 1, 1, 1), 
                anchors_float, 
                mode='nearest', 
                align_corners=True
            )
            # 4 x 10 (number of object queires) x 256
            keypoint_queries = keypoint_queries.squeeze().reshape(-1, self.num_queries, self.hidden_dim)
        else:
            # use learned MLP to map postional encoding
            anchors = keypoints.float()
            anchors_float = anchors.clone()
            anchors_float[:, :, 0] = ((anchors_float[:, :, 0] / self._image_size[1]) - 0.5) * 2
            anchors_float[:, :, 1] = ((anchors_float[:, :, 1] / self._image_size[0]) - 0.5) * 2
            keypoint_queries = self.query_mlp(anchors_float)

        # append depth_query if the model is learning depth.
        if self._depth_on:
            bs = keypoint_queries.shape[0]
            depth_query = self.depth_query.weight.unsqueeze(0).repeat(bs, 1, 1)
            keypoint_queries = torch.cat((keypoint_queries, depth_query), dim=1)

        # transformer forward
        src_proj = self.input_proj(src)
        hs, memory = self.transformer(src_proj, mask, keypoint_queries, pos[-1])

        if self._depth_on:
            depth_hs = hs[-1][:, -1:]
            ord_hs = hs[-1][:, :-1]
        else:
            ord_hs = hs[-1]

        outputs_coord = self.bbox_embed(ord_hs).sigmoid()
        outputs_movable = self.movable_embed(ord_hs)
        outputs_rigid = self.rigid_embed(ord_hs)
        outputs_kinematic = self.kinematic_embed(ord_hs)
        outputs_action = self.action_embed(ord_hs)

        # axis forward
        #bbox_valid = bbox[:, :, 0] > -0.5
        #outputs_axis = self.axis_head(features, ord_hs, box_ops.box_cxcywh_to_xyxy(outputs_coord), bbox_valid)
        outputs_axis = self.axis_embed(ord_hs).sigmoid()
        # sigmoid range is 0 to 1, we want it to be -1 to 1
        outputs_axis = (outputs_axis - 0.5) * 2
        # outputs_axis = self.axis_embed(ord_hs)

        # affordance forward
        bbox_aff = self.aff_attention(ord_hs, memory, mask=mask)
        aff_masks = self.aff_head(src_proj, bbox_aff, [features[2].tensors, features[1].tensors, features[0].tensors])
        outputs_aff_masks = aff_masks.view(bs, num_queries, aff_masks.shape[-2], aff_masks.shape[-1])

        # mask forward
        bbox_mask = self.bbox_attention(ord_hs, memory, mask=mask)
        seg_masks = self.mask_head(src_proj, bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])
        outputs_seg_masks = seg_masks.view(bs, num_queries, seg_masks.shape[-2], seg_masks.shape[-1])
        
        # depth forward
        outputs_depth = None
        if self._depth_on:
            depth_att = self.depth_attention(depth_hs, memory, mask=mask)
            depth_masks = self.depth_head(
                src_proj, 
                depth_att, 
                [features[2].tensors, features[1].tensors, features[0].tensors]
            )
            outputs_depth = depth_masks.view(bs, 1, depth_masks.shape[-2], depth_masks.shape[-1])

        out = {
            'pred_boxes': box_ops.box_cxcywh_to_xyxy(outputs_coord),
            'pred_movable': outputs_movable,
            'pred_rigid': outputs_rigid,
            'pred_kinematic': outputs_kinematic,
            'pred_action': outputs_action,
            'pred_masks': outputs_seg_masks,
            'pred_axis': outputs_axis,
            'pred_depth': outputs_depth,
            'pred_affordance': outputs_aff_masks,
        }

        if not backward:
            return out

        # backward
        src_boxes = outputs_coord
        target_boxes = bbox
        target_boxes = box_ops.box_xyxy_to_cxcywh(target_boxes)
        bbox_valid = bbox[:, :, 0] > -0.5
        num_boxes = bbox_valid.sum()
        if num_boxes == 0:
            out['loss_bbox'] = torch.tensor(0.0, requires_grad=True).to(device)
            out['loss_giou'] = torch.tensor(0.0, requires_grad=True).to(device)
        else:
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
            loss_bbox = loss_bbox * bbox_valid.unsqueeze(2) # remove invalid
            out['loss_bbox'] = loss_bbox.sum() / num_boxes
        
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes).reshape(-1, 4),
                box_ops.box_cxcywh_to_xyxy(target_boxes).reshape(-1, 4),
            )).reshape(-1, self.num_queries)
            loss_giou = loss_giou * bbox_valid # remove invalid
            out['loss_giou'] = loss_giou.sum() / num_boxes

        # affordance
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
            # classify angle
            # src_axis_angle = outputs_axis[axis_valid]
            # src_axis_theta = src_axis_angle[:, :self._axis_bins]
            # src_axis_offset = src_axis_angle[:, self._axis_bins:]
            # target_axis_xyxy = axis[axis_valid]
            # target_axis_angle = axis_ops.line_xyxy_to_angle(target_axis_xyxy)
            # target_axis_cos = target_axis_angle[:, 1]
            # target_axis_offset = target_axis_angle[:, 2]
            # target_axis_theta = torch.acos(target_axis_cos)
            # #target_axis_theta = torch.min(target_axis_theta, torch.pi - target_axis_theta)
            # target_axis_theta = (target_axis_theta / (torch.pi) * self._axis_bins).floor()#.long()
            # target_axis_theta[target_axis_theta == self._axis_bins] = self._axis_bins - 1  # edge case
            
            # loss_theta = F.cross_entropy(src_axis_theta, target_axis_theta.long())

            # target_axis_offset += np.sqrt(2) / 2
            # target_axis_offset = (target_axis_offset / np.sqrt(2) * self._axis_bins).floor()
            # target_axis_offset[target_axis_offset == self._axis_bins] = self._axis_bins - 1  # edge case
            
            
            # loss_offset = F.cross_entropy(src_axis_offset, target_axis_offset.long())
            # out['loss_axis'] = loss_theta + loss_offset

            # src_axis_theta = src_axis_theta.argmax(dim=-1) / self._axis_bins * (torch.pi)
            # src_axis_sin = torch.sin(src_axis_theta)
            # src_axis_cos = torch.cos(src_axis_theta)
            # src_axis_offset = src_axis_offset.argmax(dim=-1) / self._axis_bins * (np.sqrt(2) / 2)
            # src_axis_angle = torch.stack((src_axis_sin, src_axis_cos, src_axis_offset), dim=1)
            # src_axis_xyxy = axis_ops.line_angle_to_xyxy(src_axis_angle)
            # target_axis_xyxy = axis_ops.line_angle_to_xyxy(target_axis_angle)
            # axis_eascore, _, _ = axis_ops.ea_score(src_axis_xyxy, target_axis_xyxy)
            # loss_eascore = 1 - axis_eascore
            # out['loss_eascore'] = loss_eascore.mean()

            #pdb.set_trace()

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
            #loss_axis = F.l1_loss(src_axis_angle, target_axis_angle, reduction='sum') / num_axis

            # if 'AR_0Mi_dDnmF2Y_6_2610_25.jpg' in kwargs['img_name']:
            #     pdb.set_trace()
            out['loss_axis_angle'] = loss_axis_angle
            out['loss_axis_offset'] = loss_axis_offset
            
            src_axis_xyxy = axis_ops.line_angle_to_xyxy(src_axis_angle, center=axis_center)
            target_axis_xyxy = axis_ops.line_angle_to_xyxy(target_axis_angle, center=axis_center)

            # import matplotlib.pyplot as plt
            # j = 1
            # fig = plt.figure()
            # plt.plot(
            #     axis[axis_valid][j, 0:4:2].cpu(), 
            #     axis[axis_valid][j, 1:4:2].cpu(), 
            #     target_axis_xyxy[j, 0:4:2].cpu(), 
            #     target_axis_xyxy[j, 1:4:2].cpu(), 
            #     marker='o'
            # )
            # plt.xlim(0.0, 1.0)
            # plt.ylim(0.0, 1.0)
            # plt.savefig('vis_{}.png'.format(j))
            # plt.close(fig)

            #pdb.set_trace()
            axis_eascore, _, _ = axis_ops.ea_score(src_axis_xyxy, target_axis_xyxy)
            loss_eascore = 1 - axis_eascore
            out['loss_eascore'] = loss_eascore.mean()

            #pdb.set_trace()

            # learn xyxy
            # src_axis_xyxy = outputs_axis[-1][axis_valid]
            # target_axis_xyxy = axis[axis_valid]
            # target_axis_angle = axis_ops.line_xyxy_to_angle(target_axis_xyxy)
            # target_axis_xyxy = axis_ops.line_angle_to_xyxy(target_axis_angle)
            # loss_axis = F.l1_loss(src_axis_xyxy, target_axis_xyxy, reduction='sum') / num_axis
            # out['loss_axis'] = loss_axis
            # axis_eascore = axis_ops.ea_score(src_axis_xyxy, target_axis_xyxy)
            # loss_eascore = 1 - axis_eascore
            # out['loss_eascore'] = loss_eascore.mean()

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
        if self._depth_on:
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
