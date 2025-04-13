from omegaconf import DictConfig
from functools import partial
import torch

from .transformer import INTR
from .sam_transformer import SamTransformer
from .sam import ImageEncoderViT, MaskDecoder, PromptEncoder, TwoWayTransformer


def build_model(cfg: DictConfig):
    if cfg.model.name == 'transformer':
        model = INTR(
            backbone_name=cfg.model.backbone_name,
            image_size=cfg.data.image_size,
            num_queries=cfg.data.num_queries,
            freeze_backbone=cfg.optimizer.freeze_backbone,
            transformer_hidden_dim=cfg.model.transformer_hidden_dim,
            transformer_dropout=cfg.model.transformer_dropout,
            transformer_nhead=cfg.model.transformer_nhead,
            transformer_dim_feedforward=cfg.model.transformer_dim_feedforward,
            transformer_num_encoder_layers=cfg.model.transformer_num_encoder_layers,
            transformer_num_decoder_layers=cfg.model.transformer_num_decoder_layers,
            transformer_normalize_before=cfg.model.transformer_normalize_before,
            transformer_return_intermediate_dec=cfg.model.transformer_return_intermediate_dec,
            layers_movable=cfg.model.layers_movable,
            layers_rigid=cfg.model.layers_rigid,
            layers_kinematic=cfg.model.layers_kinematic,
            layers_action=cfg.model.layers_action,
            layers_axis=cfg.model.layers_axis,
            layers_affordance=cfg.model.layers_affordance,
            axis_bins=cfg.model.axis_bins,
            depth_on=cfg.train.depth_on,
        )
    elif cfg.model.name == 'sam':
        if cfg.model.backbone_name == 'vit_h':
            # sam_vit_h configs
            encoder_embed_dim = 1280
            encoder_depth = 32
            encoder_num_heads = 16
            encoder_global_attn_indexes = [7, 15, 23, 31]
        elif cfg.model.backbone_name == 'vit_l':
            encoder_embed_dim=1024
            encoder_depth=24
            encoder_num_heads=16
            encoder_global_attn_indexes=[5, 11, 17, 23]
        elif cfg.model.backbone_name == 'vit_b':
            # sam_vit_b
            encoder_embed_dim=768
            encoder_depth=12
            encoder_num_heads=12
            encoder_global_attn_indexes=[2, 5, 8, 11]
        else: 
            raise ValueError("backbone not found")

        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size

        model = SamTransformer(
            image_encoder=ImageEncoderViT(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim,
            ),
            prompt_encoder=PromptEncoder(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=(image_size, image_size),
                mask_in_chans=16,
            ),
            mask_decoder=MaskDecoder(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                properties_on=True,
            ),
            affordance_decoder=MaskDecoder(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                properties_on=False,
            ),
            depth_decoder=MaskDecoder(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                properties_on=False,
            ),
            transformer_hidden_dim=prompt_embed_dim,
            backbone_name=cfg.model.backbone_name,
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
            sam_pretrained=cfg.model.sam_pretrained,
            image_size=cfg.data.image_size,
            num_queries=cfg.data.num_queries,
            affordance_focal_alpha=cfg.optimizer.affordance_focal_alpha,
        )
    else:
        raise NotImplementedError

    return model