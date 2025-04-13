import torch
import os
import numpy as np
import hydra
import logging
import submitit
import pickle
import collections
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from visdom import Visdom
import matplotlib.pyplot as plt
import pdb
import random
import torch.nn.functional as F

from monoarti.dataset import get_interaction_datasets
from monoarti.stats import Stats
from monoarti.detr.misc import nested_tensor_from_tensor_list
from monoarti.detr import box_ops
from monoarti.model import build_model
from monoarti.detr.misc import interpolate
from monoarti import axis_ops, depth_ops
from monoarti.utils import compute_kl_divergence, compute_sim


CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")
logger = logging.getLogger(__name__)


def evaluate(cfg, model, val_dataloader, accelerator, stats):
    logger.info("start validation")
    model = model.eval()
    preds = {
        'movable': [],
        'rigid': [],
        'kinematic': [],
        'action': [],
        'bbox': [],
        'mask': [],
        'affordance': [],
        'affordance_kld': [],
        'affordance_sim': [],
        'axis': [],
        'axis_se': [],
        'axis_sa': [],
        'depth_thres 1.25': [],
        'depth_thres 1.25^2': [],
        'depth_thres 1.25^3': [],
        'depth_scale': [],
        'depth_shift': [],
    }
    losses = []
    for iteration, batch in enumerate(val_dataloader):
        loss = 0.0

        with torch.no_grad():
            metrics = model(**batch)

        loss += cfg.optimizer.lbd_bbox * metrics['loss_bbox']
        loss += cfg.optimizer.lbd_giou * metrics['loss_giou']

        loss += cfg.optimizer.lbd_movable * metrics['loss_movable']
        loss += cfg.optimizer.lbd_rigid * metrics['loss_rigid']
        loss += cfg.optimizer.lbd_kinematic * metrics['loss_kinematic']
        loss += cfg.optimizer.lbd_action * metrics['loss_action']

        loss += cfg.optimizer.lbd_mask * metrics['loss_mask']
        loss += cfg.optimizer.lbd_dice * metrics['loss_dice']

        # valid
        valid = batch['valid']

        # bbox
        gt_boxes = batch['bbox']
        pred_boxes = metrics['pred_boxes']
        bbox_valid = gt_boxes[:, :, 0] > -0.5
        box_iou = torch.diag(box_ops.box_iou(pred_boxes.reshape(-1, 4), gt_boxes.reshape(-1, 4))[0]).reshape(-1, cfg.data.num_queries)
        box_iou[torch.isnan(box_iou)] = 0
        box_iou, bbox_valid = accelerator.gather_for_metrics((box_iou, bbox_valid))
        valid_iou = box_iou[bbox_valid].cpu().detach().numpy().tolist()
        preds['bbox'].extend(valid_iou)

        # mask
        tgt_masks = batch['masks']
        src_masks = metrics['pred_masks']
        # src_masks, tgt_masks = accelerator.gather_for_metrics((src_masks, tgt_masks))
        src_masks = interpolate(src_masks, size=tgt_masks.shape[-2:], mode='bilinear', align_corners=False)
        src_masks = src_masks.sigmoid() > 0.5
        inter = torch.logical_and(src_masks, tgt_masks).sum(dim=-1).sum(dim=-1)
        un = torch.logical_or(src_masks, tgt_masks).sum(dim=-1).sum(dim=-1)
        valid_mask = tgt_masks.sum(dim=-1).sum(dim=-1) > 10
        inter, un, valid_mask = accelerator.gather_for_metrics((inter, un, valid_mask))
        ious = inter[valid_mask] / un[valid_mask]
        ious = ious.cpu().detach().numpy().tolist()
        preds['mask'].extend(ious)

        # affordance
        affordance_valid = batch['affordance'][:, :, 0] > -0.5
        src_aff = metrics['pred_affordance']
        src_aff = interpolate(src_aff, size=[192, 256], mode='bilinear', align_corners=False)  # TODO
        tgt_aff = batch['affordance_map']
        src_aff, tgt_aff, affordance_valid = accelerator.gather_for_metrics((src_aff, tgt_aff, affordance_valid))
        src_aff = src_aff[affordance_valid].sigmoid()
        tgt_aff = tgt_aff[affordance_valid]
        
        aff_l1 = torch.abs(src_aff - tgt_aff).mean(dim=-1).mean(dim=-1)
        aff_kld = compute_kl_divergence(src_aff, tgt_aff)
        aff_sim = compute_sim(src_aff, tgt_aff)
        # aff_l1, aff_kld, aff_sim = accelerator.gather_for_metrics((aff_l1, aff_kld, aff_sim))
        aff_l1 = aff_l1.cpu().detach().numpy().tolist()
        aff_kld = aff_kld.cpu().detach().numpy().tolist()
        aff_sim = aff_sim.cpu().detach().numpy().tolist()
        preds['affordance'].extend(aff_l1)
        preds['affordance_kld'].extend(aff_kld)
        preds['affordance_sim'].extend(aff_sim)

        # movable
        movable_valid = batch['movable'] != -100
        pred_movable = metrics['pred_movable'].argmax(dim=-1)
        gt_movable = batch['movable']
        pred_movable, gt_movable, movable_valid = accelerator.gather_for_metrics((pred_movable, gt_movable, movable_valid))
        pred_movable = pred_movable[movable_valid]
        gt_movable = gt_movable[movable_valid]
        corr_movabale = pred_movable == gt_movable
        corr_movabale = corr_movabale.cpu().detach().numpy().tolist()
        preds['movable'].extend(corr_movabale)

        # rigid
        rigid_valid = batch['rigid'] != -100
        pred_rigid = metrics['pred_rigid'].argmax(dim=-1)
        gt_rigid = batch['rigid']
        pred_rigid, gt_rigid, rigid_valid = accelerator.gather_for_metrics((pred_rigid, gt_rigid, rigid_valid))
        pred_rigid = pred_rigid[movable_valid]
        gt_rigid = gt_rigid[movable_valid]
        corr_rigid = pred_rigid == gt_rigid
        corr_rigid = corr_rigid.cpu().detach().numpy().tolist()
        preds['rigid'].extend(corr_rigid)

        # kinematic
        kinematic_valid = batch['kinematic'] != -100
        pred_kinematic = metrics['pred_kinematic'].argmax(dim=-1)
        gt_kinematic = batch['kinematic']
        pred_kinematic, gt_kinematic, kinematic_valid = accelerator.gather_for_metrics((pred_kinematic, gt_kinematic, kinematic_valid))
        pred_kinematic = pred_kinematic[kinematic_valid]
        gt_kinematic = gt_kinematic[kinematic_valid]
        pred_kinematic, gt_kinematic = accelerator.gather_for_metrics((pred_kinematic, gt_kinematic))
        corr_kinematic = pred_kinematic == gt_kinematic
        corr_kinematic = corr_kinematic.cpu().detach().numpy().tolist()
        preds['kinematic'].extend(corr_kinematic)

        # action
        action_valid = batch['action'] != -100
        pred_action = metrics['pred_action'].argmax(dim=-1)
        gt_action = batch['action']
        pred_action, gt_action, action_valid = accelerator.gather_for_metrics((pred_action, gt_action, action_valid))
        pred_action = pred_action[action_valid]
        gt_action = gt_action[action_valid]
        pred_action, gt_action = accelerator.gather_for_metrics((pred_action, gt_action))
        corr_action = pred_action == gt_action
        corr_action = corr_action.cpu().detach().numpy().tolist()
        preds['action'].extend(corr_action)

        # depth
        pred_depth = metrics['pred_depth']
        if pred_depth is not None:
            gt_depth = batch['depth']
            pred_depth = pred_depth.clamp(min=1e-5, max=1.0)
            pred_depth = interpolate(pred_depth, size=gt_depth.shape[-2:], mode='bilinear', align_corners=False)
            pred_depth, gt_depth = accelerator.gather_for_metrics((pred_depth, gt_depth))
            for j in range(gt_depth.shape[0]):
                if (gt_depth[j] < 0).all():
                    continue
                depth_valid = (gt_depth[j] > 0)#.cpu()
                pred_depth_metric = depth_ops.recover_metric_depth(pred_depth[j, 0], gt_depth[j])
                scale, shift = depth_ops.recover_scale_shift(pred_depth[j, 0], gt_depth[j])
                pred_depth_metric = torch.as_tensor(pred_depth_metric).to(depth_valid.device)[depth_valid]
                gt_depth_i = gt_depth[j][depth_valid]
                thres = torch.max(pred_depth_metric / gt_depth_i, gt_depth_i / pred_depth_metric)
                preds['depth_thres 1.25'].append(((thres < 1.25).sum() / depth_valid.sum()).item())
                preds['depth_thres 1.25^2'].append(((thres < 1.25 ** 2).sum() / depth_valid.sum()).item())
                preds['depth_thres 1.25^3'].append(((thres < 1.25 ** 3).sum() / depth_valid.sum()).item())
                preds['depth_scale'].append(scale)
                preds['depth_shift'].append(shift)
        else:
            preds['depth_thres 1.25'].append(0.0)
            preds['depth_thres 1.25^2'].append(0.0)
            preds['depth_thres 1.25^3'].append(0.0)

        # axis
        pred_axis = metrics['pred_axis']
        gt_axis = batch['axis']
        axis_valid = gt_axis[:, :, 0] > 0.0
        pred_axis, gt_axis, axis_valid, pred_boxes = accelerator.gather_for_metrics((pred_axis, gt_axis, axis_valid, pred_boxes))
        num_axis = axis_valid.sum()
        if num_axis > 0:
            pred_axis = pred_axis[axis_valid]
            gt_axis = gt_axis[axis_valid]

            pred_axis_norm = F.normalize(pred_axis[:, :2])
            pred_axis = torch.cat((pred_axis_norm, pred_axis[:, 2:]), dim=-1)
            #axis_center = box_ops.box_xyxy_to_cxcywh(gt_boxes)[axis_valid].clone()
            axis_center = box_ops.box_xyxy_to_cxcywh(pred_boxes)[axis_valid].clone()
            axis_center[:, 2:] = axis_center[:, :2] 

            # regression
            src_axis_xyxy = axis_ops.line_angle_to_xyxy(pred_axis, center=axis_center)

            tgt_axis_angle = axis_ops.line_xyxy_to_angle(gt_axis, center=axis_center)
            tgt_axis_xyxy = axis_ops.line_angle_to_xyxy(tgt_axis_angle, center=axis_center)
            axis_eascore, axis_se, axis_sa = axis_ops.ea_score(src_axis_xyxy, tgt_axis_xyxy)
            axis_eascore = axis_eascore.cpu().detach().numpy().tolist()
            axis_se = axis_se.cpu().detach().numpy().tolist()
            axis_sa = axis_sa.cpu().detach().numpy().tolist()
            preds['axis'].extend(axis_eascore)
            preds['axis_se'].extend(axis_se)
            preds['axis_sa'].extend(axis_sa)


        # Update stats with the current metrics.
        stats.update(
            {"loss": float(loss), **metrics},
            stat_set="val",
        )

        if accelerator.is_local_main_process and iteration % cfg.stats_print_interval == 0:
            stats.print(stat_set="val")

    # accumulate evaluation results
    val_results = {}
    for metric in preds:
        pred = torch.FloatTensor(preds[metric])
        if metric in ['bbox', 'axis', 'axis_se', 'axis_sa', 'mask', 'depth_scale', 'depth_shift']:  # average iou
            score = pred.mean()
            val_results[metric] = score.item()
        elif metric in ['affordance', 'affordance_kld', 'affordance_sim']:
            score = pred.mean()
            val_results[metric] = score.item()
        else:  # accuracy
            score = (pred > 0.5).sum() / pred.shape[0]
            val_results[metric] = score.item()

    if accelerator.is_local_main_process:
        logger.info("-------------------------------------------------------")
        logger.info("validation results:")
        for metric in val_results:
            logger.info("\t{}: {}".format(metric, val_results[metric]))
        logger.info("-------------------------------------------------------")

        # upload to wandb
        accelerator.log(val_results, step=stats.epoch)

        stats.update(val_results, stat_set="val")
        stats.print(stat_set="val")

    model = model.train()

    return val_results


@hydra.main(config_path=CONFIG_DIR, config_name="defaults", version_base='1.2')
def main(cfg: DictConfig):
    try:
        # Only needed when launching on cluster with slurm
        job_env = submitit.JobEnvironment()
        job_id = job_env.job_id
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)
        hostname_first_node = (
            os.popen("scontrol show hostnames $SLURM_JOB_NODELIST").read().split("\n")[0]
        )
        logger.info("[launcher] Using the following MASTER_ADDR: {}".format(hostname_first_node))
        os.environ["MASTER_ADDR"] = hostname_first_node
   
        # use slurm job array id as port id
        # id range 42000 - 44000
        if '_' in job_id:
            slurm_id = int(job_id.split('_')[0])
            subjob_id = int(job_id.split('_')[-1])
        else:
            slurm_id = int(job_id)
            subjob_id = 0
        master_port = 42000 + (slurm_id % 2000) + int(subjob_id)

        logger.info("master port: {}".format(master_port))
        os.environ["MASTER_PORT"] = str(master_port)        
    except RuntimeError:
        logger.info("Running locally")
        job_id = "local"

    experiment_name = cfg.experiment_name + '_{}'.format(job_id)

    # Set the relevant seeds for reproducibility.
    # np.random.seed(cfg.seed)
    # torch.manual_seed(cfg.seed)
    set_seed(cfg.seed)

    # output_dir is set by hydra
    output_dir = ''
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    logger.critical("launching experiment {}".format(experiment_name))

    ddp_scaler = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=cfg.optimizer.mixed_precision,
        log_with="wandb" if not cfg.train.eval_only else None,
        kwargs_handlers=[ddp_scaler],
    )

    # Init model according to the config.
    model = build_model(cfg)

    # Init stats to None before loading.
    stats = None
    optimizer_state_dict = None
    start_epoch = 0

    # load checkpoint
    checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_path)
    if cfg.resume and os.path.isfile(checkpoint_path):
        logger.info(f"Resuming from checkpoint {checkpoint_path}.")
        if not accelerator.is_local_main_process:
            map_location = {'cuda:0': 'cuda:%d' % accelerator.local_process_index}
        else:
            # Running locally
            map_location = "cuda:0"
        
        loaded_data = torch.load(checkpoint_path, map_location=map_location)
        state_dict = loaded_data["model"]

        model.load_state_dict(state_dict, strict=False)

        # continue training: load optimizer and stats
        stats = pickle.loads(loaded_data["stats"])
        logger.info(f"   => resuming from epoch {stats.epoch}.")
        # optimizer_state_dict = loaded_data["optimizer"]
        start_epoch = stats.epoch + 1

        # stats_old = pickle.loads(loaded_data["stats"])
        # logger.info(f"   => resuming from epoch {stats_old.epoch}.")
        #optimizer_state_dict = loaded_data["optimizer"]
        # start_epoch = stats_old.epoch + 1
    else:
        logger.info("Start from scratch.")

    # freeze layers
    if len(cfg.model.freeze_layers) > 0:
        logger.info("freezing layers: {}".format(cfg.model.freeze_layers))
        model.freeze_layers(cfg.model.freeze_layers)

    # reset checkpoint path to output_dir
    checkpoint_path = os.path.join(output_dir, 'checkpoints', 'checkpoint.pth')

    # save config to file
    if not os.path.isfile(os.path.join(output_dir, "config.yaml")):
        OmegaConf.save(config=cfg, f=os.path.join(output_dir, "config.yaml"))
    accelerator.init_trackers("interaction", config=OmegaConf.to_container(cfg))
    
    # Initialize the optimizer.
    if cfg.optimizer.lr_backbone < cfg.optimizer.lr:
        logger.info("using different lr for backbone and transformer!")
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.optimizer.lr_backbone,
            },
        ]
    else:
        param_dicts = model.parameters()

    if cfg.optimizer.name == 'Adam':
        optimizer = torch.optim.Adam(
            param_dicts,
            lr=cfg.optimizer.lr,
        )
    elif cfg.optimizer.name == 'AdamW':
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=cfg.optimizer.lr,
            weight_decay=1e-4,
        )
    else:
        raise NotImplementedError

    # Load the optimizer state dict in case we are resuming.
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        
    optimizer.last_epoch = start_epoch

    # Init the stats object.
    if stats is None:
        stats = Stats([
            "loss",
            'loss_movable', 'loss_rigid', 'loss_kinematic', 'loss_action',
            'loss_bbox', 'loss_giou',
            'loss_affordance',
            'loss_mask', 'loss_dice',
            'loss_axis', 'loss_eascore',
            'loss_axis_angle', 'loss_axis_offset',
            'loss_depth', 'loss_vnl',
            'movable', 'rigid', 'kinematic',
            'action', 'bbox', 'axis', 'affordance', 'mask',
            'affordance_10', 'affordance_20', 'affordance_50',
            'depth_thres 1.25',
            'depth_thres 1.25^2',
            'depth_thres 1.25^3',
            'sec/it', 'accuracy',
        ])

    # Learning rate scheduler setup.
    if cfg.optimizer.lr_scheduler_name == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=cfg.optimizer.lr_scheduler_step_size,
            gamma=cfg.optimizer.lr_scheduler_gamma,
        )
    elif cfg.optimizer.lr_scheduler_name == 'ExpLR':
        # original NeRF lr decay
        def lr_lambda(epoch):
            return cfg.optimizer.lr_scheduler_gamma ** (
                epoch / cfg.optimizer.lr_scheduler_step_size
            )
        # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
        )
    else:
        raise NotImplementedError

    # Init the visualization visdom env.
    # if accelerator.is_main_process and cfg.visualization.visdom and not cfg.train.eval_only:
    #     viz = Visdom(
    #         server=cfg.visualization.visdom_server,
    #         port=cfg.visualization.visdom_port,
    #         use_incoming_socket=False,
    #         env=experiment_name,
    #     )
    # else:
    #     viz = None
    viz = None

    train_dataset, val_dataset, test_dataset = get_interaction_datasets(
        train_dataset_names=cfg.data.train_dataset_names,
        val_dataset_names=cfg.data.val_dataset_names,
        test_dataset_names=cfg.data.test_dataset_names,
        image_size=cfg.data.image_size,
        output_size=cfg.data.output_size,
        num_views=cfg.train.num_views,
        load_depth=cfg.train.depth_on,
        affordance_radius=cfg.data.affordance_radius,
        num_queries=cfg.data.num_queries,
        bbox_to_mask=cfg.data.bbox_to_mask,
    )

    logger.info("train has {} examples".format(len(train_dataset)))
    logger.info("val has {} examples".format(len(val_dataset)))  
    logger.info("test has {} examples".format(len(test_dataset)))  

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        #collate_fn=collate_fn,
    )

    print("data split: {}".format(cfg.test.split))
    if cfg.test.split == 'train':
        val_dataset = train_dataset
    elif cfg.test.split == 'val':
        pass
    elif cfg.test.split == 'test':
        val_dataset = test_dataset
    else:
        raise NotImplementedError

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )

    # Prepare the model for accelerate and move to the relevant device
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # Set the model to the training mode.
    model.train()

    # evaluation only
    if cfg.train.eval_only:
        stats.new_epoch()
        evaluate(cfg, model, val_dataloader, accelerator, stats)
        return

    # Run the main training loop.
    for epoch in range(start_epoch, cfg.optimizer.max_epochs):
        stats.new_epoch()  # Init a new epoch.

        # Training
        for iteration, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = 0.0
            metrics = model(**batch)

            loss += cfg.optimizer.lbd_bbox * metrics['loss_bbox']
            loss += cfg.optimizer.lbd_giou * metrics['loss_giou']

            loss += cfg.optimizer.lbd_mask * metrics['loss_mask']
            loss += cfg.optimizer.lbd_mask * metrics['loss_dice']
            #loss += cfg.optimizer.lbd_dice * metrics['loss_dice']

            loss += cfg.optimizer.lbd_movable * metrics['loss_movable']
            loss += cfg.optimizer.lbd_rigid * metrics['loss_rigid']
            loss += cfg.optimizer.lbd_kinematic * metrics['loss_kinematic']
            loss += cfg.optimizer.lbd_action * metrics['loss_action']

            loss += cfg.optimizer.lbd_affordance * metrics['loss_affordance']

            loss += cfg.optimizer.lbd_axis * metrics['loss_axis_angle']
            loss += cfg.optimizer.lbd_axis_offset * metrics['loss_axis_offset']
            loss += cfg.optimizer.lbd_eascore * metrics['loss_eascore']

            loss += cfg.optimizer.lbd_depth * metrics['loss_depth']
            loss += cfg.optimizer.lbd_vnl * metrics['loss_vnl']

            accelerator.backward(loss)

            if cfg.optimizer.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.clip_max_norm)
            # if accelerator.sync_gradients and cfg.optimizer.clip_max_norm > 0:
            #     accelerator.clip_grad_value_(model.parameters(), cfg.optimizer.clip_max_norm)

            optimizer.step()

            # Update stats with the current metrics.
            stats.update(
                {"loss": float(loss), **metrics},
                stat_set="train",
            )

            if accelerator.is_local_main_process and iteration % cfg.stats_print_interval == 0:
                stats.print(stat_set="train")

            # if iteration > 20:
            #     break

        lr_scheduler.step()

        # validation
        # accelerator.wait_for_everyone()
        # if (
        #     accelerator.is_main_process
        #     and epoch % cfg.validation_epoch_interval == 0
        # ):
        #     evaluate(cfg, model, val_dataloader, accelerator, stats)

        #     if accelerator.is_main_process and viz is not None:
        #         # Plot that loss curves into visdom.
        #         stats.plot_stats(
        #             viz=viz,
        #             visdom_env=experiment_name,
        #             plot_file=None,
        #         )
        if epoch % cfg.validation_epoch_interval == 0:
            evaluate(cfg, model, val_dataloader, accelerator, stats)


        # Checkpoint.
        accelerator.wait_for_everyone()
        if (
            accelerator.is_main_process
            and epoch % cfg.checkpoint_epoch_interval == 0
        ):
            logger.info(f"Storing checkpoint {checkpoint_path}.")
            data_to_store = {
                "model": accelerator.unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "stats": pickle.dumps(stats),
            }
            torch.save(data_to_store, checkpoint_path)

            epoch_checkpoint_path = checkpoint_path.replace('.pth', '_{}.pth'.format(epoch))
            logger.info(f"Storing checkpoint {epoch_checkpoint_path}.")
            torch.save(data_to_store, epoch_checkpoint_path)


if __name__=="__main__":
    main()