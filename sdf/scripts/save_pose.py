import os
import pickle

from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--ckpt_dir", type=str)
parser.add_argument("--save_dir", type=str)

args = parser.parse_args()
ckpt_dir = args.ckpt_dir
save_dir = args.save_dir

load_config = Path(f"{ckpt_dir}/config.yml")

config, pipeline, checkpoint_path = eval_setup(load_config)

pose_dict = {}

train_cameras = pipeline.datamanager.train_dataset.cameras
train_images = pipeline.datamanager.train_dataset._dataparser_outputs.image_filenames

c2ws = train_cameras.camera_to_worlds
mvps = train_cameras.projections
camprojs = train_cameras.camprojs

for index in tqdm(range(mvps.shape[0])):
    stem = os.path.splitext(os.path.basename(train_images[index]))[0]
    pose_dict[stem] = {
        "c2w": c2ws[index].cpu().numpy().reshape(-1, 4),
        "mvp": mvps[index].cpu().numpy().reshape(-1, 4),
        "camproj": camprojs[index].cpu().numpy().reshape(-1, 4),
        "fx": float(train_cameras[index].fx),
        "fy": float(train_cameras[index].fy),
        "cx": float(train_cameras[index].cx),
        "cy": float(train_cameras[index].cy),
        "H": int(train_cameras[index].height),
        "W": int(train_cameras[index].width),
    }

    
eval_cameras = pipeline.datamanager.eval_dataset.cameras
eval_images = pipeline.datamanager.eval_dataset._dataparser_outputs.image_filenames

c2ws = eval_cameras.camera_to_worlds
mvps = eval_cameras.projections
camprojs = eval_cameras.camprojs

for index in tqdm(range(mvps.shape[0])):
    stem = os.path.splitext(os.path.basename(eval_images[index]))[0]
    pose_dict[stem] = {
        "c2w": c2ws[index].cpu().numpy().reshape(-1, 4),
        "mvp": mvps[index].cpu().numpy().reshape(-1, 4),
        "camproj": camprojs[index].cpu().numpy().reshape(-1, 4),
        "fx": float(eval_cameras[index].fx),
        "fy": float(eval_cameras[index].fy),
        "cx": float(eval_cameras[index].cx),
        "cy": float(eval_cameras[index].cy),
        "H": int(eval_cameras[index].height),
        "W": int(eval_cameras[index].width),
    }



with open(os.path.join(save_dir, "pose.pkl"), 'wb') as f:
    pickle.dump(pose_dict, f)
    