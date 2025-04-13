import os
import numpy as np 
from PIL import Image
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
args = parser.parse_args()

source_dir = args.data_dir
target_metric_dir = os.path.join(source_dir, "depth")
os.makedirs(target_metric_dir, exist_ok=True)
for name in tqdm(sorted(os.listdir(os.path.join(source_dir, "depth_npy")))):
    depth = np.load(os.path.join(source_dir, "depth_npy", name))
    np.save(os.path.join(target_metric_dir, name.replace("_pred", "")), depth)

target_metric_dir = os.path.join(source_dir, "normal")
os.makedirs(target_metric_dir, exist_ok=True)
for name in tqdm(sorted(os.listdir(os.path.join(source_dir, "normal_colored")))):
    normal_im = np.array(Image.open(os.path.join(source_dir, "normal_colored", name)), dtype=np.float32) / 255.
    normal_im = 1.0 - normal_im
    normal_im = Image.fromarray(np.clip(normal_im * 255.0, 0, 255).astype(np.uint8))
    normal_im.save(os.path.join(target_metric_dir, name.replace("_pred_colored", "")))


# target_metric_dir = os.path.join(source_dir, "depth_vis")
# os.makedirs(target_metric_dir, exist_ok=True)
# for name in tqdm(sorted(os.listdir(os.path.join(source_dir, "depth_colored")))):
#     depth = Image.open(os.path.join(source_dir, "depth_colored", name))
#     depth.save(os.path.join(target_metric_dir, name.replace("_pred_colored", "")))