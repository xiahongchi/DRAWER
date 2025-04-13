import os
import torch

import numpy as np

# visualization
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from PIL import Image

# GIF
import imageio.v2 as imageio

# customized
import sys

sys.path.append(".")

from .constants import *
# from .camera_helper import polar_to_xyz


def visualize_quad_mask(mask_image_dir, quad_mask_tensor, view_idx, view_score, device):
    quad_mask_tensor = quad_mask_tensor.unsqueeze(-1).repeat(1, 1, 1, 3)
    quad_mask_image_tensor = torch.zeros_like(quad_mask_tensor)

    for idx in PALETTE:
        selected = quad_mask_tensor[quad_mask_tensor == idx].reshape(-1, 3)
        selected = torch.FloatTensor(PALETTE[idx]).to(device).unsqueeze(0).repeat(selected.shape[0], 1)

        quad_mask_image_tensor[quad_mask_tensor == idx] = selected.reshape(-1)

    quad_mask_image_np = quad_mask_image_tensor[0].cpu().numpy().astype(np.uint8)
    quad_mask_image = Image.fromarray(quad_mask_image_np).convert("RGB")
    quad_mask_image.save(os.path.join(mask_image_dir, "{}_quad_{:.5f}.png".format(view_idx, view_score)))


def visualize_outputs(output_dir, init_image_dir, mask_image_dir, inpainted_image_dir, num_views):
    # subplot settings
    num_col = 3
    num_row = 1
    subplot_size = 4

    summary_image_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_image_dir, exist_ok=True)

    # graph settings
    print("=> visualizing results...")
    for view_idx in range(num_views):
        plt.switch_backend("agg")
        fig = plt.figure(dpi=100)
        fig.set_size_inches(subplot_size * num_col, subplot_size * (num_row + 1))
        fig.set_facecolor('white')

        # rendering
        plt.subplot2grid((num_row, num_col), (0, 0))
        plt.imshow(Image.open(os.path.join(init_image_dir, "{}.png".format(view_idx))))
        plt.text(0, 0, "Rendering", fontsize=16, color='black', backgroundcolor='white')
        plt.axis('off')

        # mask
        plt.subplot2grid((num_row, num_col), (0, 1))
        plt.imshow(Image.open(os.path.join(mask_image_dir, "{}_project.png".format(view_idx))))
        plt.text(0, 0, "Project Mask", fontsize=16, color='black', backgroundcolor='white')
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

        # inpainted
        plt.subplot2grid((num_row, num_col), (0, 2))
        plt.imshow(Image.open(os.path.join(inpainted_image_dir, "{}.png".format(view_idx))))
        plt.text(0, 0, "Inpainted", fontsize=16, color='black', backgroundcolor='white')
        plt.axis('off')

        plt.savefig(os.path.join(summary_image_dir, "{}.png".format(view_idx)), bbox_inches="tight")
        fig.clf()

    # generate GIF
    images = [imageio.imread(os.path.join(summary_image_dir, "{}.png".format(view_idx))) for view_idx in
              range(num_views)]
    imageio.mimsave(os.path.join(summary_image_dir, "output.gif"), images, duration=1)

    print("=> done!")

