import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import cv2
from PIL import Image

from .visualizer import Visualizer


movable_imap = {
    0: 'one_hand',
    1: 'two_hands',
    2: 'fixture',
    -100: 'n/a',
}

rigid_imap = {
    1: 'yes',
    0: 'no',
    2: 'bad',
    -100: 'n/a',
}

kinematic_imap = {
    0: 'freeform',
    1: 'rotation',
    2: 'translation',
    -100: 'n/a'
}

action_imap = {
    0: 'free',
    1: 'pull',
    2: 'push',
    -100: 'n/a',
}


def draw_properties(
    output_path,
    movable,
    rigid,
    kinematic,
    action,
):
    fig = plt.figure()
    vis_phy = np.ones((80, 60, 3), dtype=np.uint8) * 255
    plt.imshow(vis_phy)
    fig.axes[0].add_patch(
        mpl.patches.FancyBboxPatch(
            (10, 10), 
            40, 
            60, 
            ec=np.array([79, 129, 189]) / 255.0, 
            boxstyle=mpl.patches.BoxStyle("Round", pad=5),
            fill=False,
            linewidth=5,
        )
    )
    fontsize = 20

    is_fixture = False
    if movable_imap[movable] == 'one_hand':
        movable_text = "Move: 1 hand"
    elif movable_imap[movable] == 'two_hands':
        movable_text = "Move: 2 hands"
    else:
        is_fixture = True
        movable_text = 'Move: Fixture'
    plt.text(30, 35, movable_text, fontsize=fontsize, horizontalalignment='center')

    if rigid_imap[rigid] == 'yes' and not is_fixture:
        rigid_text = 'Rigid: Yes'
    else:
        rigid_text = 'Rigid: No'
    plt.text(30, 20, rigid_text, fontsize=fontsize, horizontalalignment='center')

    if kinematic_imap[kinematic] == 'rotation' and not is_fixture:
        kinematic_text = 'Motion: Rot'
    elif kinematic_imap[kinematic] == 'translation' and not is_fixture:
        kinematic_text = 'Motion: Trans'
    else:
        kinematic_text = 'Motion: Free'
    plt.text(30, 50, kinematic_text, fontsize=fontsize, horizontalalignment='center')

    if action_imap[action] == 'pull' and not is_fixture:
        action_text = 'Action: Pull'
    elif action_imap[action] == 'push' and not is_fixture:
        action_text = 'Action: Push'
    else:
        action_text = 'Action: Free'
    plt.text(30, 65, action_text, fontsize=fontsize, horizontalalignment='center')

    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def draw_localization(rgb, output_path, bbox, mask, axis, colors=None, alpha=0.5):
    if colors is None:
        # colors = np.array([[246, 95, 16], ]) / 255.0  # orange
        #colors = np.array([[227, 13, 255], ]) / 255.0  # purple
        colors = np.array([[255, 53, 53], ]) / 255.0  # purple


    pred_entry = {
        'keypoint': None,
        'bbox': bbox,
        'mask': mask,
        'affordance': None,
        'move': None,
        'rigid': None,
        'kinematic': None,
        'pull_or_push': None,
        'axis': axis,
    }
    instances = [pred_entry]

    vis = Visualizer(rgb)
    vis.overlay_instances(instances, assigned_colors=colors, alpha=alpha)
    vis.output.save(output_path)


def draw_affordance(rgb, output_path, affordance, alpha=0.5):
    # normalize affordance
    aff_min = affordance.min()
    aff_max = affordance.max()
    affordance = (affordance - aff_min) / (aff_max - aff_min)

    # convert rgb to gray scale for better visualization
    rgb_gray = rgb.copy()
    # rgb_gray[:, :, 0] = rgb.mean(axis=2)
    # rgb_gray[:, :, 1] = rgb.mean(axis=2)
    # rgb_gray[:, :, 2] = rgb.mean(axis=2)

    heatmap_img = cv2.applyColorMap((affordance * 255.0).astype(np.uint8), cv2.COLORMAP_HOT)[:, :, ::-1]
    vis = cv2.addWeighted(heatmap_img, alpha, rgb_gray, 1-alpha, 0)
    Image.fromarray(vis).save(output_path)