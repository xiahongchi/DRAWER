import argparse
import os
import sys

import numpy as np
import json
import torch
from PIL import Image
import gc
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything_local"))


# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


# segment anything
from segment_anything_local import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import psutil

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    model = model.eval()
    return model

def calculate_intersection_over_union(box_a, box_b):
    # Extract box coordinates
    cx_a, cy_a, dx_a, dy_a = box_a
    cx_b, cy_b, dx_b, dy_b = box_b
    
    # Calculate box coordinates
    x1_a, y1_a = cx_a - dx_a, cy_a - dy_a  # Top-left corner of box A
    x2_a, y2_a = cx_a + dx_a, cy_a + dy_a  # Bottom-right corner of box A
    x1_b, y1_b = cx_b - dx_b, cy_b - dy_b  # Top-left corner of box B
    x2_b, y2_b = cx_b + dx_b, cy_b + dy_b  # Bottom-right corner of box B
    
    # Calculate intersection coordinates
    x1_intersect = max(x1_a, x1_b)
    y1_intersect = max(y1_a, y1_b)
    x2_intersect = min(x2_a, x2_b)
    y2_intersect = min(y2_a, y2_b)
    
    # Calculate width and height of intersection rectangle
    width_intersect = max(0, x2_intersect - x1_intersect)
    height_intersect = max(0, y2_intersect - y1_intersect)
    
    # Calculate area of intersection
    intersection_area = width_intersect * height_intersect
    
    # Calculate area of each bounding box
    area_a = (2 * dx_a) * (2 * dy_a)
    area_b = (2 * dx_b) * (2 * dy_b)
    
    # Calculate IoU
    iou = intersection_area / (area_a + area_b - intersection_area)
    
    return iou, intersection_area / area_a, intersection_area / area_b

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    phrases = [elem.strip(" ") for elem in caption.split('.') if len(elem.strip(" ")) > 0 ]
    # build pred
    pred_phrases = []
    boxes_filt_final = []
    logits_filt_final = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)

        if False and pred_phrase.strip(" ") not in phrases:
            continue 
        else:
            boxes_filt_final.append(box)
            if with_logits:
                logits_filt_final.append(logit.max().item())

        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
    if len(boxes_filt_final) == 0:
        return None, None
    boxes_filt_final = torch.stack(boxes_filt_final, dim=0)

    # calculate_intersection_over_union
    overlap_box_pairs = []
    inside_boxes = []
    for i in range(boxes_filt_final.size(0)):
        for j in range(i+1, boxes_filt_final.size(0)):
            iou_ij, inter_i, inter_j = calculate_intersection_over_union(boxes_filt_final[i], boxes_filt_final[j])
            # if iou_ij > 0.5:
            #     overlap_box_pairs.append((i, j))
            if inter_i > 0.9:
                inside_boxes.append((j))
            if inter_j > 0.9:
                inside_boxes.append((i))
    
    selected_idxs = list(range(boxes_filt_final.size(0)))
    # for i, j in overlap_box_pairs:
    #     if logits_filt_final[i] > logits_filt_final[j]:
    #         if j in selected_idxs:
    #             selected_idxs.remove(j)
    for i in inside_boxes:
        if i in selected_idxs:
            selected_idxs.remove(i)

    if len(selected_idxs) == 0:
        return None, None
    boxes_filt_final = torch.stack([elem for i, elem in enumerate(boxes_filt_final) if i in selected_idxs], dim=0)
    pred_phrases = [f"{elem}" for i, elem in enumerate(pred_phrases) if i in selected_idxs]
    pred_phrases = [f"v={i+1} {elem}" for i, elem in enumerate(pred_phrases)]
    return boxes_filt_final, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label)

def show_mask_and_box(mask, box, label, ax):
    _color = np.random.random(3)
    color = np.concatenate([_color, np.array([0.6])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=(_color[0], _color[1], _color[2]), facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label, color=(_color[0], _color[1], _color[2]))


def save_mask_data(output_dir, mask_list, box_list, label_list, im_name):
    value = 0  # 0 for background

    # mask_img = torch.zeros(mask_list.shape[-2:])
    with open(os.path.join(output_dir, "mask_data", f'{im_name}.pkl'), 'wb') as f:
        pickle.dump({'mask': mask_list}, f)
    # for idx, mask in enumerate(mask_list):
    #     mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    # plt.figure(figsize=(10, 10))
    # plt.imshow(mask_img.numpy())
    # plt.axis('off')
    # plt.savefig(os.path.join(output_dir, "mask_vis", f'{im_name}.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, "mask", f'{im_name}.json'), 'w') as f:
        json.dump(json_data, f)
    
    del json_data



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_version", type=str, default="vit_h", required=False, help="SAM ViT version: vit_b / vit_l / vit_h"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=False, help="path to sam checkpoint file"
    )
    parser.add_argument(
        "--sam_hq_checkpoint", type=str, default=None, help="path to sam-hq checkpoint file"
    )
    parser.add_argument(
        "--use_sam_hq", action="store_true", help="using sam-hq for prediction"
    )
    parser.add_argument("--input_dir", type=str, required=True, help="path to image file")
    parser.add_argument("--text_prompt", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--output_dir", "-o", type=str, default="outputs", required=True, help="output directory"
    )

    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_version = args.sam_version
    sam_checkpoint = args.sam_checkpoint
    sam_hq_checkpoint = args.sam_hq_checkpoint
    use_sam_hq = args.use_sam_hq
    input_dir = args.input_dir
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    device = args.device

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "vis"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "mask_data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "mask"), exist_ok=True)
    

    image_names = sorted(os.listdir(input_dir))
    image_names = [name for name in image_names if name.endswith("_cropped_original.png")]

    # initialize SAM
    if use_sam_hq:
        predictor = SamPredictor(sam_hq_model_registry[sam_version](checkpoint=sam_hq_checkpoint).eval().to(device))
    else:
        predictor = SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).eval().to(device))

    model = load_model(config_file, grounded_checkpoint, device=device)


    with torch.no_grad():
        for name in tqdm(image_names):

            image_path = os.path.join(input_dir, name)

            # load image
            image_pil, image = load_image(image_path)
            # load model

            # run grounding dino model
            boxes_filt, pred_phrases = get_grounding_output(
                model, image, text_prompt, box_threshold, text_threshold, device=device
            )

            if boxes_filt is None:
                continue
            
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            size = image_pil.size
            H, W = size[1], size[0]
            # for i in range(boxes_filt.size(0)):
            #     for j in range(i+1, boxes_filt.size(0)):

            for i in range(boxes_filt.size(0)):
                boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                boxes_filt[i][2:] += boxes_filt[i][:2]

            boxes_filt = boxes_filt.cpu()
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

            masks, _, _ = predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes.to(device),
                multimask_output = False,
            )

            # draw output image
            vis = plt.figure(figsize=(10, 10), num=1, clear=True)
            plt.imshow(image)
            # for mask in masks:
            #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            # for box, label in zip(boxes_filt, pred_phrases):
            #     show_box(box.numpy(), plt.gca(), label)

            for mask, box, label in zip(masks, boxes_filt, pred_phrases):
                show_mask_and_box(mask.cpu().numpy(), box.numpy(), label, plt.gca())

            plt.axis('off')
            plt.savefig(
                os.path.join(output_dir, "vis", name),
                bbox_inches="tight", dpi=300, pad_inches=0.0
            )

            vis.clear()
            plt.close(vis)
            plt.clf()

            save_mask_data(output_dir, masks, boxes_filt, pred_phrases, os.path.splitext(name)[0])

            del image
            del image_pil
            del masks
            del boxes_filt
            del pred_phrases
            
            predictor.reset_image()

             
            # # Getting % usage of virtual_memory ( 3rd field)
            # print('RAM memory % used:', psutil.virtual_memory()[2])
            # # Getting usage of virtual_memory in GB ( 4th field)
            # print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
