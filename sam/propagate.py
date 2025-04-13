# from segment_anything_hq import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

import numpy as np
from PIL import Image
import os, json
import cv2
from tqdm import tqdm
import imageio
import pickle
import argparse
import subprocess

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam_ckpt", type=str, required=True) 
    parser.add_argument("--data_dir", type=str, required=True) 
    parser.add_argument("--image_dir", type=str, required=True) 
    parser.add_argument("--sdf_dir", type=str, required=True) 
    
    args = parser.parse_args()
    
    use_points = False
    use_box = True

    sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt).cuda().eval()

    predictor = SamPredictor(sam)

    input_dir = args.image_dir
    mesh_mask_dir = os.path.join(args.data_dir, "grounded_sam/mesh_mask")
    mask_json_path = os.path.join(args.sdf_dir, "mask_mesh.pkl")
    save_dir = os.path.join(args.data_dir, "grounded_sam/propagate")
    
    subprocess.run(f"rm -rf {save_dir}/*", shell=True)

    with open(mask_json_path, 'rb') as f:
        mask_json = pickle.load(f)

    mask_cnt = mask_json["mask_cnt"]
    mask_colors = np.random.rand(mask_cnt, 3)

    frames = []
    sam_input_names = sorted(list(mask_json["sam_input"].keys()))
    for name in tqdm(sam_input_names, total=len(sam_input_names)):
        image_path = os.path.join(input_dir, name+".JPG")
        if not os.path.exists(image_path):
            image_path = os.path.join(input_dir, name+".png")

        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        frame_image = image.copy()
        predictor.set_image(image)
        H, W, _ = image.shape
        frame_vis_image = image.copy()

        for mask_i, prompt_pts in enumerate(mask_json["sam_input"][name]):
            mask_idx = prompt_pts["idx"]
            mesh_mask_path = os.path.join(mesh_mask_dir, name, f"{mask_idx}.png")
            mesh_mask = np.array(Image.open(mesh_mask_path)) > 0
            mesh_mask = mesh_mask.reshape(H, W)
            vis_image = np.zeros_like(image[..., 0]).astype(np.uint8)
            
            if use_points:
                positive_pts = prompt_pts["positive_pts"]
                negative_pts = prompt_pts["negative_pts"]

                point_positive_coords = np.array(positive_pts).reshape(-1, 2)[..., ::-1]
                point_negative_coords = np.array(negative_pts).reshape(-1, 2)[..., ::-1]
                point_coords = np.concatenate((point_positive_coords, point_negative_coords), axis=0)
                point_positive_labels = np.repeat(1, len(positive_pts)).reshape(-1)
                point_negative_labels = np.repeat(0, len(negative_pts)).reshape(-1)
                point_labels = np.concatenate((point_positive_labels, point_negative_labels), axis=0)

                low_res = prompt_pts["sam_mask_input"]

                # for prompt_i in range(1):
                for prompt_i in range(point_coords.shape[0]):
                
                    masks, qualities, low_res = predictor.predict(
                        point_coords=point_coords[:prompt_i+1], 
                        point_labels=point_labels[:prompt_i+1], 
                        mask_input=low_res, multimask_output=False, return_logits=True)
                
                mask = (masks > 0).reshape(H, W)

                inter = np.logical_and(mesh_mask, mask)
                union = np.logical_or(mesh_mask, mask)
                iou = inter / union 

                if iou < 0.7:
                    continue

                frame_vis_image[mask] = np.clip((frame_vis_image[mask]*0.3) + (mask_colors[mask_idx] * 0.7 * 255), 0, 255).astype(np.uint8)
                vis_image[mask] = 255
                frame_image[mask] = np.clip((frame_image[mask]*0.3) + (mask_colors[mask_idx] * 0.7 * 255), 0, 255).astype(np.uint8)
                for pt in positive_pts:
                    frame_vis_image = cv2.circle(frame_vis_image, (pt[1], pt[0]), radius=4, color=(0, 0, 255), thickness=-1)
                for pt in negative_pts:
                    frame_vis_image = cv2.circle(frame_vis_image, (pt[1], pt[0]), radius=4, color=(0, 255, 0), thickness=-1)
            elif use_box:
                min_h, min_w, max_h, max_w = prompt_pts["boundary"]
                box = np.array([min_w, min_h, max_w, max_h]).reshape(1, 4)

                low_res = prompt_pts["sam_mask_input"]
                for _ in range(5):
                    masks, qualities, low_res = predictor.predict(box=box, mask_input=low_res, multimask_output=False, return_logits=True)
                
                mask = (masks[0] > 0).reshape(H, W)

                inter = np.count_nonzero(np.logical_and(mesh_mask, mask))
                union = np.count_nonzero(np.logical_or(mesh_mask, mask))
                iou = inter / union 

                if iou < 0.7:
                    continue

                # frame_vis_image[mask] = np.clip((frame_vis_image[mask]*0.3) + (mask_colors[mask_idx] * 0.7 * 255), 0, 255).astype(np.uint8)
                vis_image[mask] = 255
                frame_image[mask] = np.clip((frame_image[mask]*0.3) + (mask_colors[mask_idx] * 0.7 * 255), 0, 255).astype(np.uint8)

                # frame_vis_image = cv2.rectangle(frame_vis_image, (min_w, min_h), (max_w, max_h), color=(0, 255, 0), thickness=4)
            else:
                assert False

            vis_image = Image.fromarray(vis_image)
            os.makedirs(f'{save_dir}/sam/{name}/', exist_ok=True)
            vis_image.save(f"{save_dir}/sam/{name}/{mask_idx}.png")
        
        # frame_vis_image  = Image.fromarray(frame_vis_image)
        # os.makedirs(f'{save_dir}/frame_vis/', exist_ok=True)
        # frame_vis_image.save(f"{save_dir}/frame_vis/{name}.png")

        frame_image = cv2.resize(frame_image, (W // 2 if (W // 2) % 2 == 0 else (W // 2 + 1), H // 2  if (H // 2) % 2 == 0 else (H // 2 + 1)))
        frames.append(frame_image)
        
    imageio.mimsave(os.path.join(f'{save_dir}/render_mask.mp4'),
                            frames,
                            fps=24, macro_block_size=1)