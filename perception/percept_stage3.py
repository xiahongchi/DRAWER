import os
from pytorch3d.structures import Meshes
import trimesh
from sklearn.cluster import DBSCAN
import subprocess
import torch
import numpy as np
import pickle
from tqdm import tqdm
import torch.nn.functional as F
import torchvision
import nvdiffrast.torch as dr
from PIL import Image
import argparse
import time
import datetime

def rasterize_texture(vertices, faces, projection, glctx, resolution):
    """Rasterize 3D mesh to 2D using nvdiffrast"""
    vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), torch.transpose(projection, 0, 1)).float().unsqueeze(0)
    rast_out, _ = dr.rasterize(glctx, vertices_clip, faces, resolution=resolution)

    H, W = resolution
    valid = (rast_out[..., -1] > 0).reshape(H, W)
    triangle_id = (rast_out[..., -1] - 1).long().reshape(H, W)

    return valid, triangle_id


def calculate_iou_mask(mask_a, mask_b):
    """Calculate Intersection over Union between two binary masks"""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    
    # Avoid division by zero
    iou = intersection / union if union != 0 else 0.0
    
    return iou


def preprocess_masks(all_mask_dict, resolution, edge_detect_px=20):
    """Preprocess all masks to avoid repeated calculations"""
    print("Preprocessing masks...")
    H, W = resolution
    mask_infos = {}
    
    # First pass - filter edge masks and compute mask properties
    for stem in tqdm(all_mask_dict.keys()):
        mask_infos[stem] = {}
        
        for mask_idx, binary_mask in all_mask_dict[stem].items():
            # Skip masks that touch the image edge
            if np.any(binary_mask[:edge_detect_px]) or np.any(binary_mask[-edge_detect_px:]) or \
               np.any(binary_mask[:, :edge_detect_px]) or np.any(binary_mask[:, -edge_detect_px:]):
                continue
            
            # Pre-compute mask metrics
            num_mask_pixel = np.count_nonzero(binary_mask)
            
            # Skip very small masks (likely noise)
            if num_mask_pixel < 100:
                continue
                
            # Get mask dimensions
            w_mask = np.any(binary_mask, axis=0).reshape(-1)
            h_mask = np.any(binary_mask, axis=1).reshape(-1)
            w_min = np.min(np.nonzero(w_mask))
            w_max = np.max(np.nonzero(w_mask))
            h_min = np.min(np.nonzero(h_mask))
            h_max = np.max(np.nonzero(h_mask))
            mask_w = w_max - w_min
            mask_h = h_max - h_min
            
            # Calculate "squareness" ratio
            sqratio = num_mask_pixel / (mask_w * mask_h)
            
            # Resize mask to match resolution
            binary_mask_resized = torchvision.transforms.functional.resize(
                torch.from_numpy(binary_mask).unsqueeze(0) > 0, 
                (H, W), 
                torchvision.transforms.InterpolationMode.NEAREST
            ).cpu().numpy()[0]
            
            # Store pre-computed mask info
            mask_infos[stem][mask_idx] = {
                "binary_mask": binary_mask,
                "binary_mask_resized": binary_mask_resized,
                "num_pixels": num_mask_pixel,
                "sqratio": sqratio,
                "bbox": (w_min, w_max, h_min, h_max)
            }
    
    print(f"Preprocessing complete. Kept {sum(len(v) for v in mask_infos.values())} valid masks.")
    return mask_infos


def get_camera_distances(stem_list, pose_dict, mesh_center):
    """Sort cameras by distance to mesh center for more efficient processing"""
    camera_relevance = []
    
    for stem in stem_list:
        if stem not in pose_dict:
            continue
        
        pose = pose_dict[stem]
        c2w = pose["c2w"]
        # Camera position is in the translation part of the c2w matrix
        camera_pos = c2w[:3, 3]
        
        # Distance between camera and mesh center
        distance = np.linalg.norm(camera_pos - mesh_center)
        camera_relevance.append((stem, distance))
    
    # Sort cameras by distance (closest first)
    camera_relevance.sort(key=lambda x: x[1])
    return camera_relevance


def save_mesh_outputs(mesh_i, match_data, data_dir, save_dir):
    """Save output visualizations for a mesh match"""
    if match_data is None:
        return
        
    stem = match_data["stem"]
    stem_idx = stem.replace("frame_", "")
    mask_idx = match_data["mask_idx"]
    binary_mask = match_data["binary_mask"]
    
    # Load and save original image
    image_path = os.path.join(data_dir, stem+".png")
    src_image_pil = Image.open(image_path)
    src_image_pil.save(os.path.join(save_dir, f"back_match_mesh_{mesh_i:0>3d}_frame_{stem_idx}_{mask_idx:0>2d}_original.png"))

    # Create and save masked image
    src_image = np.array(src_image_pil).astype(np.float32) / 255.
    src_image_masked = src_image.copy()
    src_image_masked[binary_mask] = src_image_masked[binary_mask] * 0.7 + np.array([0., 1., 0.]) * 0.3
    Image.fromarray(np.clip(src_image_masked * 255., 0, 255).astype(np.uint8)).save(
        os.path.join(save_dir, f"back_match_mesh_{mesh_i:0>3d}_frame_{stem_idx}_{mask_idx:0>2d}_image_masked.png")
    )

    # Save mask
    Image.fromarray(np.clip(binary_mask.astype(np.float32) * 255., 0, 255).astype(np.uint8)).save(
        os.path.join(save_dir, f"back_match_mesh_{mesh_i:0>3d}_frame_{stem_idx}_{mask_idx:0>2d}_mask.png")
    )

    # Calculate and save cropped image
    w_mask = np.any(binary_mask, axis=0).reshape(-1)
    h_mask = np.any(binary_mask, axis=1).reshape(-1)
    w_min = np.min(np.nonzero(w_mask))
    w_max = np.max(np.nonzero(w_mask))
    h_min = np.min(np.nonzero(h_mask))
    h_max = np.max(np.nonzero(h_mask))
    croped_src_image = src_image[h_min:h_max, w_min:w_max]
    
    Image.fromarray(np.clip(croped_src_image * 255., 0, 255).astype(np.uint8)).save(
        os.path.join(save_dir, f"back_match_mesh_{mesh_i:0>3d}_frame_{stem_idx}_{mask_idx:0>2d}_cropped_original.png")
    )


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--early_stop_iou', type=float, default=0.95, help='IoU threshold for early stopping')
    parser.add_argument('--max_frames_per_mesh', type=int, default=None, help='Maximum frames to check per mesh')
    
    args = parser.parse_args()

    # Setup paths
    pose_path = os.path.join(args.data_dir, "pose.pkl")
    mask_dir = os.path.join(args.data_dir, "grounded_sam")
    filtered_mesh_dir = os.path.join(args.data_dir, 'perception', "vis_groups_normal_filtered")
    save_dir = os.path.join(args.data_dir, 'perception', "vis_groups_back_match")
    
    data_dir = args.image_dir

    # Prepare output directory
    os.makedirs(save_dir, exist_ok=True)
    subprocess.run(f"rm {save_dir}/*", shell=True)

    # Load data
    print("Loading data files...")
    mask_save_path = os.path.join(mask_dir, "separate_vis_gsam.pkl")

    with open(pose_path, 'rb') as f:
        pose_dict = pickle.load(f)

    with open(mask_save_path, 'rb') as f:
        all_mask_dict = pickle.load(f)

    stem_list = sorted(list(all_mask_dict.keys()))
    glctx = dr.RasterizeGLContext(output_db=False)

    # Get mesh files
    mask_mesh_fnames = sorted(os.listdir(filtered_mesh_dir))
    mask_mesh_fnames = [elem for elem in mask_mesh_fnames if elem.endswith(".ply")]

    # Determine resolution
    print("Setting up resolution...")
    H, W = np.array(Image.open(os.path.join(data_dir, os.listdir(data_dir)[0]))).shape[:2]
    while H % 8 != 0 or W % 8 != 0:
        H = H * 2
        W = W * 2
    resolution = (H, W)
    H, W = resolution[0], resolution[1]
    print(f"Using resolution: {resolution}")
    
    # Pre-process all masks - this avoids repeated computations
    mask_infos = preprocess_masks(all_mask_dict, resolution)
    
    print(f"===============================================")
    print(f"Starting optimized processing")
    print(f"Total meshes to process: {len(mask_mesh_fnames)}")
    if args.max_frames_per_mesh:
        print(f"Limiting to {args.max_frames_per_mesh} frames per mesh")
    print(f"Early stopping IoU threshold: {args.early_stop_iou}")
    print(f"===============================================")
    
    start_time = time.time()
    
    # Process each mesh
    for mesh_i, fname in tqdm(enumerate(mask_mesh_fnames), total=len(mask_mesh_fnames), desc="Processing meshes"):
        mesh_start_time = time.time()
        print(f"\nProcessing mesh {mesh_i+1}/{len(mask_mesh_fnames)}: {fname}")
        
        # Load mesh
        mask_mesh_fpath = os.path.join(filtered_mesh_dir, fname)
        mask_mesh = trimesh.exchange.load.load_mesh(mask_mesh_fpath)
        mask_mesh_verts = torch.from_numpy(mask_mesh.vertices).float().cuda()
        mask_mesh_faces = torch.from_numpy(mask_mesh.faces).int().cuda()
        
        # Initialize best matches
        potential_match = None
        maxinum_sqratio = 0
        potential_match_sec = None
        maxinum_sqratio_sec = 0
        
        # Calculate mesh center for camera ordering
        mesh_center = mask_mesh_verts.mean(dim=0).cpu().numpy()
        
        # Get cameras ordered by relevance (distance to mesh center)
        camera_relevance = get_camera_distances(stem_list, pose_dict, mesh_center)
        
        # Limit number of frames to check if specified
        if args.max_frames_per_mesh is not None:
            camera_relevance = camera_relevance[:args.max_frames_per_mesh]
        
        frames_checked = 0
        total_masks_checked = 0
        found_excellent_match = False
        
        # Process cameras in order of relevance
        for stem, distance in camera_relevance:
            frames_checked += 1
            
            # Skip if no masks for this stem or excellent match already found
            if stem not in mask_infos or not mask_infos[stem] or found_excellent_match:
                continue
                
            # Get camera pose
            pose = pose_dict[stem]
            mvp = torch.from_numpy(pose["mvp"]).cuda().float()
            
            # Rasterize mesh with this camera (only once per camera)
            valid, triangle_id = rasterize_texture(mask_mesh_verts, mask_mesh_faces, mvp, glctx, resolution)
            
            if not torch.any(valid):
                continue
                
            valid = valid.cpu().numpy() > 0
            
            # Get masks for this stem, sorted by area (largest first)
            stem_masks = [(mask_idx, info) for mask_idx, info in mask_infos[stem].items()]
            stem_masks.sort(key=lambda x: x[1]["num_pixels"], reverse=True)
            
            # Check each mask
            for mask_idx, mask_info in stem_masks:
                total_masks_checked += 1
                
                binary_mask_resized = mask_info["binary_mask_resized"]
                binary_mask = mask_info["binary_mask"]
                
                # Calculate IoU
                iou = calculate_iou_mask(valid, binary_mask_resized)
                
                # Check for excellent match for early termination
                if iou > args.early_stop_iou:
                    potential_match = {
                        "stem": stem,
                        "mask_idx": mask_idx,
                        "binary_mask": binary_mask,
                    }
                    found_excellent_match = True
                    print(f"  Found excellent match (IoU={iou:.3f}) - early stopping")
                    break
                    
                if iou > 0.8:
                    sqratio = mask_info["sqratio"]
                    if sqratio > maxinum_sqratio:
                        potential_match = {
                            "stem": stem,
                            "mask_idx": mask_idx,
                            "binary_mask": binary_mask,
                        }
                        maxinum_sqratio = sqratio
                        print(f"  Found good match (IoU={iou:.3f}, sqratio={sqratio:.3f})")
                        
                elif iou > 0.6:
                    sqratio = mask_info["sqratio"]
                    if sqratio > maxinum_sqratio_sec:
                        potential_match_sec = {
                            "stem": stem,
                            "mask_idx": mask_idx,
                            "binary_mask": binary_mask,
                        }
                        maxinum_sqratio_sec = sqratio
            
            # If we found an excellent match, stop processing frames
            if found_excellent_match:
                break
        
        # Calculate mesh processing time
        mesh_time = time.time() - mesh_start_time
        
        # Print mesh processing statistics
        print(f"  Mesh {mesh_i} processing complete in {mesh_time:.2f} seconds")
        print(f"  Checked {frames_checked}/{len(camera_relevance)} frames and {total_masks_checked} masks")
        
        # Process and save results
        if potential_match is not None:
            save_mesh_outputs(mesh_i, potential_match, data_dir, save_dir)
            print(f"  Saved primary match outputs (frame {potential_match['stem']}, mask {potential_match['mask_idx']})")
        elif potential_match_sec is not None:
            save_mesh_outputs(mesh_i, potential_match_sec, data_dir, save_dir)
            print(f"  Saved secondary match outputs (frame {potential_match_sec['stem']}, mask {potential_match_sec['mask_idx']})")
        else:
            print(f"  No suitable match found for mesh {mesh_i}")
    
    # Calculate and print overall statistics
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"===============================================")
    print(f"Processing completed!")
    print(f"Total meshes processed: {len(mask_mesh_fnames)}")
    print(f"Total time: {datetime.timedelta(seconds=int(elapsed))} ({elapsed:.2f} seconds)")
    print(f"Average time per mesh: {elapsed/len(mask_mesh_fnames):.2f} seconds")
    print(f"Results saved to {save_dir}")
    print(f"===============================================")