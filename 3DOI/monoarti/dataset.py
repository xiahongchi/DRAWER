from multiprocessing.sharedctypes import Value
import os
from typing import List, Optional, Tuple, Dict
import warnings
import numpy as np
from PIL import Image
import random
import socket
import json
import pycocotools.mask as mask_util

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .utils import gaussian_radius, draw_gaussian


DEFAULT_DATA_ROOT = '/home/ubuntu/monoarti_data'
DEFAULT_DEPTH_ROOT = '/home/ubuntu/monoarti_data/omnidata_filtered/depth_zbuffer/taskonomy'


def polygons_to_bitmask(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    if len(polygons) == 0:
        # COCOAPI does not support empty polygons
        return np.zeros((height, width)).astype(bool)
    rles = mask_util.frPyObjects(polygons, height, width)
    rle = mask_util.merge(rles)
    return mask_util.decode(rle).astype(bool)


class DemoDataset(Dataset):
    """
    demo dataset
    """
    def __init__(
        self,
        dataset_name,
        entries: List, 
        image_size,
        output_size,
        num_views: int = 1,
        load_depth: bool = False,
        affordance_radius: int = 5,
        num_queries: int = 15,
        bbox_to_mask: bool = False,
    ):
        """
        Args:
            entries: The list of dataset entries.
        """
        self._dataset_name = dataset_name
        self._entries = entries
        self._image_size = image_size
        self._affordance_radius = affordance_radius
        self._num_queries = num_queries
        self._bbox_to_mask = bbox_to_mask

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, index):
        entry = self._entries[index]

        # read image
        image_size = self._image_size
        img_path = entry
        img_name = img_path.split('/')[-1]

        image = np.array(Image.open(img_path))
        image = image[:, :, :3]  # in case some images have 4 channels
        image = self.transforms(image)

        # fake depth
        depth = np.ones_like(image[0]) * -1.0
        depth = torch.FloatTensor(depth)

        # fill in instances so that it is compatible with our data
        keypoint = []
        movable = []
        rigid = []
        kinematic = []
        action = []
        affordance = []
        bbox = []
        bbox_to_mask = []
        axis = []
        masks = []
        valid = []

        while len(valid) < self._num_queries:  # padding
            valid.append(0.0)
            keypoint.append(torch.LongTensor([0, 0]))
            movable.append(torch.LongTensor([-100]))
            rigid.append(torch.LongTensor([-100]))
            kinematic.append(torch.LongTensor([-100]))
            action.append(torch.LongTensor([-100]))
            affordance.append(torch.FloatTensor([0, 0]))
            bbox.append([0, 0, 0, 0])
            if self._bbox_to_mask:
                bbox_to_mask.append(torch.ones(tuple(image_size)).long() *-100)
            axis.append([-1, -1, -1, -1])
            masks.append(torch.zeros(tuple(image_size), dtype=bool))

        valid = torch.BoolTensor(valid)
        keypoint = torch.stack(keypoint)
        movable = torch.stack(movable).squeeze()
        rigid = torch.stack(rigid).squeeze()
        kinematic = torch.stack(kinematic).squeeze()
        action = torch.stack(action).squeeze()
        
        affordance = torch.stack(affordance)
        affordance_map = torch.zeros((self._num_queries, *image_size))

        bbox = torch.FloatTensor(bbox)
        if self._bbox_to_mask:
            bbox_to_mask = torch.stack(bbox_to_mask)

        axis = torch.FloatTensor(axis)
        masks = torch.stack(masks)
        fov = 1.0 

        # resize images to image_size
        scale_factors = [s_new / s for s, s_new in zip(image.shape[0:2], image_size)]
        scale_factor = sum(scale_factors) * 0.5
        if scale_factor != 1.0:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=tuple(image_size),
                mode="bilinear",
            )[0]
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=tuple(image_size),
                mode="nearest",
            )[0, 0]


        ret_entry = {
            'img_name': img_name,
            'image': image,
            'valid': valid,
            'keypoints': keypoint,
            'bbox': bbox,
            'masks': masks,
            'movable': movable,
            'rigid': rigid,
            'kinematic': kinematic,
            'action': action,
            'affordance': affordance,
            'affordance_map': affordance_map,
            'depth': depth,
            'axis': axis,
            'fov': fov,
        }

        return ret_entry


class InteractionDataset(Dataset):
    """
    3D Object Interaction Dataset
    """

    def __init__(
        self,
        dataset_name,
        entries: List, 
        image_size, 
        output_size,
        num_views: int = 1,
        load_depth: bool = False,
        affordance_radius: int = 5,
        num_queries: int = 15,
        bbox_to_mask: bool = False,
    ):
        """
        Args:
            entries: The list of dataset entries.
        """
        self._dataset_name = dataset_name
        self._entries = entries
        self._image_size = image_size
        self._output_size = output_size
        self._num_views = num_views
        self._load_depth = load_depth
        self._affordance_radius = affordance_radius
        self._num_queries = num_queries
        self._bbox_to_mask = bbox_to_mask

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.movable_map = {
            'one_hand': 0,
            'two_hands': 1,
            'fixture': 2,
        }

        self.rigid_map = {
            'yes': 1,
            'no': 0,
            'n/a': -100,
        }

        self.kinematic_map = {
            'freeform': 0,
            'rotation': 1,
            'translation': 2,
            'n/a': -100,
        }

        self.action_map = {
            'free movement': 0,
            'free': 0,
            'pull': 1,
            'push': 2,
            'n/a': -100,
        }



    def __len__(self):
        return len(self._entries)

    def __getitem__(self, index):
        entry = self._entries[index]

        # read image metadata
        image_size = self._image_size
        output_size = self._output_size
        img_name = entry['img_name']

        # read image
        if self._dataset_name.startswith('3doi'):
            dataset_path = 'images'
        elif self._dataset_name.startswith('whirl'):
            dataset_path = 'whirl_images'
        else:
            raise NotImplementedError("unknown dataset!")
        img_path = os.path.join(DEFAULT_DATA_ROOT, dataset_path, img_name)

        try:
            image = np.array(Image.open(img_path))
        except:
            print("error {}".format(img_path))
            return self.__getitem__(random.choice(range(len(self._entries))))


        image = self.transforms(image)

        instances = entry['instances']
        keypoint = []
        movable = []
        rigid = []
        kinematic = []
        action = []
        affordance = []
        affordance_map = []
        bbox = []
        bbox_to_mask = []
        axis = []
        masks = []
        valid = []
        for inst in instances:
            valid.append(1.0)

            keypoint_raw = inst['keypoint']
            keypoint.append(torch.LongTensor([
                int(keypoint_raw[0] * image_size[1]),
                int(keypoint_raw[1] * image_size[0]),
            ]))

            # movable (one_hand or two_hands)
            movable_str = inst['movable']
            movable.append(torch.LongTensor([self.movable_map[movable_str]]))

            # rigid
            rigid_str = inst['rigid']
            rigid.append(torch.LongTensor([self.rigid_map[rigid_str]]))

            # kinematic
            kinematic_str = inst['kinematic']
            kinematic.append(torch.LongTensor([self.kinematic_map[kinematic_str]]))

            # action: pull or push
            action_str = inst['pull_or_push']
            action.append(torch.LongTensor([self.action_map[action_str]]))

            # affordance
            affordance_raw = inst['affordance']
            affordance_scaled = torch.LongTensor([
                int(affordance_raw[0] * output_size[1] / 4), # width
                int(affordance_raw[1] * output_size[0] / 4), # height
            ])
            affordance_raw = torch.FloatTensor([
                affordance_raw[0],
                affordance_raw[1],
            ])
            affordance.append(affordance_raw)
            affordance_map_entry = torch.zeros((output_size[0] // 4, output_size[1] // 4))
            if affordance_raw[0] > -0.5:
                draw_gaussian(affordance_map_entry.numpy(), affordance_scaled, radius=self._affordance_radius)
            affordance_map.append(affordance_map_entry)

            # axis
            axis_raw = inst['axis']
            axis.append(axis_raw)

            # bbox
            bbox.append(inst['bbox'])
            if self._bbox_to_mask:
                bbox_int = [
                    int(inst['bbox'][0] * image_size[1]),
                    int(inst['bbox'][1] * image_size[0]),
                    int(inst['bbox'][2] * image_size[1]),
                    int(inst['bbox'][3] * image_size[0]),
                ]
                bbox_mask = torch.zeros(tuple(image_size)).long()
                bbox_mask[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]] = 1
                bbox_to_mask.append(bbox_mask)

            # mask
            mask_polygon = inst['mask']
            if len(mask_polygon) == 0:  # no mask annotation
                mask = torch.zeros(tuple(output_size), dtype=bool)
            else:
                polygon = np.array(mask_polygon)
                polygon[:, 0] = polygon[:, 0] * output_size[1]
                polygon[:, 1] = polygon[:, 1] * output_size[0]
                polygon = polygon.astype(int).reshape(-1)
                rles = mask_util.frPyObjects([polygon], output_size[0], output_size[1])
                rle = mask_util.merge(rles)
                mask = mask_util.decode(rle).astype(bool)
                mask = torch.as_tensor(mask)
            masks.append(mask)

        while len(valid) < self._num_queries:  # padding
            valid.append(0.0)
            keypoint.append(torch.LongTensor([0, 0]))
            movable.append(torch.LongTensor([-100]))
            rigid.append(torch.LongTensor([-100]))
            kinematic.append(torch.LongTensor([-100]))
            action.append(torch.LongTensor([-100]))
            #affordance.append(torch.LongTensor([0, 0]))
            affordance.append(torch.FloatTensor([-1, -1]))
            affordance_size = (output_size[0] // 4, output_size[1] // 4)
            #affordance_map = torch.ones(affordance_size)
            affordance_map.append(torch.ones(affordance_size) * -100)
            #affordance_map.append(torch.ones_like(affordance_map[0]) * -100)
            bbox.append([-1, -1, -1, -1])
            if self._bbox_to_mask:
                bbox_to_mask.append(torch.ones(tuple(image_size)).long() *-100)
            axis.append([-1, -1, -1, -1])
            masks.append(torch.zeros(tuple(output_size), dtype=bool))

        valid = torch.BoolTensor(valid)
        keypoint = torch.stack(keypoint)
        movable = torch.stack(movable).squeeze()
        rigid = torch.stack(rigid).squeeze()
        kinematic = torch.stack(kinematic).squeeze()
        action = torch.stack(action).squeeze()
        affordance = torch.stack(affordance)
        affordance_map = torch.stack(affordance_map)
        bbox = torch.FloatTensor(bbox)
        if self._bbox_to_mask:
            bbox_to_mask = torch.stack(bbox_to_mask)

        axis = torch.FloatTensor(axis)
        masks = torch.stack(masks)

        # read depth
        # example: /nfs/turbo/fouheyTemp/jinlinyi/omnidata/omnidata_taskonomy/depth_zbuffer/taskonomy/almena/point_426_view_3_domain_depth_zbuffer.png
        img_name = entry['img_name']
        splits = img_name.split('_')
        data_source = splits[0]
        if self._load_depth and data_source == 'taskonomy':
            scene_name = splits[1]
            point_id = int(splits[3])
            view_id = int(splits[5])
            depth_name = 'point_{}_view_{}_domain_depth_zbuffer.png'.format(point_id, view_id)
            depth_path = os.path.join(DEFAULT_DEPTH_ROOT, scene_name, depth_name)
            depth = np.array(Image.open(depth_path))
            depth = depth / 512.0

            depth_mask_name = depth_name
            mask_path = os.path.join(
                DEFAULT_DEPTH_ROOT.replace('depth_zbuffer', 'mask_valid'), 
                scene_name, 
                depth_mask_name
            )
            depth_mask = np.array(Image.open(mask_path))
            depth[depth_mask < 127] = -1.0

            camera_name = 'point_{}_view_{}_domain_point_info.json'.format(point_id, view_id)
            camera_path = os.path.join(
                DEFAULT_DEPTH_ROOT.replace('depth_zbuffer', 'point_info'), 
                scene_name, 
                camera_name,
            )
            with open(camera_path) as f:
                camera_dict = json.load(f)
            fov = camera_dict['field_of_view_rads']
        else:
            # pseudo depth
            depth = np.ones_like(image[0]) * -1.0
            fov = -1.0
        depth = torch.FloatTensor(depth)

        # resize images to image_size
        scale_factors = [s_new / s for s, s_new in zip(image.shape[0:2], image_size)]
        scale_factor = sum(scale_factors) * 0.5
        if scale_factor != 1.0:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=tuple(image_size),
                mode="bilinear",
            )[0]

            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=tuple(output_size),
                mode="nearest",
            )[0, 0]

        ret_entry = {
            'img_name': img_name,
            'image': image,
            'valid': valid,
            'keypoints': keypoint,
            'bbox': bbox,
            'masks': masks,
            'movable': movable,
            'rigid': rigid,
            'kinematic': kinematic,
            'action': action,
            'affordance': affordance,
            'affordance_map': affordance_map,
            'depth': depth,
            'axis': axis,
            'fov': fov,
        }

        if self._bbox_to_mask:
            ret_entry['bbox_to_mask'] = bbox_to_mask  

        #pdb.set_trace()
        return ret_entry


def prepare_datasets(
    dataset_name: str,
    image_size: Tuple[int, int],
    output_size: Tuple[int, int],
    data_root: str = DEFAULT_DATA_ROOT,
    autodownload: bool = False,
    num_views: int = 1,
    load_depth: bool = False,
    affordance_radius: int = 5,
    num_queries: int = 15,
    bbox_to_mask: bool = False,
    #splits: Tuple[str, str, str] = ('train', 'val', 'test'),
    split: str = 'train',
) -> Dataset:
    if dataset_name.startswith('demo'):
        # entries = [
        #     '/nfs/turbo/fouheyTemp/syqian/whirl_data/cup/000000.png',
        #     '/nfs/turbo/fouheyTemp/syqian/whirl_data/door/000000.png',
        #     '/nfs/turbo/fouheyTemp/syqian/whirl_data/fridge/000000.png',
        # ]
        filenames = os.listdir(data_root)
        entries = []
        for filename in filenames:
            entries.append(os.path.join(data_root, filename))
    else:
        dataset_dir = os.path.join(data_root, dataset_name)
        cameras_path = os.path.join(dataset_dir, 'data_{}.pt'.format(split))
        train_data = torch.load(cameras_path)
        entries = train_data

    if dataset_name.startswith('3doi') or dataset_name.startswith('whirl') or dataset_name.startswith('omnidata_supp'):
        dataset = InteractionDataset(
            dataset_name,
            entries, 
            image_size,
            output_size,
            num_views=num_views, 
            load_depth=load_depth,
            affordance_radius=affordance_radius,
            num_queries=num_queries,
            bbox_to_mask=bbox_to_mask,
        )
    elif dataset_name.startswith('demo'):
        dataset = DemoDataset(
            dataset_name,
            entries, 
            image_size, 
            output_size,
            num_views=num_views, 
            load_depth=load_depth,
            affordance_radius=affordance_radius,
            num_queries=num_queries,
            bbox_to_mask=bbox_to_mask,
        )
    else:
        raise NotImplementedError("unknown dataset!")

    return dataset


def merge_datasets(datasets: List[Dataset]) -> Dataset:
    if len(datasets) > 1:
        dataset = torch.utils.data.ConcatDataset(datasets)
    elif len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = None

    return dataset


def get_interaction_datasets(
    train_dataset_names: List[str],
    val_dataset_names: List[str],
    test_dataset_names: List[str],
    image_size: Tuple[int, int],
    output_size: Tuple[int, int],
    data_root: str = DEFAULT_DATA_ROOT,
    autodownload: bool = False,
    num_views: int = 1,
    load_depth: bool = False,
    affordance_radius: int = 5,
    num_queries: int = 15,
    bbox_to_mask: bool = False,
) -> Tuple[Dataset, Dataset, Dataset]:

    # prepare training sets
    train_datasets = []
    for dataset_name in train_dataset_names:
        train_dataset = prepare_datasets(
            dataset_name, 
            image_size,
            output_size,
            data_root, 
            autodownload, 
            num_views, 
            load_depth,
            affordance_radius,
            num_queries,
            bbox_to_mask,
            split='train',
        )
        train_datasets.append(train_dataset)
    train_dataset = merge_datasets(train_datasets)

    # prepare validation sets
    val_datasets = []
    for dataset_name in val_dataset_names:
        val_dataset = prepare_datasets(
            dataset_name, 
            image_size, 
            output_size,
            data_root, 
            autodownload, 
            num_views, 
            load_depth,
            affordance_radius,
            num_queries,
            bbox_to_mask,
            split='val',
        )
        val_datasets.append(val_dataset)
    val_dataset = merge_datasets(val_datasets)

    # prepare test sets
    # prepare validation sets
    test_datasets = []
    for dataset_name in test_dataset_names:
        test_dataset = prepare_datasets(
            dataset_name, 
            image_size, 
            output_size,
            data_root, 
            autodownload, 
            num_views, 
            load_depth,
            affordance_radius,
            num_queries,
            bbox_to_mask,
            split='test',
        )
        test_datasets.append(test_dataset)
    test_dataset = merge_datasets(test_datasets)

    return train_dataset, val_dataset, test_dataset


    
