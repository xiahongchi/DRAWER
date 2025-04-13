from typing import Dict

import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_semantics_and_mask_tensors_from_path
from torch.utils.data import Dataset

class Distill2dDataset(Dataset):
    def __init__(self, dataparser_outputs: DataparserOutputs, resolution: int = 512):
        super().__init__()
        self.cameras = dataparser_outputs.cameras
        self.resolution = resolution

    def __len__(self):
        return 1000


    def __getitem__(self, idx: int) -> Dict:
        data = {}

        return data