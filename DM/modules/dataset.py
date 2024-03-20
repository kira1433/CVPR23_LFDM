import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Tuple

class CustomVideoDataset(Dataset):
    def __init__(self, npy_path: str):
        super(CustomVideoDataset, self).__init__()
        self.videos = np.load(npy_path)

    def __len__(self) -> int:
        return self.videos.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.videos[idx]

# data = CustomVideoDataset("../../../data_108.npy")
# print(data.__getitem__(0).shape)
# print(type(data.__getitem__(0)))



