import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Tuple

class CustomVideoDataset(Dataset):
    def __init__(self, npy_pat: str = ):
        super(CustomVideoDataset, self).__init__()
        self.deepfakes = np.load("home/zeta/Workbenches/Diffusion/FFS/deepfakes_40.npy")
        self.labels = np.load("home/zeta/Workbenches/Diffusion/FFS/deepfakes_labels.npy")
        self.originals = np.load("home/zeta/Workbenches/Diffusion/FFS/originals_40.npy")

    def __len__(self) -> int:
        return self.videos.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.deepfakes[idx] , self.originals[self.labels[idx].split("_")[1],:,0,:,:]



