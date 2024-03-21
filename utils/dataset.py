import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Tuple

class CustomVideoDataset(Dataset):
    def __init__(self):
        super(CustomVideoDataset, self).__init__()
        self.deepfakes = np.load("/home/zeta/Workbenches/Diffusion/FFS/deepfakes_40.npy").astype(np.dtype('float32'))
        self.labels = np.load("/home/zeta/Workbenches/Diffusion/FFS/deepfakes_labels.npy")
        self.originals = np.load("/home/zeta/Workbenches/Diffusion/FFS/originals_40.npy").astype(np.dtype('float32'))

    def __len__(self) -> int:
        return self.deepfakes.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        ind = idx
        if ind>=365: ind+=1
        return self.deepfakes[idx], self.originals[ind] , self.originals[self.labels[idx],:,0,:,:]

# data = CustomVideoDataset()
# a,b,c = data.__getitem__(365)
# print(a.shape,b.shape,c.shape)
# save_gif(a,"original.gif")
# save_gif(b,"deepfake.gif")

# new_im = Image.fromarray(sample_img(c), 'RGB')
# new_im.save("original.jpg")