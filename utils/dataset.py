import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
import imageio
from PIL import Image

class DeepfakesDataset(Dataset):
    def __init__(self):
        super(DeepfakesDataset, self).__init__()
        self.originals = np.load("/home/zeta/Workbenches/Diffusion/FFS/originals_40.npy").astype(np.dtype('float32'))
        self.deepfakes = np.load("/home/zeta/Workbenches/Diffusion/FFS/deepfakes_40.npy").astype(np.dtype('float32'))
        self.labels = np.load("/home/zeta/Workbenches/Diffusion/FFS/deepfakes_labels.npy")

    def __len__(self) -> int:
        return self.deepfakes.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        ind = idx
        if ind>=365: ind+=1
        return self.deepfakes[idx], self.originals[ind] , self.originals[self.labels[idx],:,0,:,:]

class FaceShifterDataset(Dataset):
    def __init__(self):
        super(FaceShifterDataset, self).__init__()
        self.originals = np.load("/home/zeta/Workbenches/Diffusion/FFS/originals_40.npy").astype(np.dtype('float32'))
        self.deepfakes = np.load("/home/zeta/Workbenches/Diffusion/FFS/FaceShifter_40.npy").astype(np.dtype('float32'))
        self.labels = np.load("/home/zeta/Workbenches/Diffusion/FFS/FaceShifter_labels.npy")
        
    def __len__(self) -> int:
        return self.deepfakes.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.deepfakes[idx], self.originals[idx],  self.originals[self.labels[idx],:,0,:,:]

class FaceSwapDataset(Dataset):
    def __init__(self):
        super(FaceSwapDataset, self).__init__()
        self.originals = np.load("/home/zeta/Workbenches/Diffusion/FFS/originals_40.npy").astype(np.dtype('float32'))
        self.deepfakes = np.load("/home/zeta/Workbenches/Diffusion/FFS/FaceSwap_40.npy").astype(np.dtype('float32'))
        self.labels = np.load("/home/zeta/Workbenches/Diffusion/FFS/FaceSwap_labels.npy")

    def __len__(self) -> int:
        return self.deepfakes.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.deepfakes[idx], self.originals[idx],  self.originals[self.labels[idx],:,0,:,:]

class NeuralTexturesDataset(Dataset):
    def __init__(self):
        super(NeuralTexturesDataset, self).__init__()
        self.originals = np.load("/home/zeta/Workbenches/Diffusion/FFS/originals_40.npy").astype(np.dtype('float32'))
        self.deepfakes = np.load("/home/zeta/Workbenches/Diffusion/FFS/NeuralTextures_40.npy").astype(np.dtype('float32'))
        self.labels = np.load("/home/zeta/Workbenches/Diffusion/FFS/NeuralTextures_labels.npy")

    def __len__(self) -> int:
        return self.deepfakes.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.deepfakes[idx], self.originals[idx],  self.originals[self.labels[idx],:,0,:,:]

class Face2FaceDataset(Dataset):
    def __init__(self):
        super(Face2FaceDataset, self).__init__()
        self.originals = np.load("/home/zeta/Workbenches/Diffusion/FFS/originals_40.npy").astype(np.dtype('float32'))
        self.deepfakes = np.load("/home/zeta/Workbenches/Diffusion/FFS/Face2Face_40.npy").astype(np.dtype('float32'))
        self.labels = np.load("/home/zeta/Workbenches/Diffusion/FFS/Face2Face_labels.npy")

    def __len__(self) -> int:
        return self.deepfakes.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.deepfakes[idx], self.originals[idx],  self.originals[self.labels[idx],:,0,:,:]
    
