import h5py
import numpy as np
from scipy.ndimage import zoom
import cv2
import sys
# import os
# from PIL import Image
# from typing import Tuple


# def iter_obj(name, obj):
#     if isinstance(obj, h5py.Dataset):
#         print(f"Dataset: {name}")
#     elif isinstance(obj, h5py.Group):
#         print(f"Group: {name}")
# h5py.File("/mnt/MIG_Store/Datasets/faceforensicspp/Originalface.hdf5", "r"),

hf_files = [ 
    h5py.File("/mnt/MIG_Store/Datasets/faceforensicspp/FaceSwapface.hdf5", "r"),
    h5py.File("/mnt/MIG_Store/Datasets/faceforensicspp/Face2Faceface.hdf5", "r"),
    h5py.File("/mnt/MIG_Store/Datasets/faceforensicspp/NeuralTexturesface.hdf5", "r"),
]


def main():
    hf = hf_files[int(sys.argv[1])]
    temp_videos = [] 
    top = list(hf.keys())[0]
    folders = list(hf[top].keys())
    temp_folders = []
    for folder in folders:
        temp_folders.append(int(folder.split("_")[1]))
        temp_list = []
        for frame in range(40):
            temp_list.append(hf[f"{top}/{folder}/{frame}.jpg"][()])
        temp_videos.append(np.transpose(np.array(temp_list), (1,0, 2, 3)))
    folders = np.array(temp_folders)
    data = np.array(temp_videos)
    videos = np.zeros((data.shape[0], 3, 40, 128, 128))
    for i in range(1000):
        for j in range(3):
            for k in range(40):
                videos[i, j, k] = zoom(data[i, j, k], (128/160, 128/160), order=1)
    videos /= 255.0
    videos = videos[:,::-1,:,:,:]
    print(videos.shape)
    print(videos.dtype)
    print(folders.shape)
    print(folders.dtype)
    np.save(f"./FFS/{top}_40.npy", videos)
    np.save(f"./FFS/{top}_labels.npy", folders)


    
if __name__ == '__main__':
    main()