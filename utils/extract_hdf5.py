import h5py
import numpy as np
import cv2
# import os
# from PIL import Image
# from typing import Tuple


# def iter_obj(name, obj):
#     if isinstance(obj, h5py.Dataset):
#         print(f"Dataset: {name}")
#     elif isinstance(obj, h5py.Group):
#         print(f"Group: {name}")

hf = h5py.File("/mnt/MIG_Store/Datasets/faceforensicspp/Deepfakesface.hdf5", "r")
# hf = h5py.File("/mnt/MIG_Store/Datasets/faceforensicspp/Originalface.hdf5", "r")
# hf.visititems(iter_obj)
# print(np.array(hf["Deepfakes//000_003/0.jpg"][()]).shape)
# cv2.imwrite('test.jpg', np.transpose(hf["Deepfakes//000_003/0.jpg"][()], (1, 2, 0)))

def main():
    temp_videos = [] 
    folders = list(hf[f"Deepfakes"].keys())
    # folders = list(hf[f"Original"].keys())
    temp_list = []
    temp_folders = []
    for folder in folders:
        temp_folders.append(folder)
        for frame in range(40):
            temp_list.append(hf[f"Deepfakes/{folder}/{frame}.jpg"][()])
        temp_videos.append(np.transpose(np.array(temp_list), (1,0, 2, 3)))

    folders = np.array(temp_folders)
    videos = np.array(temp_videos)
    print(videos.shape)
    print(folders.shape)
    np.save("./deepfakes_40.npy", videos)
    np.save("./deepfakes_labels.npy", folders)
main()
# data = np.load("originals_40.npy")
# print(data.shape)
# new_data = np.transpose(data, (0, 4, 1, 2, 3))
# print(new_data.shape)
# np.save("new_originals_40.npy", new_data)