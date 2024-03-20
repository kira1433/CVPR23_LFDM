import numpy as np
from scipy.ndimage import zoom
from PIL import Image
import imageio
import cv2

videos = np.load("originals_40.npy")
new_im_arr_list = []
for nf in range(40):
    save_tar_img = videos[0, :, 40, :, :]
    new_im = Image.new('RGB', (160,160))
    new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, 0))
    new_im_arr = np.array(new_im)
    new_im_arr_list.append(new_im_arr)
new_vid_name = "test.gif"
new_vid_file = new_vid_name
imageio.mimsave(new_vid_file, new_im_arr_list)

