import numpy as np
from scipy.ndimage import zoom
from PIL import Image
import imageio
import cv2

data = np.load("FFS/originals_40.npy")


def sample_img(rec_img_batch):
    rec_img = np.transpose(rec_img_batch,(1, 2, 0)).copy()
    rec_img += np.array((0.0, 0.0, 0.0))/255.0
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    return np.array(rec_img, np.uint8)

def save_gif(data,new_vid_file = 'test.gif'):
    new_im_arr_list = []
    for nf in range(40):
        save_tar_img = sample_img(data[:, nf, :, :])
        new_im = Image.new('RGB', (128,128))
        new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, 0))
        new_im_arr = np.array(new_im)
        new_im_arr_list.append(new_im_arr)
    imageio.mimsave(new_vid_file, new_im_arr_list)

new_im = Image.fromarray(sample_img(data[0,:,0]))

for i in range(1000):
    save_gif(data[i],str(i) + ".gif")
