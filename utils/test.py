import numpy as np
from scipy.ndimage import zoom
from PIL import Image
import imageio
import cv2

# data = np.load("FFS/originals_40.npy")
# images = np.zeros((1000, 3, 128, 128))
# for i in range(1000):
#     for j in range(3):
#         images[i, j] = data[i, j, 0]
# print(images.shape)
# np.save("FFS/originals_images.npy", images)

#resize to 128x128
# def img_zoom():
    # np.save("FFS/originals_40.npy", videos)
    # videos = np.zeros((999, 3, 40, 128, 128))
    # for i in range(999):
    #     for j in range(3):
    #         for k in range(40):
    #             videos[i, j, k] = zoom(data[i, j, k], (128/160, 128/160), order=1)

    # np.save("FFS/deepfakes_40.npy", videos)


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
