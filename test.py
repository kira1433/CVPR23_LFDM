import numpy as np
from scipy.ndimage import zoom
from PIL import Image
import imageio

def convert_videos(videos):
    num_videos, num_frames, _, height, width = videos.shape
    new_num_frames = 40
    new_height, new_width = 128, 128
    
    # Calculate the frame indices to select
    frame_indices = np.linspace(0, num_frames - 1, new_num_frames, dtype=int)
    
    converted_videos = np.zeros((num_videos, 3, new_num_frames, new_height, new_width), dtype=videos.dtype)
    
    for i in range(num_videos):
        for j, frame_idx in enumerate(frame_indices):
            # Select the frame
            frame = videos[i, frame_idx]
            
            # Resize the frame using scipy.ndimage.zoom
            resized_frame = zoom(frame, zoom=(1, new_height / height, new_width / width), order=1)
            
            converted_videos[i, :, j] = resized_frame
    
    return converted_videos

videos = np.load("data_108.npy")
new_im_arr_list = []
for nf in range(40):
    save_tar_img = videos[0, nf, :, :, :]
    new_im = Image.new('RGB', (160,160))
    new_im.paste(Image.fromarray(save_tar_img, 'RGB'), (0, 0))
    new_im_arr = np.array(new_im)
    new_im_arr_list.append(new_im_arr)
new_vid_name = "test.gif"
new_vid_file = new_vid_name
imageio.mimsave(new_vid_file, new_im_arr_list)

