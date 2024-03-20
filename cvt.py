import cv2
import numpy as np

def video_to_npy(video_path):

    cap = cv2.VideoCapture(video_path)

    frames = []
    frame_count = 0

    while(cap.isOpened() and frame_count < 40):
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (128, 128))
        frames.append(frame_resized)
        frame_count+=1

    # Close the video file
    cap.release()

    # Convert the list of frames to a numpy array
    frames_array = np.array(frames)

    return frames_array

# Define paths to your videos
video1_path = '000.mp4'
video2_path = '003.mp4'

# Convert each video to numpy arrays
video1_array = video_to_npy(video1_path).astype(np.dtype('float32'))
print(f"Video1 shape: {video1_array.shape} | Datatype: {video1_array.dtype}")
video2_array = video_to_npy(video2_path).astype(np.dtype("float32"))
print(f"Video2 shape: {video2_array.shape}")

combined_array = np.stack((video1_array, video2_array))

combined_array = combined_array.reshape((2, 40, 3, 128, 128))
np.save('data_40.npy', combined_array)

data = np.load('combined_videos.npy')
data = np.transpose(data, (0, 2, 1, 3, 4))
np.save("data_40.npy", data)
print(f"Shape of npy: {data.shape}")
