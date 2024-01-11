import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def extract_frames(video_path, sample_interval=5, num_frames=16, random_sampling=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"No frames found in video: {video_path}")
        return []

    if random_sampling:
        frame_indices = random.sample(range(total_frames), min(num_frames, total_frames))
    else:
        frame_indices = [i * sample_interval for i in range(num_frames)]

    frames = []
    for i in range(max(frame_indices) + 1):
        ret, frame = cap.read()
        if i in frame_indices and ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elif i in frame_indices:
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frames.append(np.zeros((frame_height, frame_width, 3), dtype=np.uint8))

    cap.release()
    return frames

def plot_frames(frames, title):
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    axs = axs.flatten()
    for i in range(16):
        if i < len(frames) and frames[i].size != 0:
            axs[i].imshow(frames[i])
        else:
            axs[i].imshow(np.zeros((100, 100, 3), dtype=np.uint8))  # 显示一个黑色的空图像
        axs[i].axis('off')
        axs[i].set_title(f'frame_{i+1}')  # 为每个子图添加标题
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# 选择一个视频文件
video_path = 'D:\\Human-Action-Recognition-In-The-Dark-master\\Human-Action-Recognition-In-The-Dark-master\\EE6222_data\\train\\Jump\\Jump_8_5.mp4'  # 替换为您的视频文件路径

# 均匀采样
uniform_frames = extract_frames(video_path, sample_interval=5, num_frames=16, random_sampling=False)
plot_frames(uniform_frames, "Uniform Sampling")

# 随机采样
random_frames = extract_frames(video_path, num_frames=16, random_sampling=True)
plot_frames(random_frames, "Random Sampling")