import cv2
import numpy as np
import torch 
import glob
import os
import albumentations as a
from pathlib import Path
from config.constants import N_FRAMES, INTERVAL, IMG_DIM, IMG_SHAPE, OPENPOSE_STATE_DIR
from torch.utils.data import Dataset
from .openpose_pytorch.src.body import Body_HM

# Dataset class with shape (T, C, H, W)

class VideoDataset(Dataset):
    def __init__(self, df, transforms, img_path_col="path", labelAvailable=True):    
        self.df = df  
        self.transforms = transforms #存储应用于视频的帧变换
        self.img_path_col = img_path_col #存储表示图像路径的列名字
        self.labelAvailable = labelAvailable #是否有可用的数据标签

    def __len__(self):
        print("here 1") #反复收集长度和大小
        return len(self.df) #返回数据集的样本数

    def __getitem__(self, idx): #obtain single sample

        # Video path
        img_folder = self.df[self.img_path_col].iloc[idx] + "/*.jpg"
        #print(img_folder) 输出处理文件夹
        # Append path of all frames in a video
        all_img_path = glob.glob(img_folder)
        all_img_path = sorted(all_img_path)
        v_len = len(all_img_path)
        #print(v_len)文件夹中的照片总数
        
        # Uniformly samples N_FRAMES number of frames   
        if v_len > N_FRAMES*INTERVAL:
            frame_list = np.arange(np.int64((v_len - (N_FRAMES*INTERVAL))*0.5), np.int64((v_len - (N_FRAMES*INTERVAL))*0.5) + N_FRAMES*INTERVAL, INTERVAL)
        else:
            frame_list = np.arange(0, N_FRAMES*INTERVAL, INTERVAL) 
        #print("here 2") process文件夹下的12帧
        img_path = []
        for fn in range(v_len):
            if (fn in frame_list):
                img_path.append(all_img_path[fn])
        #print(img_path)
        #print(img_path) output12帧的path       
        images = []
        for p2i in img_path:
            p2i = Path(p2i)
            img = cv2.imread(p2i.as_posix())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        #print("here 3") #after 加载完所有图像RGB形式
        while (len(images) < N_FRAMES):
            #if not enough ,the black screen is added for the rest of the frames
            images.append(np.zeros((IMG_DIM,IMG_DIM,3), np.uint8))
        images_tr = []

        body_estimation_hm = Body_HM(OPENPOSE_STATE_DIR)

        for image in images:
            #print("here 4") #依次处理12帧
            if len(images_tr) == 0:
                augmented = self.transforms(image=image)
                image = augmented['image']
                body_heatmap = body_estimation_hm(image)
                hm = torch.Tensor((1 - body_heatmap[:,:,-1]))
                hm = torch.stack((hm,hm,hm), dim = 0)
                images_tr.append(torch.Tensor(hm))
                data_replay = augmented['replay']
                
            else:
                image = a.ReplayCompose.replay(data_replay, image=image)
                image = image['image']
                body_heatmap = body_estimation_hm(image)
                hm = torch.Tensor((1 - body_heatmap[:,:,-1]))
                hm = torch.stack((hm,hm,hm), dim = 0)
                images_tr.append(torch.Tensor(hm))

        if len(images_tr)>0:
            images_tr = torch.stack(images_tr)   
        #print("here 5") #处理完12帧
        if self.labelAvailable == True:
            # Label
            label = self.df["label"].iloc[idx]
            return img_folder, images_tr, label
        else:
            return img_folder, images_tr

# Train image transformation function
# 定义训练时使用的图像变换，包括增强对比度、调整伽马值、填充、裁剪、翻转和旋转。
def get_train_transforms(img_dim):
    trans = a.ReplayCompose([
        a.CLAHE(clip_limit=(10, 10), tile_grid_size=(8, 8), always_apply=True),
        a.RandomGamma((150, 150), always_apply=True),
        a.PadIfNeeded(img_dim, img_dim),
        a.CenterCrop(img_dim, img_dim, always_apply=True),
        a.HorizontalFlip(p=0.4),
        a.VerticalFlip(p=0.5),
        a.Rotate(limit=120, p=0.8),

    ])
    #print("transfered")
    return trans

# Validation image transformation function
def get_val_transforms(img_dim):
    trans = a.ReplayCompose([
        a.CLAHE(clip_limit=(10, 10), tile_grid_size=(8, 8), always_apply=True),
        a.RandomGamma((150, 150), always_apply=True),
        a.PadIfNeeded(img_dim, img_dim),
        a.CenterCrop(img_dim, img_dim, always_apply=True),
    ])
    #print("transformed-val")
    return trans

#diy COLLATE function 
def collate_fn(batch):
    if len(batch[0]) == 3:
        #print("collate-1") 长度为3，则每个样本包含图像文件夹路径、图像数据和标签
        img_folder_batch, imgs_batch, label_batch = list(zip(*batch))
        label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
        labels_tensor = torch.stack(label_batch)
    else:
        #print("collate-2") 推理或测试阶段，不需要标签
        img_folder_batch, imgs_batch = list(zip(*batch))   
    print("collate-3")    
    img_folder_batch = [folders for folders in img_folder_batch if len(folders)>0]
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    
    #imgs_tensor = torch.stack(torch.Tensor(imgs_batch))
    if IMG_SHAPE == 'BCTHW':
        #print("if") 转置处理BCTHW 批次 通道 时间 高度 宽度
        imgs_batch = [torch.transpose(imgs, 1, 0) for imgs in imgs_batch if len(imgs)>0]
    elif IMG_SHAPE == 'BTCHW':
        imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    #print("collect-5") 
    imgs_tensor = torch.stack(imgs_batch)
    
    if len(batch[0]) == 3:
        #print("collate-6") 返回包含标签的数据，需要标签来计算损失 评估性能
        return img_folder_batch, imgs_tensor,labels_tensor
    else:
        return img_folder_batch, imgs_tensor