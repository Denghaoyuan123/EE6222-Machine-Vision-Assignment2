import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import model
import glob
import time

def lowlight(image_path, model_path, result_path):
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight)/255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load(model_path))
    _, enhanced_image, _ = DCE_net(data_lowlight)

    if not os.path.exists(os.path.dirname(result_path)):
        os.makedirs(os.path.dirname(result_path))
    torchvision.utils.save_image(enhanced_image, result_path)

def process_dataset(dataset_path, result_root, model_path):
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_path = os.path.join(root, file)
                result_path = image_path.replace(dataset_path, result_root).replace('\\', '/')
                #print(f'Processing {image_path}')
                lowlight(image_path, model_path, result_path)

if __name__ == '__main__':
    model_path = 'snapshots/Epoch99.pth'
    train_data_path = 'D:/Human-Action-Recognition-In-The-Dark-master/Human-Action-Recognition-In-The-Dark-master/EE6222_data/train_img_5'
    val_data_path = 'D:/Human-Action-Recognition-In-The-Dark-master/Human-Action-Recognition-In-The-Dark-master/EE6222_data/validate_img_5'
    result_train_root = 'D:/Human-Action-Recognition-In-The-Dark-master/Human-Action-Recognition-In-The-Dark-master/EE6222_data/enhanced_train_img_7'
    result_val_root = 'D:/Human-Action-Recognition-In-The-Dark-master/Human-Action-Recognition-In-The-Dark-master/EE6222_data/enhanced_validate_img_7'

    with torch.no_grad():
        process_dataset(train_data_path, result_train_root, model_path)
        process_dataset(val_data_path, result_val_root, model_path)
