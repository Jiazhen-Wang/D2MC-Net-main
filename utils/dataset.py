import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
transforms=transforms.Compose([
    transforms.ToTensor()
])
from torch.utils.data import DataLoader
import scipy.io as sio

class MyDataset(Dataset):
    def __init__(self,clean_path,motion_path):
        self.clean_path=clean_path
        self.motion_path = motion_path
        self.name = os.listdir(clean_path)


    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        image_name=self.name[index]
        clean_image_path = os.path.join(self.clean_path,  image_name)
        motion_image_path = os.path.join(self.motion_path, image_name)
        img_clean = sio.loadmat(clean_image_path)[os.path.splitext(image_name)[0]]
        img_motion = sio.loadmat(motion_image_path)[os.path.splitext(image_name)[0]]

        return transforms(img_clean),transforms(img_motion),image_name

def load_data(batch_size,clean_path,motion_path):
    dataset=MyDataset(clean_path,motion_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,pin_memory=True, drop_last=True)
    return data_loader



