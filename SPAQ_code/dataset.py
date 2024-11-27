import os
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
from typing import Sequence
import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
import pickle

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)


class MyRotateTransform:

    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class SPAQDataset(Dataset):

    def __init__(self, path_to_csv, images_path, if_train, rotate=False, hor=False, ver=False):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.if_train = if_train
        if if_train:
            self.transform = transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.Lambda(lambda img: MyRotateTransform([0, 90])(img) if rotate is True else img),
                transforms.Lambda(lambda img: transforms.RandomHorizontalFlip()(img) if hor is True else img),
                transforms.Lambda(lambda img: transforms.RandomVerticalFlip()(img) if ver is True else img),
                # transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                normalize
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row['Colorfulness'] / 10])
        image_id = row['image_id']
        image_path = os.path.join(self.images_path, f'{image_id}.pkl')

        with open(image_path, 'rb') as f:
            image = pickle.load(f)
        image = Image.fromarray(image)
        image = image.resize((224, 224))
        x = self.transform(image)

        return x, y.astype('float32')
