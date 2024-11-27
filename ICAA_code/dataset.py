import os
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)


class ICAA17KDataset(Dataset):
    def __init__(self, path_to_csv, images_path, if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path = images_path
        self.if_train = if_train
        self.mean = 1.0 / 2.0
        self.std = 1.0 / 6.0
        self.means = [
            0.5891324490297224,
            0.5935361336281012,
            0.559279390813068,
            0.71267963645296,
        ]

        self.stds = [
            0.0043606940609912485,
            0.15503706305078224,
            0.2304456269826001,
            0.11940672015503832,
        ]

        if if_train:
            self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomHorizontalFlip(), transforms.RandomCrop((224, 224)), transforms.ToTensor(), normalize])
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        score_names = ["holistic_color", "temperature", "colorfulness", "color_harmony"]
        # score_names = ["color"]
        y = np.array([self.norm(row[k] / 10, score_names.index(k)) for k in score_names])

        image_id = row["ID"]
        image_path = os.path.join(self.images_path, f"{image_id}")
        image = default_loader(image_path)

        x = self.transform(image)

        return x, y.astype("float32")

    def norm(self, x, idx):
        if idx != 0:
            return (x - self.means[idx]) * (self.std / self.stds[idx]) + self.mean
        else:
            return x
