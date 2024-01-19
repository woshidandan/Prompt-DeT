from thop import profile
import torch
from models.build import build_model
from models.dat_ext import ICAA_CLIP, DAT
import argparse
from config import get_config
from pytorch_model_summary import summary
from tqdm import tqdm
import os
import torch
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import time
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import option


def parse_option():
    import argparse
    from config import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, metavar="FILE", default="./configs/dat_ext.yaml")
    parser.add_argument("--opts", default=None, nargs="+")
    # easy config modification
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--resume", default="")
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--pretrained", type=str, default="")

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config


def test():
    output_folder = f"./test/color_factor_{time.time()/100:.0f}"  # replace by your output image folder
    result_list = [
        "aesthetic",
        "colorfulness",
        "temperature",
        "harmony",
    ]
    for i in result_list:
        os.makedirs(os.path.join(output_folder, i), exist_ok=True)

    # replace by your test image folder
    data_folder = "/home/xiaoyi/datasets/ICAA17K/color_factors"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_scores = []

    for filename in tqdm(os.listdir(data_folder), ncols=100):
        filepath = os.path.join(data_folder, filename)
        image = Image.open(filepath)
        img = data_transform(image).unsqueeze(0)
        img = img.to(device)
        color_score, sub_scores = model(img)
        color_score = color_score.squeeze(0).data.cpu().numpy()
        sub_scores = sub_scores.squeeze(0).data.cpu().numpy()
        for i, sub_score in enumerate(sub_scores):
            image.save(
                os.path.join(
                    output_folder,
                    result_list[i],
                    f"{result_list[i]}{sub_score:.4f}_color{color_score[0]:.4f}_{filename}",
                )
            )
        image_scores.append((filename, color_score, sub_scores))

    opt_dir = "./test"  # replace by your output dir
    os.makedirs(opt_dir, exist_ok=True)
    csv_file_path = os.path.join(opt_dir, f"color_attributes_{time.time()/100:.0f}.csv")  # replace by your output csv path
    with open(csv_file_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ["Image Path"] + [f"color_scores"] + [f"{result_list[i]}_scores" for i in range(len(image_scores[0][2]))]
        csv_writer.writerow(header)
        for image_path, color_score, sub_scores in image_scores:
            row = [image_path] + color_score.tolist() + sub_scores.tolist()
            csv_writer.writerow(row)

    print(f"Result CSV file has been saved to {csv_file_path}.")


if __name__ == "__main__":
    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # replace by your test image folder
    data_folder = "/home/xiaoyi/datasets/ICAA17K"
    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")  # replace by yours
    opt = option.init()
    opt = vars(opt)
    args, config = parse_option()
    print(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    if opt["path_to_model_weight"] != "":
        state_dict = torch.load(opt["path_to_model_weight"], map_location="cuda:1")  # replace by yours
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    test()
