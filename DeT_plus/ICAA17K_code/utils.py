import os
import requests

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Gl_z = torch.ones(64, 10)


def download_file(url, local_filename, chunk_size=1024):
    if os.path.exists(local_filename):
        return local_filename
    r = requests.get(url, stream=True)
    with open(local_filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return local_filename


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EDMLoss(nn.Module):
    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        cdf_target = torch.cumsum(p_target, dim=1)
        cdf_estimate = torch.cumsum(p_estimate, dim=1)

        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))  # train

        return samplewise_emd.mean()


def print_dict(model_path):
    # 加载.pth文件中的模型
    state_dict = torch.load(model_path)
    for k, v in state_dict.items():
        # print(k)
        if "alpha" in k:
            print(v)


def plot_csv(csv_path=""):
    df = pd.read_csv(csv_path)
    data = []
    for k, v in df.items():
        data.append(v)
    x = data[0]
    y = data[1]
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.scatter(np.array(x), np.array(y), s=20, alpha=0.5)
    plt.savefig(csv_path.split(".")[0])
    # plt.savefig("softmax_logit_scale")


if __name__ == "__main__":
    epoch = 7
    csv_path = f"PCCD_test_result_epoch{epoch}.csv"
    plot_csv(csv_path)
