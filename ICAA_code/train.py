import os
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
from models.build import build_model
import torch.nn as nn
from dataset import ICAA17KDataset
from utils import AverageMeter
import option
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
import csv

warnings.filterwarnings("ignore")

device = torch.device("cuda:1")


def train(model, loader, optimizer, criterion, writer=None, global_step=None, epoch=None):
    model.train()
    train_losses = AverageMeter()
    tbar = tqdm(loader, ncols=120)

    for idx, (img, score) in enumerate(tbar):
        img = img.to(device)
        score = score.to(device)

        score_pred, _ = model(img)

        loss = criterion(score, score_pred)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        train_losses.update(loss.item(), img.size(0))
        lr = optimizer.state_dict()["param_groups"][0]["lr"]

        if writer is not None:
            writer.add_scalar(f"train_loss", train_losses.avg, global_step + idx)
            tbar.set_description(f"[epoch: {epoch}] [train_loss: {train_losses.avg:.4f}] [lr: {lr}]")

    return train_losses.avg


def validate(model, loader, criterion, writer=None, global_step=None, epoch=None):
    model.eval()
    validate_losses = AverageMeter()
    true_score = []
    pred_score = []
    vbar = tqdm(loader, ncols=120)

    with torch.no_grad():
        for idx, (img, score) in enumerate(vbar):
            img = img.to(device)
            score = score.type(torch.FloatTensor)
            score = score.to(device)

            pred, _ = model(img)

            score_pred = pred[:, 0]
            score = score[:, 0]

            pscore_np = score_pred.data.cpu().numpy().astype("float")
            tscore_np = score.data.cpu().numpy().astype("float")

            pred_score += pscore_np.tolist()
            true_score += tscore_np.tolist()

            loss = criterion(score, score_pred)
            validate_losses.update(loss.item(), img.size(0))

            if writer is not None:
                writer.add_scalar(f"test_loss", validate_losses.avg, global_step=global_step + idx)
                vbar.set_description(f"[epoch: {epoch}] [val_loss: {validate_losses.avg:.4f}]")

    srcc_mean, _ = spearmanr(pred_score, true_score)
    lcc_mean, _ = pearsonr(pred_score, true_score)

    true_score = np.array(true_score)
    true_score_lable = np.where(true_score <= 0.50, 0, 1)
    pred_score = np.array(pred_score)
    pred_score_lable = np.where(pred_score <= 0.50, 0, 1)

    acc = accuracy_score(true_score_lable, pred_score_lable)
    print(f"[epoch: {epoch}] [SROCC: {srcc_mean:.4f}] [LCC: {lcc_mean:.4f}] [Acc: {acc:.4f}] [val_loss: {validate_losses.avg:.4f}]")
    return validate_losses.avg, acc, lcc_mean, srcc_mean


def parse_option():
    import argparse
    from config import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, metavar="FILE", help="path to config file", default="configs/dat_ext.yaml")
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs="+")
    # easy config modification
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument("--resume", help="resume from checkpoint", default="")
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--pretrained", type=str, help="Finetune 384 initial checkpoint.", default="")

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config


def create_data_part(opt):
    train_csv_path = os.path.join(opt["path_to_save_csv"], "train.csv")
    test_csv_path = os.path.join(opt["path_to_save_csv"], "test.csv")

    train_ds = ICAA17KDataset(train_csv_path, opt["path_to_images"], if_train=True)
    test_ds = ICAA17KDataset(test_csv_path, opt["path_to_images"], if_train=False)

    train_loader = DataLoader(train_ds, batch_size=opt["batch_size"], num_workers=opt["num_workers"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=opt["num_workers"], shuffle=False)

    return train_loader, test_loader


def save_model(state_dict, save_path):
    model_dict = state_dict.copy()
    for k, v in state_dict.items():
        if ("icaa_clip" in k) and ("adapter" not in k):
            del model_dict[k]
    torch.save(model_dict, save_path)


def start_train(opt):
    train_loader, test_loader = create_data_part(opt)
    args, config = parse_option()
    print(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    if opt["path_to_model_weight"] != "":
        state_dict = torch.load(opt["path_to_model_weight"], map_location="cuda:1")
        model.load_state_dict(state_dict, strict=False)
    model.to("cuda:1")

    # freeze
    # for k, v in model.named_parameters():
    #     if ("adapter" in k) or ("alpha" in k):
    #         v.requires_grad_(True)
    #     else:
    #         v.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt["init_lr"], eps=1e-8)
    criterion = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer=optimizer, mode="max", factor=0.1, patience=10)

    model = model.to(device)
    criterion.to(device)

    writer = SummaryWriter(log_dir=os.path.join(opt["experiment_dir_name"], "logs"))
    ckpt_dir = os.path.join(opt["experiment_dir_name"], "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    acc_best = 0.0
    lcc_best = 0.0
    srcc_best = 0.0
    train_loss = 0.0

    for epoch in range(opt["num_epoch"]):
        train_loss = train(model, train_loader, optimizer, criterion, writer, epoch * len(train_loader), epoch)
        test_loss, tacc, tlcc, tsrcc = validate(model, test_loader, criterion, writer, epoch * len(test_loader), epoch)

        if tsrcc > srcc_best or tlcc > lcc_best or tacc > acc_best:
            srcc_best = max(tsrcc, srcc_best)
            lcc_best = max(tlcc, lcc_best)
            acc_best = max(tacc, acc_best)
            if tsrcc > 0.7:
                model_name = f"e_{epoch}_acc{tacc}_srcc{tsrcc}_lcc{tlcc}.pth"
                save_model(model.state_dict(), os.path.join(ckpt_dir, model_name))

        scheduler.step(srcc_best)

        writer.add_scalars("epoch_loss", {"train": train_loss, "val": test_loss}, global_step=epoch)
        writer.add_scalars("lcc_srcc", {"val_lcc": tlcc, "val_srcc": tsrcc}, global_step=epoch)
        writer.add_scalars("acc", {"val_acc": tacc}, global_step=epoch)

    save_model(model.state_dict(), os.path.join(opt["experiment_dir_name"], "model_latest.pth"))
    writer.close()


if __name__ == "__main__":
    opt = option.init()
    params = vars(opt)
    print(params)
    start_train(params)
