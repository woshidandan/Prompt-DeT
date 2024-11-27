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

device = torch.device("cuda:0")


def validate(model, loader):
    model.eval()
    validate_losses = AverageMeter()
    true_score = []
    pred_score = []
    pred_sub1 = []
    true_sub1 = []
    pred_sub2 = []
    true_sub2 = []
    pred_sub3 = []
    true_sub3 = []
    vbar = tqdm(loader, ncols=120)

    with torch.no_grad():
        for idx, (img, label) in enumerate(vbar):
            img = img.to(device)
            label = label.type(torch.FloatTensor).to(device)
            pred, sub_scores = model(img)
            score_pred = pred[:, 0]
            sub1_pred = sub_scores[:, 0]
            sub2_pred = sub_scores[:, 1]
            sub3_pred = sub_scores[:, 2]
            score = label[:, 0]
            sub1 = label[:, 1]
            sub2 = label[:, 2]
            sub3 = label[:, 3]
            # print(img.shape)
            # print(pred.shape)
            # print(sub_scores.shape)
            # print(sub_scores)
            pscore_np = score_pred.data.cpu().numpy().astype("float")
            tscore_np = score.data.cpu().numpy().astype("float")
            psub1_np = sub1_pred.data.cpu().numpy().astype("float")
            tsub1_np = sub1.data.cpu().numpy().astype("float")
            psub2_np = sub2_pred.data.cpu().numpy().astype("float")
            tsub2_np = sub2.data.cpu().numpy().astype("float")
            psub3_np = sub3_pred.data.cpu().numpy().astype("float")
            tsub3_np = sub3.data.cpu().numpy().astype("float")
            pred_score += pscore_np.tolist()
            true_score += tscore_np.tolist()
            pred_sub1 += psub1_np.tolist()
            true_sub1 += tsub1_np.tolist()
            pred_sub2 += psub2_np.tolist()
            true_sub2 += tsub2_np.tolist()
            pred_sub3 += psub3_np.tolist()
            true_sub3 += tsub3_np.tolist()
            # loss = criterion(score, score_pred)
            # validate_losses.update(loss.item(), img.size(0))
            # if not tsub3_np:
            # print(tsub3_np)
            # vbar.set_description(f"[epoch: {epoch}] [pred_score: {psub1_np}] [ture_score: {tsub1_np}]")

            # if writer is not None:
            #     # writer.add_scalar(f"test_loss", validate_losses.avg, global_step=global_step + idx)
            #     vbar.set_description(f"[epoch: {epoch}] [val_loss: {validate_losses.avg:.4f}]")

    srcc_mean, _ = spearmanr(pred_score, true_score)
    lcc_mean, _ = pearsonr(pred_score, true_score)
    srcc_mean1, _ = spearmanr(pred_sub1, true_sub1)
    lcc_mean1, _ = pearsonr(pred_sub1, true_sub1)
    srcc_mean2, _ = spearmanr(pred_sub2, true_sub2)
    lcc_mean2, _ = pearsonr(pred_sub2, true_sub2)
    srcc_mean3, _ = spearmanr(pred_sub3, true_sub3)
    lcc_mean3, _ = pearsonr(pred_sub3, true_sub3)

    true_score = np.array(true_score)
    true_score_lable = np.where(true_score <= 0.50, 0, 1)
    pred_score = np.array(pred_score)
    pred_score_lable = np.where(pred_score <= 0.50, 0, 1)

    true_sub1 = np.array(true_sub1)
    true_sub1_lable = np.where(true_sub1 <= 0.50, 0, 1)
    pred_sub1 = np.array(pred_sub1)
    pred_sub1_lable = np.where(pred_sub1 <= 0.50, 0, 1)

    true_sub2 = np.array(true_sub2)
    true_sub2_lable = np.where(true_sub2 <= 0.50, 0, 1)
    pred_sub2 = np.array(pred_sub2)
    pred_sub2_lable = np.where(pred_sub2 <= 0.50, 0, 1)

    true_sub3 = np.array(true_sub3)
    true_sub3_lable = np.where(true_sub3 <= 0.50, 0, 1)
    pred_sub3 = np.array(pred_sub3)
    pred_sub3_lable = np.where(pred_sub3 <= 0.50, 0, 1)

    acc = accuracy_score(true_score_lable, pred_score_lable)
    acc1 = accuracy_score(true_sub1_lable, pred_sub1_lable)
    acc2 = accuracy_score(true_sub2_lable, pred_sub2_lable)
    acc3 = accuracy_score(true_sub3_lable, pred_sub3_lable)
    print(f"[SROCC: {srcc_mean:.4f}] [LCC: {lcc_mean:.4f}] [Acc: {acc:.4f}] [val_loss: {validate_losses.avg:.4f}]")
    print(f"[SROCC: {srcc_mean1:.4f}] [LCC: {lcc_mean1:.4f}] [Acc: {acc1:.4f}] [val_loss: {validate_losses.avg:.4f}]")
    print(f"[SROCC: {srcc_mean2:.4f}] [LCC: {lcc_mean2:.4f}] [Acc: {acc2:.4f}] [val_loss: {validate_losses.avg:.4f}]")
    print(f"[SROCC: {srcc_mean3:.4f}] [LCC: {lcc_mean3:.4f}] [Acc: {acc3:.4f}] [val_loss: {validate_losses.avg:.4f}]")
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
    test_csv_path = os.path.join(opt["path_to_save_csv"], "test.csv")
    test_ds = ICAA17KDataset(test_csv_path, opt["path_to_images"], if_train=False)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=opt["num_workers"], shuffle=False)

    return test_loader


def start_val(opt):
    test_loader = create_data_part(opt)
    args, config = parse_option()
    print(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    if opt["path_to_model_weight"] != "":
        state_dict = torch.load(opt["path_to_model_weight"], map_location=device)
        model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    validate(model, test_loader)


if __name__ == "__main__":
    opt = option.init()
    params = vars(opt)
    print(params)
    start_val(params)
