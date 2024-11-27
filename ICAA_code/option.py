import argparse
import time


def init():
    parser = argparse.ArgumentParser(description="PyTorch")

    parser.add_argument("--path_to_images", type=str, default="/home/xiaoyi/datasets/ICAA20K", help="directory to images")
    parser.add_argument("--path_to_save_csv", type=str, default="./ICAA20K_csv", help="directory to csv_folder")
    parser.add_argument("--experiment_dir_name", type=str, default=f"./experiment/DAT+_ICAA20K_time{time.time():.2f}", help="directory to project")
    parser.add_argument(
        "--path_to_model_weight",
        type=str,
        default="./checkpoint/DAT+_ICAA_acc0.9647495361781077_srcc0.8967938935771766_lcc0.9111222137750093.pth",
        # default="",
        help="directory to pretrain model",
    )

    parser.add_argument("--init_lr", type=int, default=1e-5, help="learning_rate")
    parser.add_argument("--num_epoch", type=int, default=20, help="epoch num for train")
    parser.add_argument("--batch_size", type=int, default=8, help="16how many pictures to process one time")
    parser.add_argument("--num_workers", type=int, default=10, help="num_workers")
    parser.add_argument("--gpu_id", type=str, default="0", help="which gpu to use")

    args = parser.parse_args()
    return args


def init_finetune():
    parser = argparse.ArgumentParser(description="PyTorch")

    parser.add_argument("--path_to_images", type=str, default="/home/xiaoyi/datasets/ICAA20K", help="directory to images")
    parser.add_argument("--path_to_save_csv", type=str, default="./ICAA20K_csv", help="directory to csv_folder")
    parser.add_argument("--experiment_dir_name", type=str, default=f"./experiment/finetune/DAT+_ICAA20K_time{time.time():.2f}", help="directory to project")
    parser.add_argument(
        "--path_to_model_weight",
        type=str,
        default="./checkpoint/DAT+_ICAA_acc0.9647495361781077_srcc0.8967938935771766_lcc0.9111222137750093.pth",
        # default="",
        help="directory to pretrain model",
    )

    parser.add_argument("--init_lr", type=int, default=1e-5, help="learning_rate")
    parser.add_argument("--num_epoch", type=int, default=5, help="epoch num for train")
    parser.add_argument("--batch_size", type=int, default=8, help="16how many pictures to process one time")
    parser.add_argument("--num_workers", type=int, default=10, help="num_workers")
    parser.add_argument("--gpu_id", type=str, default="0", help="which gpu to use")

    args = parser.parse_args()
    return args
