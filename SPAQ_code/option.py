import argparse
import time


def init():
    parser = argparse.ArgumentParser(description="PyTorch")

    parser.add_argument("--path_to_images", type=str, default="/home/xiaoyi/datasets/SPAQ_process", help="directory to images")
    parser.add_argument("--path_to_save_csv", type=str, default="./SPAQ_csv", help="directory to csv_folder")
    parser.add_argument("--experiment_dir_name", type=str, default=f"./experiment/DAT+_SPAQ_time{time.time():.2f}", help="directory to project")
    parser.add_argument(
        "--path_to_model_weight",
        type=str,
        default="./checkpoint/DAT+_SPAQ_acc0.8533333333333334_srcc0.7808414373522549_lcc0.8094436890361592.pth",
        help="directory to pretrain model",
    )

    parser.add_argument("--init_lr", type=int, default=1e-6, help="learning_rate")
    parser.add_argument("--num_epoch", type=int, default=60, help="epoch num for train")
    parser.add_argument("--batch_size", type=int, default=32, help="16how many pictures to process one time")
    parser.add_argument("--num_workers", type=int, default=6, help="num_workers")
    parser.add_argument("--gpu_id", type=str, default="0", help="which gpu to use")

    args = parser.parse_args()
    return args
