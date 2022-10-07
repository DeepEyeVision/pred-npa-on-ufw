import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="config/base.yaml")

    # message
    parser.add_argument("-m", "--message", type=str, default=None)
    parser.add_argument("--debug", action="store_true")

    # dataset
    parser.add_argument("-c", "--classes", default=2, help="number of classes")
    parser.add_argument(
        "--img_types",
        type=str,
        nargs="+",
        default=["npa", "non_npa"],
    )
    parser.add_argument(
        "--annotators",
        type=str,
        nargs="+",
        default=["tampo"],
    )
    parser.add_argument("--read_sub_img", action="store_true")
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--base_size", type=int)
    parser.add_argument("--num_non_npa_epoch", type=int, default=0)

    # model
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--in_channel", type=int, default=3)
    parser.add_argument("--nncf_config", type=str, default=None)

    # logging and train
    parser.add_argument(
        "--learning_method",
        type=str,
        default="finetune",
        choices=["transfer", "finetune"],
    )
    parser.add_argument(
        "--batch_size", type=int, default=48, help="batch size"
    )
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-t", "--trials", type=int, default=10, help="number of trials")
    parser.add_argument("--gpus", type=str, default="-1")
    parser.add_argument("--validate_freq", type=int, default=4)
    parser.add_argument("--summarize_freq", type=int, default=4)
    parser.add_argument("--logdir", type=str, default="../logs")
    parser.add_argument("--master_log", type=str, default="../logs/EXPERIMENTS.md")
    parser.add_argument("-s", "--storage", type=str)
    parser.add_argument("--num_workers", type=int, default=4)

    # metrics and visualization
    parser.add_argument("--compare_gt", action="store_true")
    parser.add_argument("--show_perf", action="store_true")
    parser.add_argument("--save_res_indiv", action="store_true")
    parser.add_argument("--save_pics", action="store_true")
    parser.add_argument("--gif", action="store_true")

    return parser.parse_args()
