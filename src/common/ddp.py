import os

import torch.distributed as dist


def setup_DDP(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_DDP():
    dist.destroy_process_group()
