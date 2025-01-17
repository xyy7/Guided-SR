# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   options.py
@Time    :   2023/2/1 18:25
@Desc    :
"""
import argparse

import yaml

from set_config import set_config
from utils import init_distributed_mode, set_random_seed

parser = argparse.ArgumentParser(description="Config", add_help=False)
parser.add_argument(
    "-c", "--config", default="", type=str, metavar="FILE", help="YAML config file specifying default arguments"
)

args, remaining = parser.parse_known_args()

parser.add_argument("--device", default="cuda")
parser.add_argument("--sync_bn", action="store_true")
parser.add_argument("--seed", type=int, default=60)
parser.add_argument("--num_workers", type=int, default=0)

parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--opt", type=str, default="AdamW")
parser.add_argument("--loss", type=str, default="1*L1")
parser.add_argument("--hdelta", type=float, default=1)  # HuberLoss
parser.add_argument("--epochs", type=int, default=120)
parser.add_argument("--sched", default="multistep", type=str)  # cosine multistep
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--warmup_epochs", type=int, default=0)
parser.add_argument("--cooldown_epochs", type=int, default=10)
parser.add_argument("--min_lr", type=float, default=1e-5)
parser.add_argument("--warmup_lr", type=float, default=1e-5)  # warmup 初始的LR，warmup-epoch以后变为设定的lr
parser.add_argument("--decay_rate", type=float, default=0.5)
parser.add_argument("--decay_epochs", type=str, default="100")

parser.add_argument("--resume", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--test_only", action="store_true")
parser.add_argument("--distributed", action="store_true")
parser.add_argument("--start_epoch", type=int, default=0)

parser.add_argument("--scale", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--patch_size", type=int, default=256)
parser.add_argument("--val_batch_size", type=int, default=1)
parser.add_argument("--in_channels", type=int, default=3)

parser.add_argument("--model_name", type=str, default="Base2")
parser.add_argument("--embed_dim", type=int)

parser.add_argument("--load_name", type=str, default="model_best.pth")
parser.add_argument("--test_name", type=str, default="val")  # valid when test only or resume # no HR
parser.add_argument("--exp_name", type=str, default="")

parser.add_argument("--dist_url", default="env://")
parser.add_argument("--world_size", default=1, type=int)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--repeated_aug", action="store_true")

parser.add_argument("--save_result", action="store_true")
parser.add_argument("--tlc_enhance", action="store_true")
parser.add_argument("--print_freq", type=int, default=100)
parser.add_argument("--show_every", type=int)  # data times, 扩大epoch
parser.add_argument("--data_range", type=int, default=1)

# 先加载config
if args.config:
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)

# 再加载命令行传入
args = parser.parse_args(remaining)

set_random_seed(args.seed)
if args.distributed:
    init_distributed_mode(args)

set_config(args)
