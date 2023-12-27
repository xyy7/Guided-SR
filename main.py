# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   main.py
@Time    :   2022/3/1 20:06
@Desc    :
"""

import json
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter

import loss
import utils
from data import get_loader
from models import get_model
from options import args
from scheduler import create_scheduler
from trainer import evaluate, train_one_epoch
from utils import make_optimizer, master_only, set_checkpoint_and_log_dir


def main():
    ## init
    device = torch.device("cuda")
    set_checkpoint_and_log_dir(args)
    writer = SummaryWriter("./logs/{}/{}".format(args.dataset, args.exp_name))
    model = get_model(args)
    model.to(device)
    criterion = loss.Loss(args)
    optimizer = make_optimizer(args, model)
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    if args.debug:
        print("args:", args)
        print(criterion.loss)
        print(optimizer)
        print(lr_scheduler)
    if args.distributed:
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    print("===> Parameter Number:", utils.get_parameter_number(model_without_ddp))

    ## test
    ckpt_path = f"./checkpoints/{args.dataset}/{args.exp_name}"
    if args.resume or args.test_only:
        model_path = args.load_name if os.path.exists(args.load_name) else f"{ckpt_path}/{args.load_name}"
        try:
            checkpoint = torch.load(model_path, map_location="cuda:{}".format(args.local_rank))
            model_without_ddp.load_state_dict(checkpoint["model"])
            if args.resume:
                args.start_epoch = checkpoint["epoch"] + 1
                lr_scheduler.step(args.start_epoch)
                # optimizer.load_state_dict(checkpoint['optimizer'])
        except FileNotFoundError:
            print("===> File {} not exists".format(model_path))
        else:
            print("===> File {} loaded".format(model_path))
        evaluate(model, criterion, args.test_name, device=device, val_data=get_loader(args, args.test_name), args=args)

    ## train
    train_data = get_loader(args, "train")
    if not args.test_only:
        for epoch in range(args.start_epoch, num_epochs):
            if args.distributed:
                train_data.sampler.set_epoch(epoch)
            train_stats = train_one_epoch(model, criterion, train_data, optimizer, device, epoch, args)
            log_stats = {**{f"TRAIN_{k}".upper(): v for k, v in train_stats.items()}}

            test_stats = evaluate(model, criterion, "val", device=device, val_data=get_loader(args, "val"), args=args)
            log_stats.update({**{f"{k}".upper(): v for k, v in test_stats.items()}})

            # test_stats = evaluate(model, criterion, "test", device=device, val_data=get_loader(args, "test"), args=args) # no HR
            # log_stats.update({**{f"{k}".upper(): v for k, v in test_stats.items()}, "EPOCH": epoch})
            print(log_stats)

            if args.local_rank == 0:
                [writer.add_scalar(k.replace("_", "/"), v, epoch) for k, v in log_stats.items() if k != "EPOCH"]
                with open("./logs/{}/{}/{}_log.txt".format(args.dataset, args.exp_name, time.time()), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            if args.debug or (epoch + 1) % 10 == 0:
                utils.save_on_master(
                    {
                        "optimizer": optimizer.state_dict(),
                        "model": model_without_ddp.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    "{}/model_{}.pth".format(ckpt_path, str(epoch + 1).zfill(6)),
                )
                if args.debug:
                    exit()
            lr_scheduler.step(epoch)


if __name__ == "__main__":
    main()
