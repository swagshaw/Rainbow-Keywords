"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/2/4 下午10:55
@Author  : Yang "Jan" Xiao 
@Description : main script
"""
import logging
import logging.config
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from configuration import config
from utils.data_loader import get_train_datalist, get_test_datalist
from utils.method_manager import select_method


def main():
    args = config.base_parser()
    # args.debug = True
    if args.debug:
        args.n_epoch = 1
    # Save file name
    tr_names = ""
    # for trans in args.transforms:  # multiple choices: cutmix, cutout, randaug, autoaug
    #     tr_names += "_" + trans
    save_path = f"{args.dataset}/{args.mode}_msz{args.memory_size}_rnd{args.rnd_seed}{tr_names}"
    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()
    os.makedirs(f"logs/{args.dataset}", exist_ok=True)
    fileHandler = logging.FileHandler("logs/{}.log".format(save_path), mode="w")
    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    writer = SummaryWriter("tensorboard")

    if torch.cuda.is_available() and args.debug is False:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Set the device ({device})")
    # Fix the random seeds
    # https://hoya012.github.io/blog/reproducible_pytorch/
    torch.manual_seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    logger.info(f"[1] Select a CIL method ({args.mode})")
    criterion = nn.CrossEntropyLoss(reduction="mean")
    n_classes = 30
    method = select_method(
        args, criterion, device, n_classes
    )

    logger.info(f"[2] Incrementally training {args.n_tasks} tasks")
    task_records = defaultdict(list)
    start_time = time.time()
    # start to train each tasks
    for cur_iter in range(args.n_tasks):
        if args.mode == "joint" and cur_iter > 0:
            return
        print("\n" + "#" * 50)
        print(f"# Task {cur_iter} iteration")
        print("#" * 50 + "\n")
        logger.info("[2-1] Prepare a datalist for the current task")
        task_acc = 0.0
        eval_dict = dict()

        # get datalist
        cur_train_datalist = get_train_datalist(args, cur_iter)
        cur_test_datalist = get_test_datalist(args, args.exp_name, cur_iter)
        # Reduce datalist in Debug mode
        if args.debug:
            random.shuffle(cur_train_datalist)
            random.shuffle(cur_test_datalist)
            cur_train_datalist = cur_train_datalist[:2560]
            cur_test_datalist = cur_test_datalist[:2560]

        logger.info("[2-2] Set environment for the current task")
        method.set_current_dataset(cur_train_datalist, cur_test_datalist)
        # Increment known class for current task iteration.
        if args.mode == "bic" or args.mode == "gdumb":
            method.before_task(datalist=cur_train_datalist, init_model=False, init_opt=True, cur_iter=cur_iter)
        else:
            method.before_task(datalist=cur_train_datalist, init_model=False, init_opt=True)
        # The way to handle streamed samles
        logger.info(f"[2-3] Start to train under {args.stream_env}")
        if args.stream_env == "offline":
            # Offline Train
            task_acc, eval_dict = method.train(
                cur_iter=cur_iter,
                n_epoch=args.n_epoch,
                batch_size=args.batchsize,
                n_worker=args.n_worker,
            )
            if args.mode == "joint":
                logger.info(f"joint accuracy: {task_acc}")
        logger.info("[2-4] Update the information for the current task")
        method.after_task(cur_iter)
        task_records["task_acc"].append(task_acc)
        # task_records['cls_acc'][k][j] = break down j-class accuracy from 'task_acc'
        task_records["cls_acc"].append(eval_dict["cls_acc"])
        if cur_iter > 0:
            task_records["bwt_list"].append(np.mean(
                [task_records["task_acc"][i + 1] - task_records["task_acc"][i] for i in
                 range(len(task_records["task_acc"]) - 1)]))
        # Notify to NSML
        logger.info("[2-5] Report task result")
        writer.add_scalar("Metrics/TaskAcc", task_acc, cur_iter)
    np.save(f"results/{save_path}.npy", task_records["task_acc"])
    # Total time (T)
    duration = time.time() - start_time

    # Accuracy (A)
    A_avg = np.mean(task_records["task_acc"])
    A_last = task_records["task_acc"][args.n_tasks - 1]

    # Forgetting (F)
    acc_arr = np.array(task_records["cls_acc"])
    # cls_acc = (k, j), acc for j at k
    cls_acc = acc_arr.reshape(-1, args.n_cls_a_task).mean(1).reshape(args.n_tasks, -1)
    for k in range(args.n_tasks):
        forget_k = []
        for j in range(args.n_tasks):
            if j < k:
                forget_k.append(cls_acc[:k, j].max() - cls_acc[k, j])
            else:
                forget_k.append(None)
        task_records["forget"].append(forget_k)
    F_last = np.mean(task_records["forget"][-1][:-1])

    # Intrasigence (I)
    I_last = args.joint_acc - A_last

    logger.info(f"======== Summary =======")
    logger.info(f"Total time {duration}, Avg: {duration / args.n_tasks}s")
    logger.info(f'BWT: {np.mean(task_records["bwt_list"])}, std: {np.std(task_records["bwt_list"])}')
    logger.info(f"A_last {A_last} | A_avg {A_avg} | F_last {F_last} | I_last {I_last}")
    writer.close()


if __name__ == "__main__":
    main()
