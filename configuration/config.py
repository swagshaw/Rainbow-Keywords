"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/1/25 下午10:25
@Author  : Yang "Jan" Xiao 
@Description :

"""
import argparse


def base_parser():
    parser = argparse.ArgumentParser(description="Class Incremental Learning Research")
    # Data root.
    parser.add_argument("--data_root", type=str, default='/home/xiaoyang/Dev/kws-efficient-cl/dataset/collection')
    # Mode and Exp. Settings.
    parser.add_argument(
        "--mode",
        type=str,
        default="rainbow_keywords",
        help="CIL methods [finetune ,native_rehearsal,joint, rwalk, icarl, rainbow_keywords, ewc, bic, gdumb]",
    )
    parser.add_argument(
        "--mem_manage",
        type=str,
        default='default',
        help="memory management [default, random, uncertainty, reservoir, prototype]",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsc",
        help="[gsc]",
    )
    parser.add_argument("--n_tasks", type=int, default=6, help="The number of tasks")
    parser.add_argument(
        "--n_cls_a_task", type=int, default=3, help="The number of class of each task"
    )
    parser.add_argument(
        "--n_init_cls",
        type=int,
        default=15,
        help="The number of classes of initial task",
    )
    parser.add_argument("--rnd_seed", type=int, default=1, help="Random seed number.")
    parser.add_argument(
        "--memory_size", type=int, default=500, help="Episodic memory size"
    )
    parser.add_argument(
        "--stream_env",
        type=str,
        default="online",
        choices=["offline", "online"],
        help="the restriction whether to keep streamed data or not",
    )

    # Dataset
    parser.add_argument(
        "--log_path",
        type=str,
        default="results",
        help="The path logs are saved. Only for local-machine",
    )
    parser.add_argument(
        "--neptune", default=True, action='store_true',
        help="record the experiment into web neptune.ai"
    )
    # Model
    parser.add_argument(
        "--model_name", type=str, default="tcresnet8", help="[tcresnet8, tcresnet14]"
    )
    parser.add_argument("--pretrain", action="store_true", help="pretrain model or not")

    # Train
    parser.add_argument("--opt_name", type=str, default="adam", help="[adam, sgd]")
    parser.add_argument("--sched_name", type=str, default="cos", help="[cos, anneal]")
    parser.add_argument("--batchsize", type=int, default=128, help="batch size")
    parser.add_argument("--n_epoch", type=int, default=50, help="Epoch")

    parser.add_argument("--n_worker", type=int, default=10, help="The number of workers")

    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--init_model",
        action="store_true",
        help="Initilize model parameters for every iterations",
    )
    parser.add_argument(
        "--init_opt",
        action="store_true",
        help="Initilize optimizer states for every iterations",
    )
    parser.add_argument(
        "--topk", type=int, default=1, help="set k when we want to set top k accuracy"
    )
    parser.add_argument(
        "--joint_acc",
        type=float,
        default=0.940,
        help="Accuracy when training all the tasks at once",
    )
    # Transforms
    parser.add_argument(
        "--transforms",
        nargs="*",
        default=["specaugment mixup"],
        help="Additional train transforms [mixup, specaugment]",
    )

    # Benchmark
    parser.add_argument("--exp_name", type=str, default="disjoint", help="[disjoint, blurry]")

    # ICARL
    parser.add_argument(
        "--feature_size",
        type=int,
        default=256,
        help="Feature size when embedding a sample",
    )

    # BiC
    parser.add_argument(
        "--distilling",
        action="store_true",
        help="use distilling loss with classification",
    )

    # Regularization
    parser.add_argument(
        "--reg_coef",
        type=int,
        default=0.1,
        help="weighting for the regularization loss term",
    )

    # Uncertain
    parser.add_argument(
        "--uncert_metric",
        type=str,
        default="vr",
        choices=["vr", "vr1", "vr_randaug", "loss"],
        help="A type of uncertainty metric",
    )

    # Debug
    parser.add_argument("--debug", action="store_true", help="Turn on Debug mode")

    args = parser.parse_args()
    return args
