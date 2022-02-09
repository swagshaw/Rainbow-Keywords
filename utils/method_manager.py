"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/1/25 下午11:02
@Author  : Yang "Jan" Xiao 
@Description : method_manager
"""
import logging

from methods.efficient_memory import EM
from methods.finetune import Finetune
from methods.icarl import ICaRL
from methods.regularization import EWC,RWalk
from methods.joint import Joint
logger = logging.getLogger()


def select_method(args, criterion, device, n_classes):
    kwargs = vars(args)
    if args.mode == "finetune":
        method = Finetune(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "native_rehearsal":
        method = Finetune(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "efficient_memory":
        method = EM(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs
        )
    elif args.mode == "icarl":
        method = ICaRL(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs
        )
    elif args.mode == "ewc":
        method = EWC(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )

    elif args.mode == "rwalk":
        method = RWalk(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    elif args.mode == "joint":
        method = Joint(
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            **kwargs,
        )
    else:
        raise NotImplementedError(
            "Choose the args.mode in [finetune,native_rehearsal, icarl, efficient_memory,ewc, rwalk]"
        )

    logger.info(f"CIL Scenario: {args.mode}")
    print(f"n_tasks: {args.n_tasks}")
    print(f"n_init_cls: {args.n_init_cls}")
    print(f"n_cls_a_task: {args.n_cls_a_task}")
    print(f"total cls: {args.n_tasks * args.n_cls_a_task}")

    return method
