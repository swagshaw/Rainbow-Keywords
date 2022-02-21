"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/2/9 下午9:41
@Author  : Yang "Jan" Xiao 
@Description : joint
"""
import logging
from torch.utils.tensorboard import SummaryWriter
from methods.base import BaseMethod
from utils.train_utils import select_model, select_optimizer


logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class Joint(BaseMethod):
    def __init__(self, criterion, device, n_classes, **kwargs):

        super().__init__(criterion, device, n_classes, **kwargs)
        self.model = select_model(self.model_name,  n_classes)
        self.optimizer, self.scheduler = select_optimizer(
            kwargs["opt_name"], kwargs["lr"], self.model
        )
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.num_learning_class = n_classes


    def before_task(self, datalist, init_model, init_opt):
        pass

    def after_task(self, cur_iter):
        pass

    def update_memory(self, cur_iter, num_class=None):
        pass
