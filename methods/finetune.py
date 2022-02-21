"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/1/25 下午11:06
@Author  : Yang "Jan" Xiao 
@Description : base_methods fine tune and native rehearsal method
"""
from methods.base import BaseMethod


class Finetune(BaseMethod):
    def __init__(self, criterion, device, n_classes, **kwargs):
        super().__init__(criterion, device, n_classes, **kwargs)
        if self.mode == "finetune":
            self.memory_size = 0
            if kwargs["stream_env"] == "online":
                raise Exception("Finetune method with online environment will have 0 samples for training, please set "
                                "offline environment")
        if self.mode == "native_rehearsal" and self.mem_manage == "default":
            self.mem_manage = "random"


