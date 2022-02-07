"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/2/7 下午2:56
@Author  : Yang "Jan" Xiao 
@Description : test_tensorboard
"""
from torch.utils.tensorboard import SummaryWriter
import numpy as np
writer = SummaryWriter("/home/xiaoyang/Dev/kws-efficient-cl/tensorboard")

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)