"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/2/14 下午2:24
@Author  : Yang "Jan" Xiao 
@Description : data_augmentation
"""
import random

import numpy as np
import torch


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def spec_augmentation(x,
                      num_time_mask=2,
                      num_freq_mask=2,
                      max_time=10,
                      max_freq=12):
    """perform spec augmentation
    Args:
        x: input feature, T * F 2D
        num_t_mask: number of time mask to apply
        num_f_mask: number of freq mask to apply
        max_t: max width of time mask
        max_f: max width of freq mask
    Returns:
        augmented feature
    """
    _, max_freq_channel, max_frames = x.size()

    # time mask
    for i in range(num_time_mask):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_time)
        end = min(max_frames, start + length)
        x[:, :, start:end] = 0

    # freq mask
    for i in range(num_freq_mask):
        start = random.randint(0, max_freq_channel - 1)
        length = random.randint(1, max_freq)
        end = min(max_freq_channel, start + length)
        x[:, start:end, :] = 0

    return x
