"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/2/9 下午11:40
@Author  : Yang "Jan" Xiao 
@Description : test_bic
"""
import torch

from methods.bic import BiasCorrection, BiasCorrectionLayer


def bias_forward(input):
    """
    forward bias correction layer.
    input: the output of classification model
    """
    n_new_cls = 3
    n_split = round((input.size(1) - 12) / n_new_cls)  # TODO: as the task 0 is 12 classes more than the other tasks
    out = []
    for i in range(n_split):
        sub_out = input[:, i * n_new_cls+12: (i + 1) * n_new_cls+12]
        # Only for new classes
        if i == n_split - 1:
            sub_out = input[:, i * n_new_cls+12: (i + 1) * n_new_cls+12]
        assert n_split < 6
        print(f"i is {i}, n_split is {n_split}, input_size() is {input.size()}")
        if i == 0:
            sub_out = input[:, 0:(i + 1) * n_new_cls + 12]
        out.append(sub_out)
    ret = torch.cat(out, dim=1)
    assert ret.size(1) == input.size(1), f"final out: {ret.size()}, input size: {input.size()}"
    return ret


input = torch.zeros((1, ))

print(bias_forward(input).size())
