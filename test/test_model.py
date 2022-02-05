"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/2/5 下午3:29
@Author  : Yang "Jan" Xiao 
@Description : test_model
"""
from utils.train_utils import select_model

model = select_model('tcresnet8', total_class_num=15)
