"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/2/4 下午5:00
@Author  : Yang "Jan" Xiao 
@Description : test_data_loader
"""
import json

from utils.data_loader import get_train_datalist, SpeechDataset
from configuration import config
import pandas as pd

args = config.base_parser()
train_list = get_train_datalist(args, cur_iter=0)

dataset = SpeechDataset(data_frame=train_list,dataset=)