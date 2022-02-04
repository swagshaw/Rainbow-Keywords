"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/2/4 下午5:00
@Author  : Yang "Jan" Xiao 
@Description : test_data_loader
"""
import json

from torch.utils.data import DataLoader

from utils.data_loader import get_train_datalist, SpeechDataset
from configuration import config
import pandas as pd

args = config.base_parser()
train_list = get_train_datalist(args, cur_iter=0)

train_dataset = SpeechDataset(data_frame=pd.DataFrame(train_list), dataset='gsc')
# drop last becasue of BatchNorm1D in IcarlNet
train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=16,
    num_workers=8,
    drop_last=True,
)
