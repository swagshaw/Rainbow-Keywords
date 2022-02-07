"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/2/7 下午5:55
@Author  : Yang "Jan" Xiao 
@Description : test_augmentation
"""
import os
import time

import numpy as np
import pandas as pd
import torch
import torchaudio
from audiomentations import Compose, AddGaussianNoise, PitchShift, Shift,  FrequencyMask, ClippingDistortion

from utils.data_loader import TimeMask
from utils.data_loader import SpeechDataset
# from torch_audiomentations import Compose, Gain, PitchShift, PolarityInversion, Shift
# transform_cands = [
#     # AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1),
#     # TimeStretch(min_rate=0.8, max_rate=1.25, p=1),
#     Gain(min_gain_in_db=-15.0, max_gain_in_db=5.0, p=1),
#     PitchShift(sample_rate=16000, min_transpose_semitones=-4.0, max_transpose_semitones=4.0, p=1),
#     Shift(min_shift=-0.5, max_shift=0.5, p=1),
#     PolarityInversion(p=1)
# ]
transform_cands = [
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1),
    PitchShift(min_semitones=-4, max_semitones=4, p=1),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=1),
    TimeMask(min_band_part=0,max_band_part=0.1),
    FrequencyMask(min_frequency_band=0,max_frequency_band=0.1,p=1),
    ClippingDistortion(min_percentile_threshold=0,max_percentile_threshold=10,p=1)

]
datalist = pd.read_json(
    "/home/xiaoyang/Dev/kws-efficient-cl/dataset/collection/gsc_test_rand1_cls3_task0.json"
).to_dict(orient="records")



# waveform = torchaudio.load('test.wav')[0]  # .numpy().astype(np.float32)
# waveform = waveform.unsqueeze(1)
# print(waveform.shape)
# print(type(waveform))
start = time.time()
for idx, tr in enumerate(transform_cands):
    _tr = Compose([tr])
    train_dataset = SpeechDataset(
        pd.DataFrame(datalist),
        dataset='gsc',
        is_training=False,
        transform=_tr
    )
    waveform = train_dataset.__getitem__(0)
    print(waveform["waveform"].dtype)
print(time.time()-start)