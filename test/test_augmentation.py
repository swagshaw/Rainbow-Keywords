"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/2/7 下午5:55
@Author  : Yang "Jan" Xiao 
@Description : test_augmentation
"""
import numpy as np
import torch
import torchaudio
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

transform_cands = [
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1),
    # TimeStretch(min_rate=0.8, max_rate=1.25, p=1),
    PitchShift(min_semitones=-4, max_semitones=4, p=1),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=1)
]
waveform = torchaudio.load('test.wav')[0]  # .numpy().astype(np.float32)
print(type(waveform))
for idx, tr in enumerate(transform_cands):
    _tr = Compose([tr])
    waveform = _tr(samples=waveform, sample_rate=16000)
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
print(waveform.dtype)
