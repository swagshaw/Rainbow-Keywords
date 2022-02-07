"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/1/25 下午10:39
@Author  : Yang "Jan" Xiao 
@Description : speech data_loader
"""
import logging.config
import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F

logger = logging.getLogger()


class SpeechDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, dataset: str, is_training=True, transform=None):
        super(SpeechDataset, self).__init__()
        """
        Args:
            data_frame: pd.DataFrame
            filename: train_filename or valid_filename
            is_training: True or False
        """
        self.data_frame = data_frame
        self.dataset = dataset
        self.transform = transform
        self.sampling_rate = 16000
        self.sample_length = 16000
        self.is_training = is_training

    def __len__(self):
        return len(self.data_frame)

    def load_audio(self, speech_path):
        waveform, sample_rate = torchaudio.load(speech_path)
        if waveform.shape[1] < self.sample_length:
            # padding if the audio length is smaller than samping length.
            waveform = F.pad(waveform, [0, self.sample_length - waveform.shape[1]])

        if self.is_training == True:
            pad_length = int(waveform.shape[1] * 0.1)
            waveform = F.pad(waveform, [pad_length, pad_length])
            offset = torch.randint(0, waveform.shape[1] - self.sample_length + 1, size=(1,)).item()
            waveform = waveform.narrow(1, offset, self.sample_length)
        return waveform

    def __getitem__(self, idx):
        sample = dict()
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.data_frame.iloc[idx]["file_name"]
        label = self.data_frame.iloc[idx].get("label", -1)

        audio_path = os.path.join("/home/xiaoyang/Dev/kws-efficient-cl/dataset/data", file_name)
        waveform = self.load_audio(audio_path)
        if self.transform:
            # waveform = waveform.unsqueeze(1)
            # waveform.to("cuda")
            waveform = self.transform(samples=waveform, sample_rate=self.sampling_rate)
            waveform = torch.as_tensor(waveform, dtype=torch.float32)
            # waveform = waveform.squeeze(1)
        sample["waveform"] = waveform
        sample["label"] = label
        sample["file_name"] = file_name
        return sample

    def get_audio_class(self, y):
        return self.data_frame[self.data_frame["label"] == y]


def get_train_datalist(args, cur_iter: int) -> List:
    if args.mode == "joint":
        datalist = []
        for cur_iter_ in range(args.n_tasks):
            collection_name = get_train_collection_name(
                dataset=args.dataset,
                exp=args.exp_name,
                rnd=args.rnd_seed,
                n_cls=args.n_cls_a_task,
                iter=cur_iter_,
            )
            datalist += pd.read_json(
                os.path.join(args.data_root, f"{collection_name}.json")
            ).to_dict(orient="records")
            logger.info(f"[Train] Get datalist from {collection_name}.json")
    else:
        collection_name = get_train_collection_name(
            dataset=args.dataset,
            exp=args.exp_name,
            rnd=args.rnd_seed,
            n_cls=args.n_cls_a_task,
            iter=cur_iter,
        )

        datalist = pd.read_json(os.path.join(args.data_root, f"{collection_name}.json")
                                ).to_dict(orient="records")
        logger.info(f"[Train] Get datalist from {collection_name}.json")

    return datalist


def get_train_collection_name(dataset, exp, rnd, n_cls, iter):
    collection_name = "{dataset}_train_{exp}_rand{rnd}_cls{n_cls}_task{iter}".format(
        dataset=dataset, exp=exp, rnd=rnd, n_cls=n_cls, iter=iter
    )
    return collection_name


def get_test_datalist(args, exp_name: str, cur_iter: int) -> List:
    if exp_name is None:
        exp_name = args.exp_name

    if exp_name == "disjoint":
        # merge current and all previous tasks
        tasks = list(range(cur_iter + 1))
    else:
        raise NotImplementedError

    datalist = []
    for iter_ in tasks:
        collection_name = "{dataset}_test_rand{rnd}_cls{n_cls}_task{iter}".format(
            dataset=args.dataset, rnd=args.rnd_seed, n_cls=args.n_cls_a_task, iter=iter_
        )
        datalist += pd.read_json(
            os.path.join(args.data_root, f"{collection_name}.json")
        ).to_dict(orient="records")
        logger.info(f"[Test ] Get datalist from {collection_name}.json")

    return datalist


# from https://github.com/drimpossible/GDumb/blob/74a5e814afd89b19476cd0ea4287d09a7df3c7a8/src/utils.py#L102:5
def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    assert alpha > 0
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class TimeMask:
    def __init__(self, min_band_part=0.0, max_band_part=0.5):
        self.min_band_part = min_band_part
        self.max_band_part = max_band_part

    def __call__(self, samples, sample_rate):
        num_samples = samples.shape[-1]
        t = random.randint(
            int(num_samples * self.min_band_part),
            int(num_samples * self.max_band_part),
        )
        t0 = random.randint(
            0, num_samples - t
        )
        new_samples = samples.clone()
        mask = torch.zeros(t)
        new_samples[..., t0: t0 + t] *= mask
        return new_samples
