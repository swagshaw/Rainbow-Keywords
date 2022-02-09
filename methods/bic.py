"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/2/9 下午10:10
@Author  : Yang "Jan" Xiao 
@Description : bic
"""

import logging
import os

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.data_loader import cutmix_data
from utils.train_utils import select_model, select_optimizer
from utils.data_loader import SpeechDataset

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class BiasCorrectionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=True)
        self.linear.weight.data.fill_(1.0)
        self.linear.bias.data.fill_(0.0)

    def forward(self, x):
        correction = self.linear(x.unsqueeze(dim=2))
        correction = correction.squeeze(dim=2)
        return correction


class BiasCorrection:
    def __init__(
            self, criterion, device, n_classes, **kwargs
    ):
        """
        self.valid_list: valid set which is used for training bias correction layer.
        self.memory_list: training set only including old classes. As already mentioned in the paper,
        memory list and valid list are exclusive.
        self.bias_layer_list - the list of bias correction layers. The index of the list means the task number.
        """
        self.num_learned_class = 0
        self.num_learning_class = kwargs["n_init_cls"]
        self.n_classes = n_classes
        self.learned_classes = []
        self.class_mean = [None] * n_classes
        self.exposed_classes = []
        self.seen = 0
        self.topk = kwargs["topk"]

        self.device = device
        self.criterion = criterion
        self.dataset = kwargs["dataset"]
        self.model_name = kwargs["model_name"]
        self.opt_name = kwargs["opt_name"]
        self.sched_name = kwargs["sched_name"]
        self.lr = kwargs["lr"]

        self.cutmix = "cutmix" in kwargs["transforms"]

        self.prev_streamed_list = []
        self.streamed_list = []
        self.test_list = []
        self.memory_list = []
        self.memory_size = kwargs["memory_size"]
        self.mem_manage = kwargs["mem_manage"]

        self.model = select_model(self.model_name, kwargs["n_init_cls"])
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        self.init_cls = kwargs["n_init_cls"]
        self.already_mem_update = False
        self.mode = kwargs["mode"]
        self.prev_model = select_model(
            self.model_name, kwargs["n_init_cls"]
        )
        self.bias_layer = None
        self.valid_list = []
        if kwargs["mem_manage"] == "default":
            self.mem_manage = "prototype"
        self.stream_env = kwargs["stream_env"]

        self.valid_size = round(self.memory_size * 0.1)
        self.memory_size = self.memory_size - self.valid_size
        self.n_tasks = kwargs["n_tasks"]
        self.bias_layer_list = []
        for _ in range(self.n_tasks):
            bias_layer = BiasCorrectionLayer().to(self.device)
            self.bias_layer_list.append(bias_layer)
        self.n_class_a_task = kwargs["n_cls_a_task"]
        self.distilling = kwargs["distilling"]
        if self.mode == "bic":
            self.distilling = True

        self.feature_extractor = self.model
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.sample_length = 16000

    def distillation_loss(self, old_logit, new_logit):
        # new_logit should have same dimension with old_logit.(dimension = n)
        assert new_logit.size(1) == old_logit.size(1)
        T = 2
        old_softmax = torch.softmax(old_logit / T, dim=1)
        new_log_softmax = torch.log_softmax(new_logit / T, dim=1)
        loss = -(old_softmax * new_log_softmax).sum(dim=1).mean()
        return loss

    def bias_forward(self, input):
        """
        forward bias correction layer.
        input: the output of classification model
        """
        n_new_cls = self.n_class_a_task
        n_split = round((input.size(1) - (
                self.init_cls - n_new_cls)) / n_new_cls)  # TODO: as the task 0 is 12 classes more than the other tasks
        out = []
        for i in range(n_split):
            sub_out = input[:,
                      i * n_new_cls + (self.init_cls - n_new_cls): (i + 1) * n_new_cls + (self.init_cls - n_new_cls)]
            # Only for new classes
            if i == n_split - 1:
                if i == 0:
                    sub_out = self.bias_layer_list[i](
                        input[:, 0: (i + 1) * n_new_cls + (self.init_cls - n_new_cls)]
                    )
                else:
                    sub_out = self.bias_layer_list[i](
                        input[:,
                        i * n_new_cls + (self.init_cls - n_new_cls): (i + 1) * n_new_cls + (self.init_cls - n_new_cls)]
                    )
            if i == 0:
                sub_out = input[:,
                          0: (i + 1) * n_new_cls + (
                                  self.init_cls - n_new_cls)]
            # logger.info(f"i is {i}, n_split is {n_split}, input_size() is {input.size()}")
            out.append(sub_out)
        ret = torch.cat(out, dim=1)
        assert ret.size(1) == input.size(1), f"final out: {ret.size()}, input size: {input.size()}"
        return ret

    def set_current_dataset(self, train_datalist, test_datalist):
        random.shuffle(train_datalist)
        self.prev_streamed_list = self.streamed_list
        self.streamed_list = train_datalist
        self.test_list = test_datalist

    def get_dataloader(self, batch_size, n_worker, train_list, test_list):
        # Loader
        train_loader = None
        test_loader = None
        if train_list is not None and len(train_list) > 0:
            train_dataset = SpeechDataset(
                pd.DataFrame(train_list),
                dataset=self.dataset,
                is_training=True
            )
            # drop last becasue of BatchNorm1D in IcarlNet
            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=n_worker,
                drop_last=True,
            )

        if test_list is not None:
            test_dataset = SpeechDataset(
                pd.DataFrame(test_list),
                dataset=self.dataset,
                is_training=False
            )
            test_loader = DataLoader(
                test_dataset, shuffle=False, batch_size=batch_size, num_workers=n_worker
            )

        return train_loader, test_loader

    def before_task(self, datalist, init_model=False, init_opt=True, cur_iter=None):
        logger.info("Apply before_task")
        incoming_classes = pd.DataFrame(datalist)["klass"].unique().tolist()
        self.exposed_classes = list(set(self.learned_classes + incoming_classes))
        self.num_learning_class = max(
            len(self.exposed_classes), self.num_learning_class
        )

        in_features = self.model.tc_resnet.channels[-1]
        out_features = self.model.tc_resnet.out_features
        # To care the case of decreasing head
        new_out_features = max(out_features, self.num_learning_class)
        if init_model:
            # init model parameters in every iteration
            logger.info("Reset model parameters")
            self.model = select_model(self.model_name, incoming_classes)
        else:
            self.model.tc_resnet.linear = nn.Linear(in_features, new_out_features)
        self.model = self.model.to(self.device)
        if init_opt:
            # reinitialize the optimizer and scheduler
            logger.info("Reset the optimizer and scheduler states")
            self.optimizer, self.scheduler = select_optimizer(
                self.opt_name, self.lr, self.model, self.sched_name
            )

        logger.info(f"Increasing the head of fc {out_features} -> {new_out_features}")

        self.already_mem_update = False
        self.bias_layer = self.bias_layer_list[cur_iter]
        # When task num > 0, the correction list needs to reduce.
        if cur_iter != 0:
            n_sample = self.valid_size // len(self.learned_classes)
            logger.info(f"[task {cur_iter}] n_sample: {n_sample} in valid_list")
            self.reduce_correction_examplers(num_sample=n_sample)

    def after_task(self, cur_iter):
        logger.info("Apply after_task")
        self.learned_classes = self.exposed_classes
        self.num_learned_class = self.num_learning_class
        self.update_memory(cur_iter)

    def update_memory(self, cur_iter, num_class=None):
        if num_class is None:
            num_class = self.num_learning_class

        if not self.already_mem_update:
            logger.info(f"Update memory over {num_class} classes by {self.mem_manage}")
            candidates = self.streamed_list + self.memory_list
            if len(candidates) <= self.memory_size:
                self.memory_list = candidates
                self.seen = len(candidates)
                logger.warning("Candidates < Memory size")
            else:
                if self.mem_manage == "prototype":
                    self.memory_list = self.mean_feature_sampling(
                        exemplars=self.memory_list,
                        samples=self.streamed_list,
                        num_class=num_class,
                    )
                else:
                    logger.error("Not implemented memory management")
                    raise NotImplementedError

            assert len(self.memory_list) <= self.memory_size
            logger.info("Memory statistic")
            memory_df = pd.DataFrame(self.memory_list)
            logger.info(f"\n{memory_df.klass.value_counts(sort=True)}")
            # memory update happens only once per task iterating.
            self.already_mem_update = True
        else:
            logger.warning(f"Already updated the memory during this iter ({cur_iter})")

    def reduce_correction_examplers(self, num_sample):
        correction_df = pd.DataFrame(self.valid_list)
        valid_list = []
        for label in correction_df["label"].unique():
            prev_correction_df = correction_df[correction_df.label == label].sample(
                n=num_sample
            )
            valid_list += prev_correction_df.to_dict(orient="records")
        self.valid_list = valid_list

    def construct_correction_examplers(self, num_sample):
        sample_df = pd.DataFrame(self.streamed_list)
        if len(sample_df) == 0:
            sample_df = pd.DataFrame(self.prev_streamed_list)

        if len(sample_df) == 0:
            logger.warning("No candidates for validset from current dataset")
            return False

        for label in sample_df["label"].unique():
            try:
                new_correction_df = sample_df[sample_df.label == label].sample(
                    n=num_sample
                )
                self.valid_list += new_correction_df.to_dict(orient="records")
            except ValueError:
                logger.warning(
                    f"Not available samples for bias_removal against label-{label}"
                )

    def train(self, cur_iter, n_epoch, batch_size, n_worker, n_passes=1):

        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Test samples: {len(self.test_list)}")

        random.shuffle(self.streamed_list)

        test_list = self.test_list
        train_list = self.streamed_list + self.memory_list
        logger.info(
            "[Task {}] self.training_list length: {}".format(cur_iter, len(train_list))
        )

        train_loader, test_loader = self.get_dataloader(
            batch_size, n_worker, train_list, test_list
        )

        # TRAIN
        best_acc = 0.0
        self.model = self.model.to(self.device)
        for epoch in range(n_epoch):
            # https://github.com/drimpossible/GDumb/blob/master/src/main.py
            # initialize for each task
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:  # Aand go!
                self.scheduler.step()

            train_loss = self._train(
                train_loader=train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                epoch=epoch,
                total_epochs=n_epoch,
                cur_iter=cur_iter,
                n_passes=n_passes,
            )
            eval_dict = self.evaluation(
                test_loader=test_loader, criterion=self.criterion
            )

            writer.add_scalar(f"task{cur_iter}/train/loss", train_loss, epoch)
            writer.add_scalar(
                f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch
            )
            writer.add_scalar(f"task{cur_iter}/test/acc", eval_dict["avg_acc"], epoch)

            logger.info(
                f"Task {cur_iter} | Epoch {epoch + 1}/{n_epoch} | train_loss {train_loss:.4f} | "
                f"test_acc {eval_dict['avg_acc']:.4f} | train_lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )

        if cur_iter == 0:
            n_sample = self.valid_size // len(self.exposed_classes)
            logger.info(
                f"[task {cur_iter}] num samples for bias correction : {n_sample}"
            )
            self.construct_correction_examplers(num_sample=n_sample)

        else:
            n_sample = self.valid_size // len(self.learned_classes)
            logger.info(
                f"[task {cur_iter}] num samples for bias correction : {n_sample}"
            )
            self.construct_correction_examplers(num_sample=n_sample)

            correction_df = pd.DataFrame(self.valid_list)
            correction_dataset = SpeechDataset(
                correction_df, dataset=self.dataset, is_training=False
            )
            correction_loader = DataLoader(
                correction_dataset, shuffle=True, batch_size=100, num_workers=2
            )
            self.bias_correction(
                bias_loader=correction_loader,
                test_loader=test_loader,
                criterion=self.criterion,
                n_epoch=n_epoch,
            )

        eval_dict = self.evaluation(test_loader=test_loader, criterion=self.criterion)
        if best_acc < eval_dict["avg_acc"]:
            best_acc = eval_dict["avg_acc"]
            self.prev_model = deepcopy(self.model)
        return best_acc, eval_dict

    def _train(
            self,
            train_loader,
            optimizer,
            criterion,
            epoch,
            total_epochs,
            cur_iter,
            n_passes=1,
    ):
        total_loss, distill_loss, classify_loss = 0.0, 0.0, 0.0
        self.model.train()

        for i, data in enumerate(train_loader):
            x = data["waveform"]
            xlabel = data["label"]
            x = x.to(self.device)
            xlabel = xlabel.to(self.device)
            if cur_iter != 0:
                self.prev_model.eval()
                with torch.no_grad():
                    logit_old = self.prev_model(x)
                    logit_old = self.bias_forward(logit_old)

            for pass_ in range(n_passes):
                optimizer.zero_grad()
                logit_new = self.model(x)
                logit_new = self.bias_forward(logit_new)
                loss_c = criterion(logit_new, xlabel)
                if cur_iter == 0:
                    loss_d = torch.tensor(0.0).to(self.device)
                else:
                    loss_d = self.distillation_loss(
                        logit_old, logit_new[:, : logit_old.size(1)]
                    )

                lam = len(self.learned_classes) / len(self.exposed_classes)
                if self.distilling:
                    loss = lam * loss_d + (1 - lam) * loss_c
                else:
                    loss = loss_c
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                distill_loss += loss_d.item()
                classify_loss += loss_c.item()

        n_batches = len(train_loader)
        logger.debug(
            "Stage 1 | Epoch {}/{} | loss: {:.4f} | distill_loss: {:.4f} | classify_loss:{:.4f}".format(
                epoch + 1,
                total_epochs,
                total_loss / n_batches,
                distill_loss / n_batches,
                classify_loss / n_batches,
            )
        )
        return total_loss / n_batches

    def bias_correction(self, bias_loader, test_loader, criterion, n_epoch):
        """
        Train the bias correction layer with bias_loader. (Linear Regression)

        :param bias_loader: data loader from valid list.

        """
        optimizer = torch.optim.Adam(params=self.bias_layer.parameters(), lr=0.001)

        self.model.eval()
        total_loss = None
        # Get the logit from the STAGE 1
        for iteration in range(n_epoch):
            self.bias_layer.train()
            total_loss = 0.0
            for i, data in enumerate(bias_loader):
                x = data["waveform"]
                xlabel = data["label"]
                x = x.to(self.device)
                xlabel = xlabel.to(self.device)
                out = self.model(x)
                logit = self.bias_forward(out)

                loss = criterion(logit, xlabel)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            logger.info(
                "[Stage 2] [{}/{}]\tloss: {:.4f}\talpha: {:.4f}\tbeta: {:.4f}".format(
                    iteration + 1,
                    n_epoch,
                    total_loss,
                    self.bias_layer.linear.weight.item(),
                    self.bias_layer.linear.bias.item(),
                )
            )
            if iteration % 50 == 0:
                self.evaluation(test_loader=test_loader, criterion=criterion)

        assert total_loss is not None

        self.print_bias_layer_parameters()
        return total_loss

    def evaluation_ext(self, test_list):
        # evaluation from out of class
        test_dataset = SpeechDataset(
            pd.DataFrame(test_list),
            dataset=self.dataset,
            is_training=False
        )
        test_loader = DataLoader(
            test_dataset, shuffle=False, batch_size=32, num_workers=2
        )
        eval_dict = self.evaluation(test_loader, self.criterion)

        return eval_dict

    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = (
            0.0,
            0.0,
            0.0,
        )
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        self.bias_layer.eval()

        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["waveform"]
                xlabel = data["label"]
                x = x.to(self.device)
                logit = self.model(x)
                logit = self.bias_forward(logit)
                logit = logit.detach().cpu()
                loss = criterion(logit, xlabel)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == xlabel.unsqueeze(1)).item()
                total_num_data += xlabel.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(xlabel, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += xlabel.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)

        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret

    def print_bias_layer_parameters(self):
        for i, layer in enumerate(self.bias_layer_list):
            logger.info(
                "[{}] alpha: {:.4f}, beta: {:.4f}".format(
                    i, layer.linear.weight.item(), layer.linear.bias.item()
                )
            )

    def _interpret_pred(self, y, pred):
        # xlabel is batch
        ret_num_data = torch.zeros(self.n_classes)
        ret_corrects = torch.zeros(self.n_classes)

        xlabel_cls, xlabel_cnt = y.unique(return_counts=True)
        for cls_idx, cnt in zip(xlabel_cls, xlabel_cnt):
            ret_num_data[cls_idx] = cnt

        correct_xlabel = y.masked_select(y == pred)
        correct_cls, correct_cnt = correct_xlabel.unique(return_counts=True)
        for cls_idx, cnt in zip(correct_cls, correct_cnt):
            ret_corrects[cls_idx] = cnt

        return ret_num_data, ret_corrects

    def mean_feature_sampling(self, exemplars, samples, num_class):
        """Prototype sampling

        Args:
            features ([Tensor]): [features corresponding to the samples]
            samples ([Datalist]): [datalist for a class]

        Returns:
            [type]: [Sampled datalist]
        """

        def _reduce_exemplar_sets(exemplars, mem_per_cls):
            if len(exemplars) == 0:
                return exemplars

            exemplar_df = pd.DataFrame(exemplars)
            ret = []
            for y in range(self.num_learned_class):
                cls_df = exemplar_df[exemplar_df["label"] == y]
                ret += cls_df.sample(n=min(mem_per_cls, len(cls_df))).to_dict(
                    orient="records"
                )

            num_dups = pd.DataFrame(ret).duplicated().sum()
            if num_dups > 0:
                logger.warning(f"Duplicated samples in memory: {num_dups}")

            return ret

        mem_per_cls = self.memory_size // num_class
        exemplars = _reduce_exemplar_sets(exemplars, mem_per_cls)
        old_exemplar_df = pd.DataFrame(exemplars)

        new_exemplar_set = []
        sample_df = pd.DataFrame(samples)
        for y in range(self.num_learning_class):
            cls_samples = []
            cls_exemplars = []
            if len(sample_df) != 0:
                cls_samples = sample_df[sample_df["label"] == y].to_dict(
                    orient="records"
                )
            if len(old_exemplar_df) != 0:
                cls_exemplars = old_exemplar_df[old_exemplar_df["label"] == y].to_dict(
                    orient="records"
                )

            if len(cls_exemplars) >= mem_per_cls:
                new_exemplar_set += cls_exemplars
                continue

            # Assign old exemplars to the samples
            cls_samples += cls_exemplars
            if len(cls_samples) <= mem_per_cls:
                new_exemplar_set += cls_samples
                continue

            features = []
            self.feature_extractor.eval()
            with torch.no_grad():
                for data in cls_samples:
                    audio_path = os.path.join("/home/xiaoyang/Dev/kws-efficient-cl/dataset/data", data["file_name"])
                    waveform, sample_rate = torchaudio.load(audio_path)
                    if waveform.shape[1] < self.sample_length:
                        # padding if the audio length is smaller than samping length.
                        waveform = F.pad(waveform, [0, self.sample_length - waveform.shape[1]])
                    waveform = waveform.to(self.device)
                    feature = (
                        self.feature_extractor(waveform.unsqueeze(0)).detach().cpu().numpy()
                    )
                    feature = feature / np.linalg.norm(feature, axis=1)  # Normalize
                    features.append(feature.squeeze())

            features = np.array(features)
            logger.debug(f"[Prototype] features: {features.shape}")

            # do not replace the existing class mean
            if self.class_mean[y] is None:
                cls_mean = np.mean(features, axis=0)
                cls_mean /= np.linalg.norm(cls_mean)
                self.class_mean[y] = cls_mean
            else:
                cls_mean = self.class_mean[y]
            assert cls_mean.ndim == 1

            phi = features
            mu = cls_mean
            # select exemplars from the scratch
            exemplar_features = []
            num_exemplars = min(mem_per_cls, len(cls_samples))
            for j in range(num_exemplars):
                S = np.sum(exemplar_features, axis=0)
                mu_p = 1.0 / (j + 1) * (phi + S)
                mu_p = mu_p / np.linalg.norm(mu_p, axis=1, keepdims=True)

                dist = np.sqrt(np.sum((mu - mu_p) ** 2, axis=1))
                i = np.argmin(dist)

                new_exemplar_set.append(cls_samples[i])
                exemplar_features.append(phi[i])

                # Avoid to sample the duplicated one.
                del cls_samples[i]
                phi = np.delete(phi, i, 0)

        return new_exemplar_set
