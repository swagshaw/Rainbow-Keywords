"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/1/25 下午11:06
@Author  : Yang "Jan" Xiao 
@Description : base_methods fine tune and native rehearsal method
"""

import logging

import random

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.data_loader import cutmix_data
from utils.train_utils import select_model, select_optimizer
from utils.data_loader import SpeechDataset

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


class Finetune:
    def __init__(
            self, criterion, device, n_classes, **kwargs
    ):
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

        self.already_mem_update = False
        self.mode = kwargs["mode"]
        if self.mode == "finetune":
            self.memory_size = 0
        if self.mode == "native_rehearsal" and self.mem_manage == "default":
            self.mem_manage = "random"

    def set_current_dataset(self, train_datalist, test_datalist):
        random.shuffle(train_datalist)
        self.prev_streamed_list = self.streamed_list
        self.streamed_list = train_datalist
        self.test_list = test_datalist

    def before_task(self, datalist, init_model=False, init_opt=True):
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
            self.model = select_model(self.model_name, new_out_features)
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
                if self.mem_manage == "random":
                    self.memory_list = self.rnd_sampling(candidates)
                elif self.mem_manage == "default":
                    self.memory_list = []
                else:
                    logger.error("Not implemented memory management")
                    raise NotImplementedError

            assert len(self.memory_list) <= self.memory_size
            logger.info("Memory statistic")
            memory_df = pd.DataFrame(self.memory_list)
            if len(self.memory_list) > 0:
                logger.info(f"\n{memory_df.klass.value_counts(sort=True)}")
            # memory update happens only once per task iteratin.
            self.already_mem_update = True
        else:
            logger.warning(f"Already updated the memory during this iter ({cur_iter})")

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

    def train(self, cur_iter, n_epoch, batch_size, n_worker, n_passes=1):

        train_list = self.streamed_list + self.memory_list
        random.shuffle(train_list)
        test_list = self.test_list
        train_loader, test_loader = self.get_dataloader(
            batch_size, n_worker, train_list, test_list
        )

        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(train_list)}")
        logger.info(f"Test samples: {len(test_list)}")

        # TRAIN
        best_acc = 0.0
        eval_dict = dict()
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

            train_loss, train_acc = self._train(
                train_loader=train_loader,
                optimizer=self.optimizer,
                criterion=self.criterion,
                epoch=epoch,
                total_epochs=n_epoch,
                n_passes=n_passes,
            )

            eval_dict = self.evaluation(
                test_loader=test_loader, criterion=self.criterion
            )

            writer.add_scalar(f"task{cur_iter}/train/loss", train_loss, epoch)
            writer.add_scalar(f"task{cur_iter}/train/acc", train_acc, epoch)
            writer.add_scalar(f"task{cur_iter}/test/loss", eval_dict["avg_loss"], epoch)
            writer.add_scalar(f"task{cur_iter}/test/acc", eval_dict["avg_acc"], epoch)
            writer.add_scalar(
                f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch
            )

            logger.info(
                f"Task {cur_iter} | Epoch {epoch + 1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )

            best_acc = max(best_acc, eval_dict["avg_acc"])

        return best_acc, eval_dict

    def _train(
            self, train_loader, optimizer, criterion, epoch, total_epochs, n_passes=1
    ):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        self.model.train()
        for i, data in enumerate(train_loader):
            for pass_ in range(n_passes):
                x = data["waveform"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                # do_cutmix = self.cutmix and np.random.rand(1) < 0.5
                do_cutmix = False
                if do_cutmix:
                    x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                    logit = self.model(x)
                    loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(
                        logit, labels_b
                    )
                else:
                    logit = self.model(x)
                    loss = criterion(logit, y)
                _, preds = logit.topk(self.topk, 1, True, True)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                correct += torch.sum(preds == y.unsqueeze(1)).item()
                num_data += y.size(0)

        n_batches = len(train_loader)

        return total_loss / n_batches, correct / num_data

    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["waveform"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit = self.model(x)

                loss = criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)

                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret

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

    def rnd_sampling(self, samples):
        random.shuffle(samples)
        return samples[: self.memory_size]
