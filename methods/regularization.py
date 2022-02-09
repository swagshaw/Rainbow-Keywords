"""
# !/usr/bin/env python
-*- coding: utf-8 -*-
@Time    : 2022/2/9 下午8:15
@Author  : Yang "Jan" Xiao 
@Description : regularization based methods -- ewc and rwalk
"""
import logging

import random

import numpy as np
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


class Regularization:
    """
    """

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
        self.feature_size = kwargs["feature_size"]

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

        # except for last layers.
        self.params = {
            n: p for n, p in list(self.model.named_parameters())[:-2] if p.requires_grad
        }  # For convenience
        self.regularization_terms = {}
        self.task_count = 0
        self.reg_coef = kwargs["reg_coef"]
        if kwargs["mem_manage"] == "default":
            self.mem_manage = "reservoir"
        self.online_reg = "online" in kwargs["stream_env"]

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
                elif self.mem_manage == "reservoir":
                    self.reservoir_sampling(self.streamed_list)
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

    def rnd_sampling(self, samples):
        random.shuffle(samples)
        return samples[: self.memory_size]

    def reservoir_sampling(self, samples):
        for sample in samples:
            if len(self.memory_list) < self.memory_size:
                self.memory_list += [sample]
            else:
                j = np.random.randint(0, self.seen)
                if j < self.memory_size:
                    self.memory_list[j] = sample
            self.seen += 1

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

    def calculate_importance(self, dataloader):
        # Use an identity importance so it is an L2 regularization.
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(1)  # Identity
        return importance

    def update_model(self, inputs, targets, optimizer):
        out = self.model(inputs)
        loss = self.regularization_loss(out, targets)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        return loss.detach(), out

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

    def train(self, cur_iter, n_epoch, batch_size, n_worker):
        # Loader
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
            # learning rate scheduling from
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
                epoch=epoch,
                total_epochs=n_epoch,
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

            if best_acc < eval_dict["avg_acc"]:
                best_acc = eval_dict["avg_acc"]

        # 2.Backup the weight of current task
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()

        # 3.Calculate the importance of weights for current task
        importance = self.calculate_importance(train_loader)

        # Save the weight and importance of weights of current task
        self.task_count += 1

        # Use a new slot to store the task-specific information
        if self.online_reg and len(self.regularization_terms) > 0:
            # Always use only one slot in self.regularization_terms
            self.regularization_terms[1] = {
                "importance": importance,
                "task_param": task_param,
            }
        else:
            # Use a new slot to store the task-specific information
            self.regularization_terms[self.task_count] = {
                "importance": importance,
                "task_param": task_param,
            }
        logger.debug(f"# of reg_terms: {len(self.regularization_terms)}")

        return best_acc, eval_dict

    def _train(self, train_loader, optimizer, epoch, total_epochs):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        self.model.train()
        for i, data in enumerate(train_loader):
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
                loss = lam * self.criterion(logit, labels_a) + (
                        1 - lam
                ) * self.criterion(logit, labels_b)
            else:
                logit = self.model(x)
                loss = self.criterion(logit, y)

            reg_loss = self.regularization_loss()

            loss += reg_loss
            loss.backward(retain_graph=True)
            optimizer.step()

            _, preds = logit.topk(self.topk, 1, True, True)
            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

        n_batches = len(train_loader)
        return total_loss / n_batches, correct / num_data

    def regularization_loss(
            self,
    ):
        reg_loss = 0
        if len(self.regularization_terms) > 0:
            # Calculate the reg_loss only when the regularization_terms exists
            for _, reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term["importance"]
                task_param = reg_term["task_param"]

                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()

                max_importance = 0
                max_param_change = 0
                for n, p in self.params.items():
                    max_importance = max(max_importance, importance[n].max())
                    max_param_change = max(
                        max_param_change, ((p - task_param[n]) ** 2).max()
                    )
                if reg_loss > 1000:
                    logger.warning(
                        f"max_importance:{max_importance}, max_param_change:{max_param_change}"
                    )
                reg_loss += task_reg_loss
            reg_loss = self.reg_coef * reg_loss

        return reg_loss


class EWC(Regularization):
    """
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        year={2017},
        url={https://arxiv.org/abs/1612.00796}
    }
    """

    def __init__(
            self, criterion, device, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, n_classes, **kwargs
        )

        self.n_fisher_sample = None
        self.empFI = False

    def calculate_importance(self, dataloader):
        # Update the diag fisher information
        # There are several ways to estimate the F matrix.
        # We keep the implementation as simple as possible while maintaining a similar performance to the literature.
        logger.info("Computing EWC")

        # Initialize the importance matrix
        importance = {}
        for n, p in self.params.items():
            importance[n] = p.clone().detach().fill_(0)  # zero initialized

        # Sample a subset (n_fisher_sample) of data to estimate the fisher information (batch_size=1)
        # Otherwise it uses mini-batches for the estimation. This speeds up the process a lot with similar performance.
        if self.n_fisher_sample is not None:
            n_sample = min(self.n_fisher_sample, len(dataloader.dataset))
            logger.info("Sample", self.n_fisher_sample, "for estimating the F matrix.")
            rand_ind = random.sample(list(range(len(dataloader.dataset))), n_sample)
            subdata = torch.utils.data.Subset(dataloader.dataset, rand_ind)
            dataloader = torch.utils.data.DataLoader(
                subdata, shuffle=True, num_workers=2, batch_size=1
            )

        self.model.eval()
        # Accumulate the square of gradients
        for data in dataloader:
            x = data["waveform"]
            y = data["label"]

            x = x.to(self.device)
            y = y.to(self.device)

            logit = self.model(x)

            pred = torch.argmax(logit, dim=-1)
            if self.empFI:  # Use groundtruth label (default is without this)
                pred = y

            loss = self.criterion(logit, pred)
            reg_loss = self.regularization_loss()
            loss += reg_loss

            self.model.zero_grad()
            loss.backward()

            for n, p in importance.items():
                # Some heads can have no grad if no loss applied on them.
                if self.params[n].grad is not None:
                    p += (self.params[n].grad ** 2) * len(x) / len(dataloader.dataset)

        return importance


class RWalk(Regularization):
    def __init__(
            self, criterion, device, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, n_classes, **kwargs
        )

        self.score = []
        self.fisher = []
        self.n_fisher_sample = None
        self.empFI = False
        self.alpha = 0.5
        self.epoch_score = {}
        self.epoch_fisher = {}
        for n, p in self.params.items():
            self.epoch_score[n] = (
                p.clone().detach().fill_(0).to(self.device)
            )  # zero initialized
            self.epoch_fisher[n] = (
                p.clone().detach().fill_(0).to(self.device)
            )  # zero initialized

    def update_fisher_and_score(
            self, new_params, old_params, new_grads, old_grads, epsilon=0.001
    ):
        for n, _ in self.params.items():
            if n in old_grads:
                new_p = new_params[n]
                old_p = old_params[n]
                new_grad = new_grads[n]
                old_grad = old_grads[n]
                self.epoch_score[n] += (new_grad - old_grad) / (
                        0.5 * self.epoch_fisher[n] * (new_p - old_p) ** 2 + epsilon
                )
                if self.epoch_score[n].max() > 1000:
                    logger.debug(
                        "Too large score {} / {}".format(
                            new_grad - old_grad,
                            0.5 * self.epoch_fisher[n] * (new_p - old_p) ** 2 + epsilon,
                        )
                    )
                if (self.epoch_fisher[n] == 0).all():  # First time
                    self.epoch_fisher[n] = new_grad ** 2
                else:
                    self.epoch_fisher[n] = (1 - self.alpha) * self.epoch_fisher[
                        n
                    ] + self.alpha * new_grad ** 2

    def _train(self, train_loader, optimizer, epoch, total_epochs):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        self.model.train()
        for i, data in enumerate(train_loader):
            x = data["waveform"]
            y = data["label"]
            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()

            old_params = {n: p.detach() for n, p in self.params.items()}
            old_grads = {
                n: p.grad.detach() for n, p in self.params.items() if p.grad is not None
            }

            # do_cutmix = self.cutmix and np.random.rand(1) < 0.5
            do_cutmix = False
            if do_cutmix:
                x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
                logit = self.model(x)
                loss = lam * self.criterion(logit, labels_a) + (
                        1 - lam
                ) * self.criterion(logit, labels_b)
            else:
                logit = self.model(x)
                loss = self.criterion(logit, y)

            reg_loss = self.regularization_loss()
            loss += reg_loss
            loss.backward(retain_graph=True)
            optimizer.step()

            new_params = {n: p.detach() for n, p in self.params.items()}
            new_grads = {
                n: p.grad.detach() for n, p in self.params.items() if p.grad is not None
            }

            self.update_fisher_and_score(new_params, old_params, new_grads, old_grads)

            _, preds = logit.topk(self.topk, 1, True, True)
            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)
        n_batches = len(train_loader)

        return total_loss / n_batches, correct / num_data

    def calculate_importance(self, dataloader):
        importance = {}
        self.fisher.append(self.epoch_fisher)
        if self.task_count == 0:
            self.score.append(self.epoch_score)
        else:
            score = {}
            for n, p in self.params.items():
                score[n] = 0.5 * self.score[-1][n] + 0.5 * self.epoch_score[n]
            self.score.append(score)

        for n, p in self.params.items():
            importance[n] = self.fisher[-1][n] + self.score[-1][n]
            self.epoch_score[n] = self.params[n].clone().detach().fill_(0)
        return importance
