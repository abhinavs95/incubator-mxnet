# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=wildcard-import
"""Gluon Estimator"""


import warnings

from .event_handler import LoggingHandler
from ... import *
from ... import gluon, autograd
from ...metric import EvalMetric, Loss
import copy

__all__ = ['Estimator']


class Estimator(object):
    """
    Estimator Class for easy model training
    TODO: update doc
    """

    def __init__(self, net,
                 trainers,
                 loss=None,
                 metrics=None,
                 ctx=None):

        self.net = net

        if isinstance(loss, gluon.loss.Loss):
            self.loss = [loss]
        else:
            self.loss = loss or []
        if not self.loss:
            warnings.warn("No loss specified, default SoftmaxCrossEntropyLoss() is used.")
            self.loss = [gluon.loss.SoftmaxCrossEntropyLoss()]

        if isinstance(metrics, EvalMetric):
            self.train_metrics = [metrics]
        else:
            self.train_metrics = metrics or []
        self.test_metrics = copy.deepcopy(self.train_metrics)

        # store training statistics
        self.train_stats = {}
        self.train_stats['epochs'] = []
        self.train_stats['learning_rate'] = []
        # time used for each epoch
        self.train_stats['step'] = ''
        for metric in self.train_metrics:
            # record a history of metrics over each epoch
            self.train_stats['train_' + metric.name] = []
            # only record the latest metric numbers after each batch
            self.train_stats['batch_' + metric.name] = 0.
        for metric in self.test_metrics:
            self.train_stats['test_' + metric.name] = []
        self.train_loss_metrics = []
        self.test_loss_metrics = []
        # using the metric wrapper for loss to record loss value
        for loss in self.loss:
            self.train_loss_metrics.append(Loss(loss.name))
            self.test_loss_metrics.append(Loss(loss.name))
            self.train_stats['train_' + loss.name] = []
            self.train_stats['test_' + loss.name] = []
            # only record the latest loss numbers after each batch
            self.train_stats['batch_' + loss.name] = 0.

        if isinstance(ctx, Context):
            self.ctx = [ctx]
        else:
            if isinstance(ctx, list) and isinstance(ctx[0], Context):
                self.ctx = ctx
        if not ctx:
            num_gpus = len(mx.test_utils.list_gpus())
            self.ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

        if isinstance(trainers, gluon.Trainer):
            self.trainers = [trainers]
        else:
            self.trainers = trainers or []
        if not self.trainers:
            raise ValueError("No trainer specified, trainer is a required argument.")

    def _batch_fn(self, batch):
        data = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=self.ctx, batch_axis=0)
        return data, label

    def _test(self, val_data):
        for metric in self.test_metrics + self.test_loss_metrics:
            metric.reset()

        for i, batch in enumerate(val_data):
            data, label = self._batch_fn(batch)
            pred = [self.net(x) for x in data]
            losses = []
            for loss in self.loss:
                losses.append([loss(y_hat, y) for y_hat, y in zip(pred, label)])

            # update metrics
            for metric in self.test_metrics:
                metric.update(label, pred)
            for loss, loss_metric, in zip(losses, self.test_loss_metrics):
                loss_metric.update(0, [l for l in loss])

        for metric in self.test_metrics + self.test_loss_metrics:
            self.train_stats['test_' + metric.name].append(metric.get()[1])


    def fit(self, train_data,
            val_data=None,
            epochs=1,
            batch_size=None,
            event_handlers=None):

        if not batch_size:
            batch_size = 32 * len(self.ctx)

        event_handlers = event_handlers or []
        if not event_handlers:
            event_handlers.append(LoggingHandler(self))

        do_validation = False
        if val_data:
            do_validation = True

        # training begin
        for handler in event_handlers:
            handler.train_begin()

        for epoch in range(epochs):
            # epoch begin
            self.train_stats["epochs"].append(epoch)
            self.train_stats["learning_rate"].append(self.trainers[0].learning_rate)

            for handler in event_handlers:
                handler.epoch_begin()

            for metric in self.train_metrics + self.train_loss_metrics:
                metric.reset()

            for i, batch in enumerate(train_data):
                data, label = self._batch_fn(batch)

                # batch begin
                for handler in event_handlers:
                    handler.batch_begin()

                with autograd.record():
                    pred = [self.net(x) for x in data]
                    losses = []
                    for loss in self.loss:
                        losses.append([loss(y_hat, y) for y_hat, y in zip(pred, label)])

                for loss in losses:
                    for l in loss:
                        l.backward()

                # update train metrics
                for metric in self.train_metrics:
                    metric.update(label, pred)
                    self.train_stats['batch_' + metric.name] = metric.get()[1]
                for loss, loss_metric, in zip(losses, self.train_loss_metrics):
                    loss_metric.update(0, [l for l in loss])
                    self.train_stats['batch_' + loss_metric.name] = loss_metric.get()[1]

                self.train_stats['step'] = str(batch_size * (i + 1)) + '/' + str(len(train_data._dataset))

                for trainer in self.trainers:
                    trainer.step(batch_size)

                # batch end
                for handler in event_handlers:
                    handler.batch_end()

            # do validation
            if do_validation:
                self._test(val_data)

            for metric in self.train_metrics + self.train_loss_metrics:
                self.train_stats['train_' + metric.name].append(metric.get()[1])
            # epoch end
            for handler in event_handlers:
                handler.epoch_end(do_validation)

        # train end
        for handler in event_handlers:
            handler.train_end()
