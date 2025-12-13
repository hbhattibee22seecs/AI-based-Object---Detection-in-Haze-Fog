#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.data import RTTSDataset, TrainTransform, ValTransform, get_yolox_datadir
from yolox.evaluators import VOCEvaluator
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 5
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        rtts_root = os.path.join(get_yolox_datadir(), "RTTS")
        self.data_dir = rtts_root
        self.train_splits = ("train",)
        self.val_splits = ("val",)
        self.test_splits = ("test",)

        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 1

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        return RTTSDataset(
            data_dir=self.data_dir,
            image_sets=self.train_splits,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        legacy = kwargs.get("legacy", False)
        splits = self.test_splits if kwargs.get("testdev", False) else self.val_splits
        return RTTSDataset(
            data_dir=self.data_dir,
            image_sets=splits,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        return VOCEvaluator(
            dataloader=self.get_eval_loader(
                batch_size, is_distributed, testdev=testdev, legacy=legacy
            ),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
