#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import os
import shutil
from loguru import logger

import torch


def _normalize_ckpt_state_dict(ckpt: dict) -> dict:
    """Normalize common checkpoint key formats.

    Supports:
    - DataParallel/DistributedDataParallel prefixes: `module.`
    - Some training scripts prefix everything with `model.`
    - Older YOLOX layouts where backbone keys were `backbone.*` but the current
      model uses `backbone.backbone.*`.
    """
    if not isinstance(ckpt, dict):
        return ckpt

    keys = list(ckpt.keys())
    if keys and all(isinstance(k, str) and k.startswith("module.") for k in keys):
        ckpt = {k[len("module.") :]: v for k, v in ckpt.items()}

    keys = list(ckpt.keys())
    if keys and all(isinstance(k, str) and k.startswith("model.") for k in keys):
        ckpt = {k[len("model.") :]: v for k, v in ckpt.items()}

    # Add derived keys for backbone nesting mismatch.
    # If a checkpoint has `backbone.dark2...`, the current model may expect
    # `backbone.backbone.dark2...`.
    derived = {}
    stage_to_dark = {
        "stage1": "dark2",
        "stage2": "dark3",
        "stage3": "dark4",
        "stage4": "dark5",
    }
    for k, v in ckpt.items():
        if not isinstance(k, str):
            continue
        if k.startswith("backbone.") and not k.startswith("backbone.backbone."):
            kk = "backbone.backbone." + k[len("backbone.") :]
            if kk not in ckpt:
                derived[kk] = v

        # Some checkpoints use stage naming instead of dark* naming.
        # Try a best-effort remap (shape mismatches are filtered later).
        for stage, dark in stage_to_dark.items():
            prefix = f"backbone.{stage}."
            if k.startswith(prefix):
                rest = k[len(prefix) :]
                cand1 = f"backbone.{dark}." + rest
                cand2 = f"backbone.backbone.{dark}." + rest
                if cand1 not in ckpt and cand1 not in derived:
                    derived[cand1] = v
                if cand2 not in ckpt and cand2 not in derived:
                    derived[cand2] = v
    if derived:
        ckpt = {**ckpt, **derived}

    return ckpt


def load_ckpt(model, ckpt):
    ckpt = _normalize_ckpt_state_dict(ckpt)
    model_state_dict = model.state_dict()
    load_dict = {}
    missing_keys = 0
    shape_mismatch = 0
    max_warn = 20
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            missing_keys += 1
            if missing_keys <= max_warn:
                logger.warning(
                    "{} is not in the ckpt. Please double check and see if this is desired.".format(
                        key_model
                    )
                )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            shape_mismatch += 1
            if shape_mismatch <= max_warn:
                logger.warning(
                    "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                        key_model, v_ckpt.shape, key_model, v.shape
                    )
                )
            continue
        load_dict[key_model] = v_ckpt

    if missing_keys > max_warn:
        logger.warning(
            "... {} more keys are missing from the checkpoint.".format(missing_keys - max_warn)
        )
    if shape_mismatch > max_warn:
        logger.warning(
            "... {} more keys have shape mismatches.".format(shape_mismatch - max_warn)
        )

    model.load_state_dict(load_dict, strict=False)
    return model


def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth")
        shutil.copyfile(filename, best_filename)
