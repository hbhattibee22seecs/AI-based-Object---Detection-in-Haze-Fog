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

    # Add derived keys for common naming/layout mismatches across YOLOX forks.
    #
    # Examples we support:
    # - `backbone.stage1.*` (older naming) -> `backbone.backbone.dark2.*`
    # - CSPLayer attrs: `main_conv/short_conv/final_conv/blocks` -> `conv1/conv2/conv3/m`
    # - Separate modules: `neck.*` -> `backbone.*` (YOLOPAFPN)
    # - MMDet style: `bbox_head.multi_level_*` -> `head.*`
    derived: dict[str, torch.Tensor] = {}
    stage_to_dark = {
        "stage1": "dark2",
        "stage2": "dark3",
        "stage3": "dark4",
        "stage4": "dark5",
    }
    csp_attr_map = {
        ".main_conv.": ".conv1.",
        ".short_conv.": ".conv2.",
        ".final_conv.": ".conv3.",
        ".blocks.": ".m.",
    }

    def _add_candidate(key: str, value) -> None:
        if key not in ckpt and key not in derived:
            derived[key] = value

    def _apply_csp_attr_map(key: str) -> str:
        for a, b in csp_attr_map.items():
            key = key.replace(a, b)
        return key

    def _yield_candidates(key: str, value):
        # Start with raw key and a few structural variants.
        cands = {key}

        # Some checkpoints omit the extra nesting: `backbone.*` -> `backbone.backbone.*`.
        if key.startswith("backbone.") and not key.startswith("backbone.backbone."):
            cands.add("backbone.backbone." + key[len("backbone.") :])

        # Some checkpoints store YOLOPAFPN parameters under `neck.*`.
        # Map those into our `backbone.*` module.
        if key.startswith("neck."):
            cands.add("backbone." + key[len("neck.") :])

        # Map common YOLOPAFPN list-based names to our attribute names.
        # (We still rely on shape checks later, but these are exact matches for YOLOX-S.)
        more = set()
        for k2 in cands:
            more.add(k2.replace("backbone.reduce_layers.0.", "backbone.lateral_conv0."))
            more.add(k2.replace("backbone.reduce_layers.1.", "backbone.reduce_conv1."))
            more.add(k2.replace("backbone.top_down_blocks.0.", "backbone.C3_p4."))
            more.add(k2.replace("backbone.top_down_blocks.1.", "backbone.C3_p3."))
            more.add(k2.replace("backbone.bottom_up_blocks.0.", "backbone.C3_n3."))
            more.add(k2.replace("backbone.bottom_up_blocks.1.", "backbone.C3_n4."))
            more.add(k2.replace("backbone.downsamples.0.", "backbone.bu_conv2."))
            more.add(k2.replace("backbone.downsamples.1.", "backbone.bu_conv1."))
        cands |= more

        # Stage naming -> dark* naming for CSPDarknet.
        stage_more = set()
        for k2 in cands:
            for stage, dark in stage_to_dark.items():
                stage_more.add(k2.replace(f"backbone.{stage}.", f"backbone.{dark}."))
                stage_more.add(
                    k2.replace(
                        f"backbone.backbone.{stage}.", f"backbone.backbone.{dark}."
                    )
                )
        cands |= stage_more

        # CSPLayer attribute naming differences.
        cands = {_apply_csp_attr_map(k2) for k2 in cands}

        # MMDet YOLOX head naming -> our YOLOX head naming.
        head_more = set()
        for k2 in cands:
            head_more.add(k2.replace("bbox_head.multi_level_cls_convs.", "head.cls_convs."))
            head_more.add(k2.replace("bbox_head.multi_level_reg_convs.", "head.reg_convs."))
            head_more.add(k2.replace("bbox_head.multi_level_conv_obj.", "head.obj_preds."))
            head_more.add(k2.replace("bbox_head.multi_level_conv_reg.", "head.reg_preds."))
            head_more.add(k2.replace("bbox_head.multi_level_conv_cls.", "head.cls_preds."))
        cands |= head_more

        # Apply CSPLayer mapping again after renames (cheap + safe).
        cands = {_apply_csp_attr_map(k2) for k2 in cands}

        return cands

    for k, v in ckpt.items():
        if not isinstance(k, str):
            continue
        for cand in _yield_candidates(k, v):
            _add_candidate(cand, v)

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
