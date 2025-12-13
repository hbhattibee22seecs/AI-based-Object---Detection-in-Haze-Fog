#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import pickle
import xml.etree.ElementTree as ET
from typing import Iterable, List, Optional

import cv2
import numpy as np

from yolox.evaluators.voc_eval import voc_eval

from .datasets_wrapper import CacheDataset, cache_read_img
from .voc import AnnotationTransform

RTTS_CLASSES = ("person", "car", "bus", "bicycle", "motorbike")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


class RTTSDataset(CacheDataset):
    """Pascal VOC-style dataset loader tailored to the RTTS directory layout."""

    def __init__(
        self,
        data_dir: str,
        image_sets: Iterable[str] = ("train",),
        img_size=(640, 640),
        preproc=None,
        target_transform: Optional[AnnotationTransform] = None,
        cache: bool = False,
        cache_type: str = "ram",
    ):
        self.root = data_dir
        self.image_sets = tuple(image_sets)
        self.img_size = img_size
        self.preproc = preproc
        if target_transform is None:
            class_to_ind = dict(zip(RTTS_CLASSES, range(len(RTTS_CLASSES))))
            target_transform = AnnotationTransform(
                class_to_ind=class_to_ind, keep_difficult=False
            )
        self.target_transform = target_transform
        self.name = "RTTS"
        self._classes = RTTS_CLASSES
        self.cats = [{"id": idx, "name": val} for idx, val in enumerate(RTTS_CLASSES)]
        self.class_ids = list(range(len(RTTS_CLASSES)))
        self._ann_root = os.path.join(self.root, "Annotations")
        self._img_root = os.path.join(self.root, "JPEGImages")

        self.ids = self._collect_ids(self.image_sets)
        if not self.ids:
            raise RuntimeError(
                f"No RTTS images found for splits {self.image_sets} under {self.root}."
            )
        self.eval_split = self.image_sets[0]
        self.num_imgs = len(self.ids)
        self.annotations = [self.load_anno_from_ids(i) for i in range(self.num_imgs)]

        path_filename = [self._relative_image_path(i) for i in range(self.num_imgs)]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=self.root,
            cache_dir_name="cache_rtts",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type,
        )

    def _collect_ids(self, splits: Iterable[str]) -> List[str]:
        ids: List[str] = []
        for split in splits:
            split_file = os.path.join(self.root, "ImageSets", "Main", f"{split}.txt")
            if not os.path.exists(split_file):
                raise FileNotFoundError(
                    f"Missing RTTS split file: {split_file}. Expected per docs/rtts_training.md"
                )
            with open(split_file, "r", encoding="utf-8") as fh:
                for line in fh:
                    img_id = line.strip()
                    if img_id:
                        ids.append(img_id)
        return ids

    def __len__(self):
        return self.num_imgs

    # Annotation helpers -------------------------------------------------
    def _annotation_path(self, img_id: str) -> str:
        return os.path.join(self._ann_root, f"{img_id}.xml")

    def load_anno_from_ids(self, index: int):
        img_id = self.ids[index]
        target = ET.parse(self._annotation_path(img_id)).getroot()
        res, img_info = self.target_transform(target)
        height, width = img_info
        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        resized_info = (int(height * r), int(width * r))
        return res, img_info, resized_info

    def load_anno(self, index: int):
        return self.annotations[index][0]

    # Image helpers ------------------------------------------------------
    def _resolve_image_path(self, img_id: str) -> str:
        for ext in _IMAGE_EXTS:
            candidate = os.path.join(self._img_root, img_id + ext)
            if os.path.exists(candidate):
                return candidate
        fallback = os.path.join(self._img_root, img_id)
        if os.path.exists(fallback):
            return fallback
        raise FileNotFoundError(
            f"Unable to locate image for id {img_id} under {self._img_root}."
        )

    def _relative_image_path(self, index: int) -> str:
        abs_path = self._resolve_image_path(self.ids[index])
        rel_path = os.path.relpath(abs_path, self.root)
        return rel_path.replace(os.sep, "/")

    def load_image(self, index: int):
        img_path = self._resolve_image_path(self.ids[index])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        assert img is not None, f"file named {img_path} not found"
        return img

    def load_resized_img(self, index: int):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    @cache_read_img(use_cache=True)
    def read_img(self, index: int):
        return self.load_resized_img(index)

    def pull_item(self, index: int):
        target, img_info, _ = self.annotations[index]
        img = self.read_img(index)
        return img, target, img_info, index

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index: int):
        img, target, img_info, img_id = self.pull_item(index)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id

    # Evaluation ---------------------------------------------------------
    def evaluate_detections(self, all_boxes, output_dir=None):
        self._write_voc_results_file(all_boxes)
        iou_thresholds = np.linspace(0.5, 0.95, 10, endpoint=True)
        mAPs = []
        for iou in iou_thresholds:
            mAP = self._do_python_eval(output_dir, iou)
            mAPs.append(mAP)
        print("--------------------------------------------------------------")
        print("map_5095:", np.mean(mAPs))
        print("map_50:", mAPs[0])
        print("--------------------------------------------------------------")
        return np.mean(mAPs), mAPs[0]

    def _get_results_file_template(self):
        filename = f"comp4_det_{self.eval_split}_{{:s}}.txt"
        filedir = os.path.join(self.root, "results", self.eval_split, "Main")
        os.makedirs(filedir, exist_ok=True)
        return os.path.join(filedir, filename)

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(RTTS_CLASSES):
            if cls == "__background__":
                continue
            print(f"Writing {cls} RTTS results file")
            filename = self._get_results_file_template().format(cls)
            with open(filename, "wt", encoding="utf-8") as f:
                for im_ind, img_id in enumerate(self.ids):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                img_id,
                                dets[k, -1],
                                dets[k, 0] + 1,
                                dets[k, 1] + 1,
                                dets[k, 2] + 1,
                                dets[k, 3] + 1,
                            )
                        )

    def _do_python_eval(self, output_dir="output", iou=0.5):
        imagesetfile = os.path.join(
            self.root, "ImageSets", "Main", f"{self.eval_split}.txt"
        )
        cachedir = os.path.join(
            self.root, "annotations_cache", self.eval_split
        )
        os.makedirs(cachedir, exist_ok=True)
        aps = []
        print(f"Eval IoU : {iou:.2f}")
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for cls in RTTS_CLASSES:
            if cls == "__background__":
                continue
            filename = self._get_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename,
                os.path.join(self._ann_root, "{:s}.xml"),
                imagesetfile,
                cls,
                cachedir,
                ovthresh=iou,
                use_07_metric=True,
            )
            aps += [ap]
            if iou == 0.5:
                print(f"AP for {cls} = {ap:.4f}")
            if output_dir is not None:
                with open(os.path.join(output_dir, cls + "_pr.pkl"), "wb") as f:
                    pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        if iou == 0.5:
            print(f"Mean AP = {np.mean(aps):.4f}")
        return np.mean(aps)
