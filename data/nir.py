# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   nir.py
@Time    :   2023/2/1 20:01
@Desc    :
"""

import os

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.image_resize import imresize

from data import augment


class NIR(Dataset):
    def __init__(self, args, attr):
        self.args = args
        self.attr = attr
        if attr == "test":
            gt_root = os.path.join("data", args.dataset, "testing", "test_input_THER_LR_bicubic", "X8")  # no HR
            rgb_root = os.path.join("data", args.dataset, "testing", "test_VIS_HR")
        elif attr == "train":
            gt_root = os.path.join("data", args.dataset, "training", "train_output_gt_THER_HR")
            rgb_root = os.path.join("data", args.dataset, "training", "train_VIS_HR")
        else:
            gt_root = os.path.join("data", args.dataset, "validation", "valid_output_gt_THER_HR")
            rgb_root = os.path.join("data", args.dataset, "validation", "valid_VIS_HR")

        gt_paths = os.listdir(gt_root)
        rgb_paths = os.listdir(rgb_root)
        self.gt_imgs = [Image.open(os.path.join(gt_root, img)) for img in gt_paths]
        self.rgb_imgs = [Image.open(os.path.join(rgb_root, img)) for img in rgb_paths]
        self.img_names = gt_paths

    def __len__(self):
        return int(self.args.show_every * len(self.gt_imgs)) if self.attr == "train" else len(self.gt_imgs)

    def __getitem__(self, item):
        item = item % len(self.gt_imgs)

        gt_img, rgb_img = np.array(self.gt_imgs[item]), np.array(self.rgb_imgs[item])
        gt_img, rgb_img = np.expand_dims(gt_img, 0), np.transpose(rgb_img, (2, 0, 1))

        if self.attr == "train":
            gt_img, rgb_img = augment.get_patch(gt_img, rgb_img, patch_size=self.args.patch_size)
            gt_img, rgb_img = augment.random_rot(gt_img, rgb_img, hflip=True, rot=True)

        if self.attr != "test":
            lr_img = cv2.resize(gt_img.squeeze(), fx=1 / self.args.scale, fy=1 / self.args.scale, dsize=None)
        lr_img = imresize(lr_img.astype(float), scalar_scale=self.args.scale) / 255
        lr_img = np.expand_dims(lr_img, 0)

        gt_img, rgb_img = gt_img / 255, rgb_img / 255
        lr_img, gt_img, rgb_img = augment.np_to_tensor(lr_img, gt_img, rgb_img, input_data_range=1)
        if self.args.debug:
            print(self.img_names[item], "data_shape", gt_img.shape, rgb_img.shape, lr_img.shape)

        return {"img_gt": gt_img, "img_rgb": rgb_img, "lr_up": lr_img, "img_name": self.img_names[item]}
