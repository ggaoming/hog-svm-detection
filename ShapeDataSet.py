# -*- coding:utf-8 -*
"""
@version: 0.0
@Author by Ggao
@Mail: ggao_liming@qq.com
@File: ShapeDataSet.py
@time: 2019-08-20 下午2:03
"""
import cv2
import numpy as np
import random
import math


class ShapeData(object):
    def __init__(self, image_shape=[500, 500], batch_size=32,
                 cls_names=[]):
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.cls_names = cls_names
        pass

    def random_color(self, color_list):
        tmp_color = np.array([random.randint(0, 255) for _ in range(3)])
        color_sum = [c[0]*1000000 + c[1] * 1000 + c[2] for c in color_list]
        try_index = 0
        while tmp_color[0]**1000000 + tmp_color[1] * 1000 + tmp_color[2] in color_sum:
            tmp_color = np.array([random.randint(0, 255) for _ in range(3)])
        color_list.append(tmp_color)
        return tmp_color

    def random_image(self):
        color_list = []
        bg_color = self.random_color(color_list)
        img_h, img_w = self.image_shape
        bg_image = np.ones(self.image_shape + [3], dtype=np.uint8) * bg_color
        bg_image = bg_image.astype(np.uint8)
        label_image = np.zeros(self.image_shape, dtype=np.int32)
        bnds = []
        labels = []
        for _ in range(random.randint(self.image_shape[0]//50, self.image_shape[0]//10)):
            shape_type = random.randint(1, 3)
            buffer = 20
            y = random.randint(buffer, img_h - buffer - 1)
            x = random.randint(buffer, img_w - buffer - 1)
            s = random.randint(buffer, img_h // 5)
            s = random.randint(10, 20)
            wh = random.randint(1, 3)
            tmp_color = self.random_color(color_list)
            if shape_type == 1:  # 矩形
                # print(x, y, s)
                # print(tmp_color)
                cv2.rectangle(bg_image, (x - s, y - s), (x + s, y + s), tmp_color.tolist(), -1)
                label_image[np.where(bg_image == tmp_color)[:2]] = 1
            elif shape_type == 2:  # 圆形
                cv2.circle(bg_image, (x, y), s, tmp_color.tolist(), -1)
                label_image[np.where(bg_image == tmp_color)[:2]] = 2
            else:  # 三角形
                points = np.array([[(x, y - s),
                                    (x - s / math.sin(math.radians(60)), y + s),
                                    (x + s / math.sin(math.radians(60)), y + s),
                                    ]], dtype=np.int32)
                cv2.fillPoly(bg_image, points, tmp_color.tolist())
                label_image[np.where(bg_image == tmp_color)[:2]] = 3
            bnds.append([x - s, y - s, x + s, y + s])
            labels.append(shape_type)
        new_labels, new_bnds, mask = [], [], []
        for i in range(1, len(color_list)):
            b = color_list[i]
            tmp = (np.sum(np.equal(bg_image, b), axis=-1) == 3).astype(np.int)
            idxs = np.where(np.equal(tmp, 1))
            if len(idxs[0]) == 0:
                continue
            y0, y1 = np.min(idxs[0]), np.max(idxs[0])
            x0, x1 = np.min(idxs[1]), np.max(idxs[1])
            w = x1 - x0
            h = y1 - y0
            if min(w, h) < 5:
                continue
            new_bnds.append([x0, y0, x1, y1])
            new_labels.append(labels[i-1])
            mask.append(tmp*new_labels[-1])
        bnds, labels = new_bnds, new_labels
        return bg_image, label_image, bnds, labels, _

    def next2(self):
        images = []
        labels = []
        for _ in range(self.batch_size):
            img, label_image, bnd, label, masks = self.random_image()
            images.append(img)
            labels.append(label_image)
        return images, labels

    def next(self):
        images = []
        bnds = []
        for _ in range(self.batch_size):
            img, label_image, bnd, label, masks = self.random_image()
            new_bnds = []
            for i in range(len(bnd)):
                new_bnds.append(bnd[i] + [label[i], ])
            images.append(img)
            bnds.append(new_bnds)
        bnds = np.array(bnds)
        return images, bnds

    def next_mask(self):
        images, bnds, labels, masks = [], [], [], []
        for _ in range(self.batch_size):
            img, label_image, bnd, label, mask = self.random_image()
            new_bnds = []
            for i in range(len(bnd)):
                new_bnds.append(bnd[i] + [label[i], ])
            images.append(img)
            bnds.append(new_bnds)
            masks.append(mask)
        return images, bnds, masks

