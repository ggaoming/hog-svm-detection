# -*- coding:utf-8 -*
"""
@version: ??
@Author by Ggao
@Mail: ggao_liming@qq.com
@File: hog_svm.py
@time: 2019-08-20 下午2:05
"""
from ShapeDataSet import ShapeData
import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm
import pickle
import random
import tqdm
import time


def calculate_iou(a, b):
    """
    :param a: [N, 4]
    :param b: [M, 4]
    :return: [N, M]
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def py_nms(dets, thresh, mode="Union"):
    """
    greedily select boxes with high confidence
    keep boxes overlap <= thresh
    rule out overlap > thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap <= thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            uion = areas[i] + areas[order[1:]] - inter
            m = np.array(inter <= 0)
            ovr = inter / (uion + m)  # incase div by zero
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
        #keep
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class proposal_layer():
    def __init__(self):
        self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        pass

    def generate_train_info(self, image, bnd, gt_bnd):
        bnd = np.array(bnd)
        gt_bnd = np.array(gt_bnd)
        ious = calculate_iou(bnd, gt_bnd[..., :4])
        best_ious = np.max(ious, axis=-1)
        pos_mask = best_ious >= 0.7
        neg_mask = best_ious <= 0.5
        pos_bnd = bnd[pos_mask]
        neg_bnd = bnd[neg_mask]
        return pos_bnd, neg_bnd

    def process(self, image, gt_bnds=None):

        self.ss.setBaseImage(image)
        self.ss.switchToSelectiveSearchFast()
        # self.ss.switchToSelectiveSearchQuality()
        rects = self.ss.process()

        bnds = []
        for rect in rects:
            x, y, w, h = rect
            bnds.append([x, y, x+w, y+h])
        if gt_bnds is None:
            return bnds
        else:
            return self.generate_train_info(image, bnds, gt_bnds)

    def extract_features(self, image, bnds, debug=False):
        features = []
        sub_images = []
        hog_images = []
        if debug:
            dfd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                 cells_per_block=(1, 1), visualize=True, multichannel=True)
            sub_images.append(image)
            hog_images.append(hog_image)
        for bnd in bnds:
            x0, y0, x1, y1 = bnd
            sub_image = image[y0:y1, x0:x1]
            sub_image = cv2.resize(sub_image, (50, 50))

            dfd, hog_image = hog(sub_image*1.0, orientations=8, pixels_per_cell=(16, 16),
                                 cells_per_block=(1, 1), visualize=True, multichannel=True)
            features.append(dfd)
            hog_images.append(hog_image)
            sub_images.append(sub_image)
        return features, hog_images, sub_images


class svm_detector():
    def __init__(self):
        self.model = svm.SVC(probability=True)
        self.init=False
        self.proposal = proposal_layer()
        pass

    def train(self, train_x, train_y, eval_x, eval_y):
        self.model.fit(train_x, train_y)
        print(self.model.score(eval_x, eval_y))
        pickle.dump(self.model, open("svm.model", "wb+"))
        pass

    def predict(self, image):
        if self.init is False:
            self.model = pickle.load(open("svm.model", "rb"))
        proposal_bnds = self.proposal.process(image)
        features, _, _ = self.proposal.extract_features(image, proposal_bnds, debug=True)
        pred = self.model.predict_proba(features)
        print(pred)
        bnds = []
        for i in range(len(proposal_bnds)):
            bnds.append(proposal_bnds[i] + [pred[i][1]])
        bnds = np.array(bnds)
        keep = py_nms(bnds, 0.4)
        bnds = np.array(bnds)[keep]
        return bnds


def hog_demo():
    app = ShapeData(batch_size=1)
    for _ in range(100):
        images, bnds, masks = app.next_mask()
        image = images[0]
        dfd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                             cells_per_block=(1, 1), visualize=True, multichannel=True)
        cv2.imshow("src", image)
        cv2.imshow("demo", hog_image)
        cv2.waitKey()


def check_proposal():
    app = ShapeData(image_shape=[300, 300], batch_size=1)
    layer = proposal_layer()
    for _ in range(100):
        start_time = time.time()
        images, bnds, masks = app.next_mask()
        print("data generate {}".format(time.time()-start_time))
        image = images[0]
        bnds = bnds[0]

        cv2.imshow("src", image)

        start_time = time.time()
        pos_bnds, neg_bnds = layer.process(image, bnds)
        print(len(pos_bnds), len(neg_bnds))
        print("proposal generate {}".format(time.time()-start_time))

        start_time = time.time()
        pos_feat, hog_imgs, sub_images = layer.extract_features(image, pos_bnds, True)
        print("feature generate {}".format(time.time()-start_time))

        for bnd in pos_bnds:
            x0, y0, x1, y1 = bnd
            cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 1)
        for bnd in neg_bnds:
            x0, y0, x1, y1 = bnd
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 1)
        print(hog_imgs[0].shape)
        cv2.imshow("features", np.concatenate(hog_imgs[1:6], axis=1))
        cv2.imshow("subimg", np.concatenate(sub_images[1:6], axis=1))
        cv2.imshow("hog", hog_imgs[0])
        cv2.imshow("bnd", image)
        cv2.waitKey()

def train():
    app = ShapeData(batch_size=1)
    layer = proposal_layer()
    detector = svm_detector()
    x = []
    y = []
    start_time = time.time()
    for _ in tqdm.tqdm(range(50)):
        images, bnds, masks = app.next_mask()
        image = images[0]
        bnds = bnds[0]
        pos_bnds, neg_bnds = layer.process(image, bnds)
        pos_feat, _, _ = layer.extract_features(image, pos_bnds)
        neg_feat, _, _ = layer.extract_features(image, neg_bnds)
        num_neg = len(neg_feat)
        num_pos = len(pos_feat)
        if num_pos > num_neg > 0:
            pos_idx = list(range(len(pos_bnds)))
            random.shuffle(pos_idx)
            pos_feat_ = []
            for i in range(num_neg):
                pos_feat_.append(pos_feat[i])
            x += pos_feat_
            y += [1, ] * num_neg
            x += neg_feat
            y += [0, ] * num_neg
    print("load data done: {}s".format(time.time() - start_time))
    start_time1 = time.time()
    total_num = len(x)
    train_num = int(total_num * 0.8)
    train_x = x[:train_num]
    train_y = y[:train_num]
    test_x = x[train_num:]
    test_y = y[train_num:]
    detector.train(train_x, train_y, test_x, test_y)
    print("train info done: {}s".format(time.time() - start_time1))
    print("total time: {}s".format(time.time() - start_time))


def predict():
    app = ShapeData(batch_size=1)
    detector = svm_detector()
    for _ in tqdm.tqdm(range(200)):
        images, bnds, masks = app.next_mask()
        image = images[0]
        bnds = bnds[0]
        pre_bnds = detector.predict(image)
        for bnd in pre_bnds:
            x0, y0, x1, y1, score = bnd
            if score < 0.5:
                continue
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            cv2.rectangle(image, (x0, y0), (x1, y0-20), (255, 255, 255), -1)
            cv2.putText(image, "{:.2f}".format(score), (x0, y0), 1, 1, (255, 0, 0), 1)
            cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 1)
        cv2.imshow("demo", image)
        cv2.waitKey()


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    predict()
    pass

