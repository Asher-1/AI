#!usr/bin/env python
# -*- coding:utf-8 -*-

"""
@author: Asher
@file: train.py
@time: 2019/03/20
"""

import os
from os.path import join
import cv2
import caffe
import numpy as np
from skimage import io
import dlib

ROOT_PATH = "D:/develop/workstations/GitHub/Datasets/DL/Images/face_data/deep_landmark/"

# 网络模型配置文件，训练模型参数
deploy_path = ROOT_PATH + "prototxt"
model_path = ROOT_PATH + "model"
model_name = "_iter_100000.caffemodel"

# 人脸关键点定位测试输出结果
result_path = ROOT_PATH + 'test_output/result-folder/'
test_folder = ROOT_PATH + 'test_output/test-folder/'

CNN_TYPES = ['LE1', 'RE1', 'N1', 'LM1', 'RM1', 'LE2', 'RE2', 'N2', 'LM2', 'RM2']


class CNN(object):
    """
        Generalized CNN for simple run forward with given Model
    """

    def __init__(self, net, model):
        self.net = net
        self.model = model
        self.cnn = caffe.Net(net, model, caffe.TEST)  # failed if not exists

    def forward(self, data, layer='fc2'):
        # print data.shape
        fake = np.zeros((len(data), 1, 1, 1))
        self.cnn.set_input_arrays(data.astype(np.float32), fake.astype(np.float32))
        self.cnn.forward()
        result = self.cnn.blobs[layer].data[0]
        # 2N --> Nx(2)
        t = lambda x: np.asarray([np.asarray([x[2 * i], x[2 * i + 1]]) for i in range(len(x) / 2)])
        result = t(result)
        return result


class BBox(object):
    """
        Bounding Box of face
    """

    def __init__(self, bbox):
        self.left = int(bbox[0])
        self.right = int(bbox[1])
        self.top = int(bbox[2])
        self.bottom = int(bbox[3])
        self.x = bbox[0]
        self.y = bbox[2]
        self.w = bbox[1] - bbox[0]
        self.h = bbox[3] - bbox[2]

    def expand(self, scale=0.05):
        bbox = [self.left, self.right, self.top, self.bottom]
        bbox[0] -= int(self.w * scale)
        bbox[1] += int(self.w * scale)
        bbox[2] -= int(self.h * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)

    def project(self, point):
        x = (point[0] - self.x) / self.w
        y = (point[1] - self.y) / self.h
        return np.asarray([x, y])

    def reproject(self, point):
        x = self.x + self.w * point[0]
        y = self.y + self.h * point[1]
        return np.asarray([x, y])

    def reprojectLandmark(self, landmark):
        # print len(landmark)
        if not len(landmark) == 5:
            landmark = landmark[0]
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p

    def projectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p

    def subBBox(self, leftR, rightR, topR, bottomR):
        leftDelta = self.w * leftR
        rightDelta = self.w * rightR
        topDelta = self.h * topR
        bottomDelta = self.h * bottomR
        left = self.left + leftDelta
        right = self.left + rightDelta
        top = self.top + topDelta
        bottom = self.top + bottomDelta
        return BBox([left, right, top, bottom])

    def cropImage(self, img):
        """
            crop img with left,right,top,bottom
            **Make Sure is not out of box**
        """
        return img[self.top:self.bottom + 1, self.left:self.right + 1]


class Landmarker(object):
    """
        class Landmarker wrapper functions for predicting facial landmarks
    """

    def __init__(self):
        """
            Initialize Landmarker with files under VERSION
        """
        level1 = [(join(deploy_path, '1_F_deploy.prototxt'), join(model_path, '1_F', model_name))]
        level2 = [(join(deploy_path, '2_%s_deploy.prototxt' % name),
                   join(model_path, '2_%s' % name, model_name)) for name in CNN_TYPES]
        level3 = [(join(deploy_path, '3_%s_deploy.prototxt' % name),
                   join(model_path, '3_%s/' % name, model_name)) for name in CNN_TYPES]
        self.level1 = [CNN(p, m) for p, m in level1]
        self.level2 = [CNN(p, m) for p, m in level2]
        self.level3 = [CNN(p, m) for p, m in level3]

    def detectLandmark(self, image, bbox, mode='three'):
        """
            Predict landmarks for face with bbox in image
            fast mode will only apply level-1 and level-2
        """
        if not isinstance(bbox, BBox) or image is None:
            return None, False

        face = bbox.cropImage(image)
        face = cv2.resize(face, (39, 39))
        face = face.reshape((1, 1, 39, 39))
        face = self._processImage(face)

        # level-1, only F in implemented
        landmark = self.level1[0].forward(face)
        if mode == 'one':
            return landmark

        if mode == 'two':
            # level-2
            landmark = self._level(image, bbox, landmark, self.level2, [0.16, 0.18])
            return landmark

        if mode == 'three':
            # level-2
            landmark = self._level(image, bbox, landmark, self.level2, [0.16, 0.18])
            # level-3
            landmark = self._level(image, bbox, landmark, self.level3, [0.11, 0.12])

        return landmark

    def _level(self, img, bbox, landmark, cnns, padding):
        """
            LEVEL-?
        """
        for i in range(5):
            x, y = landmark[i]
            patch, patch_bbox = self._getPatch(img, bbox, (x, y), padding[0])
            patch = cv2.resize(patch, (15, 15)).reshape((1, 1, 15, 15))
            patch = self._processImage(patch)
            d1 = cnns[i].forward(patch)  # size = 1x2
            patch, patch_bbox = self._getPatch(img, bbox, (x, y), padding[1])
            patch = cv2.resize(patch, (15, 15)).reshape((1, 1, 15, 15))
            patch = self._processImage(patch)
            d2 = cnns[i + 5].forward(patch)

            d1 = bbox.project(patch_bbox.reproject(d1[0]))
            d2 = bbox.project(patch_bbox.reproject(d2[0]))
            landmark[i] = (d1 + d2) / 2
        return landmark

    def _getPatch(self, img, bbox, point, padding):
        """
            Get a patch iamge around the given point in bbox with padding
            point: relative_point in [0, 1] in bbox
        """

        point_x = bbox.x + point[0] * bbox.w
        point_y = bbox.y + point[1] * bbox.h
        patch_left = point_x - bbox.w * padding
        patch_right = point_x + bbox.w * padding
        patch_top = point_y - bbox.h * padding
        patch_bottom = point_y + bbox.h * padding
        patch = img[int(patch_top): int(patch_bottom + 1), int(patch_left): int(patch_right + 1)]
        patch_bbox = BBox([patch_left, patch_right, patch_top, patch_bottom])
        return patch, patch_bbox
        """
        point_x = bbox[0] + point[0] * bbox[2]
        point_y = bbox[1] + point[1] * bbox[3]
        patch_left = point_x - bbox[2] * padding
        patch_right = point_x + bbox[2] * padding
        patch_top = point_y - bbox[3] * padding
        patch_bottom = point_y + bbox[3] * padding
        patch = img[patch_top: patch_bottom+1, patch_left: patch_right+1]
        #patch_bbox = BBox([patch_left, patch_right, patch_top, patch_bottom])
        patch_bbox = [patch_left,patch_top,patch_right-patch_left,patch_bottom-patch_top]
        return patch, patch_bbox
        """

    def _processImage(self, imgs):
        """
            process images before feeding to CNNs
            imgs: N x 1 x W x H
        """
        imgs = imgs.astype(np.float32)
        for i, img in enumerate(imgs):
            m = img.mean()
            s = img.std()
            imgs[i] = (img - m) / s
        return imgs


def drawLandmark(img, landmark):
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), -1)
    return img


if __name__ == '__main__':

    TEST_MODE = "three"
    test_images = os.listdir(test_folder)
    for image in test_images:
        img = cv2.imread(test_folder + image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # image = io.imread(file_name)

        # 建立人脸探测器
        detector = dlib.get_frontal_face_detector()
        #
        detected_faces = detector(gray, 1)
        # 获取the boundingbox of face
        face_box = detected_faces[0]
        bbox = BBox([face_box.left(), face_box.right(), face_box.top(), face_box.bottom()])

        # bbox = BBox([84, 161, 92, 169])
        cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (0, 0, 255), 2)

        get_landmark = Landmarker()
        final_landmark = get_landmark.detectLandmark(gray, bbox, TEST_MODE)
        # print final_landmark

        final_landmark = bbox.reprojectLandmark(final_landmark)
        # print final_landmark
        # print final_landmark.shape
        img = drawLandmark(img, final_landmark)

        if TEST_MODE == 'one':
            cv2.imwrite(result_path + 'level1-' + image + '.jpg', img)
        elif TEST_MODE == 'two':
            cv2.imwrite(result_path + 'level1-' + image + 'level2-.jpg', img)
        elif TEST_MODE == 'three':
            cv2.imwrite(result_path + 'level1-' + image + 'level2-' + 'level3.jpg', img)
        else:
            print "TEST_MODE value is invalid..."

    print "test images output successfully!!!"
