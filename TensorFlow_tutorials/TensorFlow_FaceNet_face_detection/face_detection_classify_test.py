#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Asher
@time:2018/3/24 17:09
"""

import time
import os
import cv2
from src import face
import numpy as np

test_image_dir = 'data/sex_test_raw/'
classifier_model = os.path.dirname(__file__) + "/models/classifiers/sex_SVC_classifier.pkl"


def add_overlays(frame, faces, points):
    if faces is not None:
        num = -1
        for face in faces:
            num += 1
            face_bb = face.bounding_box.astype(int)
            # draw face
            cv2.rectangle(frame, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), (0, 255, 0), 2)
            # draw feature points
            cv2.circle(frame, (points[0][num], points[5][num]), 2, (0, 255, 0), -1)
            cv2.circle(frame, (points[1][num], points[6][num]), 2, (0, 255, 0), -1)
            cv2.circle(frame, (points[2][num], points[7][num]), 2, (0, 255, 0), -1)
            cv2.circle(frame, (points[3][num], points[8][num]), 2, (0, 255, 0), -1)
            cv2.circle(frame, (points[4][num], points[9][num]), 2, (0, 255, 0), -1)

            # 绘制所属分类类别
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def main():
    # 计时开始
    start = time.clock()
    # 获取脸部识别器
    face_recognition = face.Recognition(classifier_model)
    # 记录点1
    end1 = time.clock()
    print('获取脸部识别器face.Recognition()耗时：%f s --- %f min' % (end1 - start, (end1 - start) / 60.0))

    # 遍历目录
    for root, dirs, files in os.walk(test_image_dir):
        for file in files:
            # 获取待验证图片
            img_path = os.path.join(os.path.dirname(__file__), root, file)
            image = cv2.imread(img_path)
            # 预处理
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if gray.ndim == 2:
                img = to_rgb(gray)
                img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)

            # 统一尺寸
            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_CUBIC)

            # 人脸识别
            faces, points = face_recognition.identify(img)

            # 添加识别标记
            add_overlays(image, faces, points)

            # show result
            cv2.imshow(file, image)
            cv2.waitKey(0)

    # 计时结束，并统计运行时长
    end = time.clock()
    print('测试结束！！！')
    print('测试总时长：%f s --- %f min' % (end - start, (end - start) / 60.0))


if __name__ == '__main__':
    main()
