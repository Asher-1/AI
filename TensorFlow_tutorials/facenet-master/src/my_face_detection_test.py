#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Asher
@time:2018/3/24 17:09
"""

import argparse
import sys
import time

import cv2
sys.path.append("G:/develop/PycharmProjects/TensorFlow_tutorials/facenet-master/")
from contributed import face
import numpy as np

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def add_overlays(frame, faces):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)

    # cv2.putText(frame, (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
    #             thickness=2, lineType=2)


def main():
    # 获取脸部识别器
    face_recognition = face.Recognition()

    # 获取待验证图片
    image = cv2.imread('../data/sex_test_raw/Female/2008_007676.jpg')

    find_results = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if gray.ndim == 2:
        image = to_rgb(gray)

    faces = face_recognition.identify(image)
    add_overlays(image, faces)

    # show result
    # image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Show Result", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
