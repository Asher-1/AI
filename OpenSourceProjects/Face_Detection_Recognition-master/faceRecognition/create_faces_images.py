#!usr/bin/env python
# -*- coding:utf-8 -*-

"""
@author: Asher
@file: create_faces_images.py
@time: 2019/04/06
"""

import numpy as np
from utils import image_processing, file_processing
import face_recognition
import facenet
from scipy import misc
import copy
import os
from tqdm import tqdm


def create_face(image_list, image_size=160, margin=32):
    """
    生成人脸数据图库，保存在out_face_dir中，这些数据库将用于生成embedding数据库
    :param image_list:
    :param image_size:
    :param margin:
    :return:
    """
    print("align dataset......")
    tmp_image_list = copy.copy(image_list)
    face_detect = face_recognition.Facedetection()
    nrof_samples = len(image_list)
    faces_list = []
    names_list = []
    for i in tqdm(range(nrof_samples)):
        try:
            img = misc.imread(os.path.expanduser(tmp_image_list[i]), mode='RGB')
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(tmp_image_list[i], e)
            print(errorMessage)
        else:
            if img.ndim == 2:
                img = facenet.to_rgb(img)
            img = img[:, :, 0:3]
            img_size = np.asarray(img.shape)[0:2]
            # 获取 判断标识 bounding_box crop_image
            bounding_boxes, _ = face_detect.detect_face(img)
            if len(bounding_boxes) < 1:
                image_list.remove(tmp_image_list[i])
                print("can't detect face, remove ", tmp_image_list[i])
                continue
            names_list.append(os.path.basename(tmp_image_list[i]))
            det = np.squeeze(bounding_boxes[0, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            face_img = facenet.prewhiten(aligned)
            faces_list.append(face_img)
    if len(faces_list) > 0:
        return faces_list, names_list
    else:
        return None, None
