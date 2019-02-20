#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Asher
@time:2018/3/16 19:28
"""

import tensorflow as tf
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 再训练自己的数据的模型图和分类标签
output_graph = 'D:/develop/workstations/GitHub/Datasets/DL/trained_outputs/' \
               'flowers_classification_output/output_graph/final_graph.pb'
output_labels = 'D:/develop/workstations/GitHub/Datasets/DL/trained_outputs/' \
                'flowers_classification_output/output_graph/final_labels.txt'

# 图片数据文件夹。 在这个文件夹中每个子文件夹代表一个需要区分的类别，
# 每个子文件夹中存放了对应类别的图片
test_image_dir = 'test_images/'


# 获取分类名称函数
def id_to_string(node_id):
    if node_id not in uid_to_human:
        return ''
    return uid_to_human[node_id]


def get_name_score(node_id):
    # 获取分类名称
    human_string = id_to_string(node_id)
    # 获取该分类的置信度
    score = predictions[node_id]
    return human_string, score


def add_flags(frame, top_k_temp):
    sp = frame.shape
    text1 = 'rank:  ' + str(top_k_temp)
    cv2.putText(frame, text1, (int(sp[0]/2-60), int(sp[1]/4)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)

    result = map(get_name_score, top_k_temp)
    num = 0
    for human_string, score in result:
        text2 = '%s (score = %.5f)' % (human_string, score)
        num += 40
        cv2.putText(frame, text2, (int(sp[0]/2-100), int(sp[1]/3 + num)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2, lineType=2)


lines = tf.gfile.GFile(output_labels).readlines()
uid_to_human = {}
# 一行一行读取数据
for uid, line in enumerate(lines):
    # 去掉换行符
    line = line.strip('\n')
    uid_to_human[uid] = line

# 创建一个图来存放google训练好的模型
with tf.Graph().as_default() as graph:
    with tf.gfile.FastGFile(output_graph, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        softmax_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, name='',
                                                               return_elements=[
                                                                'final_result:0', 'import/DecodeJpeg/contents:0'])

# 开始测试
with tf.Session(graph=graph) as sess:
    # softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    # 遍历目录
    for root, dirs, files in os.walk(test_image_dir):
        for file in files:
            # 载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
            predictions = sess.run(softmax_tensor, feed_dict={jpeg_data_tensor: image_data})  # 图片格式是jpg格式
            predictions = np.squeeze(predictions)  # 把结果转为1维数据

            # 打印图片路径及名称
            image_path = os.path.join(root, file)
            # print(image_path)
            # 读取测试图片
            img = cv2.imread(image_path)
            # 设置窗口标题，并设置窗口大小可调整
            # cv2.namedWindow(file, cv2.WINDOW_NORMAL)
            # 统一尺寸
            image = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)

            # 排序
            top_k = predictions.argsort()[::-1]

            # 添加识别标记
            add_flags(image, top_k)

            # show result
            cv2.imshow(file, image)
            cv2.waitKey(0)
            cv2.destroyWindow(file)