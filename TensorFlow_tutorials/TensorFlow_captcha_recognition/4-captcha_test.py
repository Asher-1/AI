#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Asher
@time:2018/3/14 23:08
"""

import os
import tensorflow as tf
from PIL import Image
from nets import nets_factory
import numpy as np
import matplotlib.pyplot as plt

# 不同字符数量
CHAR_SET_LEN = 36
# 图片高度
IMAGE_HEIGHT = 60
# 图片宽度
IMAGE_WIDTH = 160
# 批次
BATCH_SIZE = 1
# tfrecord文件存放路径
file_path = "D:/develop/workstations/GitHub/Datasets/DL/"
TFRECORD_FILE = file_path + "Images/captcha/test.tfrecords"
PRE_TRAIED_MODEL = file_path + "trained_outputs/captcha_output/models/crack_captcha.model-3800"

# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])


# 从tfrecord读出数据
def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label0': tf.FixedLenFeature([], tf.int64),
                                           'label1': tf.FixedLenFeature([], tf.int64),
                                           'label2': tf.FixedLenFeature([], tf.int64),
                                           'label3': tf.FixedLenFeature([], tf.int64),
                                       })
    # 获取图片数据
    image = tf.decode_raw(features['image'], tf.uint8)
    # 没有经过预处理的灰度图
    image_raw = tf.reshape(image, [224, 224])
    # tf.train.shuffle_batch必须确定shape
    image = tf.reshape(image, [224, 224])
    # 图片预处理
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    # 获取label
    label0 = tf.cast(features['label0'], tf.int32)
    label1 = tf.cast(features['label1'], tf.int32)
    label2 = tf.cast(features['label2'], tf.int32)
    label3 = tf.cast(features['label3'], tf.int32)

    return image, image_raw, label0, label1, label2, label3


# 数据转换
def data_convert(labels):
    # 处理数据
    num_labels = []
    for j in range(4):
        label = int(labels[j])
        if label > 9:
            # 将字母转换成数字即10-35转换成a-z
            num_labels.append(chr(label + 87))
        else:
            num_labels.append(label)
    return num_labels


# 获取图片数据和标签
image, image_raw, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)

# 使用shuffle_batch可以随机打乱
image_batch, image_raw_batch, label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
    [image, image_raw, label0, label1, label2, label3], batch_size=BATCH_SIZE,
    capacity=50000, min_after_dequeue=10000, num_threads=1)

# 定义网络结构
train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2',
    num_classes=CHAR_SET_LEN,
    weight_decay=0.0005,
    is_training=False)

with tf.Session() as sess:
    # inputs: a tensor of size [batch_size, height, width, channels]
    X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
    # 数据输入网络得到输出值
    logits0, logits1, logits2, logits3, end_points = train_network_fn(X)

    # 预测值
    predict0 = tf.reshape(logits0, [-1, CHAR_SET_LEN])
    predict0 = tf.argmax(predict0, 1)

    predict1 = tf.reshape(logits1, [-1, CHAR_SET_LEN])
    predict1 = tf.argmax(predict1, 1)

    predict2 = tf.reshape(logits2, [-1, CHAR_SET_LEN])
    predict2 = tf.argmax(predict2, 1)

    predict3 = tf.reshape(logits3, [-1, CHAR_SET_LEN])
    predict3 = tf.argmax(predict3, 1)

    # 初始化
    sess.run(tf.global_variables_initializer())
    # 载入训练好的模型
    saver = tf.train.Saver()
    saver.restore(sess, PRE_TRAIED_MODEL)

    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner, 此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(10):
        # 获取一个批次的数据和标签，这里获取一张图的数据和标签
        b_image, b_image_raw, b_label0, b_label1, b_label2, b_label3 = sess.run([image_batch,
                                                                                 image_raw_batch,
                                                                                 label_batch0,
                                                                                 label_batch1,
                                                                                 label_batch2,
                                                                                 label_batch3])

        # 数据转换
        label_list = [np.squeeze(b_label0), np.squeeze(b_label1), np.squeeze(b_label2), np.squeeze(b_label3)]
        label_list = data_convert(label_list)

        # 显示图片
        img = Image.fromarray(b_image_raw[0], 'L')
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        print('图%d 的标签和预测标签值如下：' % (i + 1))
        # 打印标签
        print('label:', label_list)
        # 预测
        label0, label1, label2, label3 = sess.run([predict0, predict1, predict2, predict3], feed_dict={x: b_image})

        # 数据转换
        label_list_predict = [np.squeeze(label0), np.squeeze(label1), np.squeeze(label2), np.squeeze(label3)]
        label_list_predict = data_convert(label_list_predict)
        # 打印预测值
        print('predict:', label_list_predict, '\n')

    # 通知其他线程关闭
    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    coord.join(threads)
