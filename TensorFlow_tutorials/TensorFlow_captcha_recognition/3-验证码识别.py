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
from datetime import datetime
import time
# import numpy as np

# 不同字符数量
CHAR_SET_LEN = 36
# 图片高度
IMAGE_HEIGHT = 60
# 图片宽度
IMAGE_WIDTH = 160
# 最大步数
Max_step = 10000
# 批次
BATCH_SIZE = 100
# tfrecord文件存放路径
TFRECORD_FILE = "D:/develop/workstations/GitHub/Datasets/DL/Images/captcha/train.tfrecords"

output_model_dir = "D:/develop/workstations/GitHub/Datasets/DL/trained_outputs/captcha_output/models"

# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224])
y0 = tf.placeholder(tf.float32, [None])
y1 = tf.placeholder(tf.float32, [None])
y2 = tf.placeholder(tf.float32, [None])
y3 = tf.placeholder(tf.float32, [None])

# 学习率(初始化：0.002)
lr = tf.Variable(0.001, dtype=tf.float32)

# 定义网络结构
train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2',
    num_classes=CHAR_SET_LEN,
    weight_decay=0.0005,
    is_training=True)


# Setup the directory
def prepare_file_system(dir_name):
    if os.path.exists(dir_name):
        print("模型已经存在，防止训练好的模型被意外删除：\n请用户手动删除模型文件夹: %s ！！！" % output_model_dir)
        exit()
    else:
        # Makes sure the folder exists on disk
        os.makedirs(dir_name)
    return


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

    return image, label0, label1, label2, label3


# 数据处理，训练网络部署
def depoy_net():
    global total_loss, optimizer, accuracy0, accuracy1, accuracy2, accuracy3
    # inputs: a tensor of size [batch_size, height, width, channels]
    X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
    # 数据输入网络得到输出值
    logits0, logits1, logits2, logits3, end_points = train_network_fn(X)
    # 把标签转成one_hot的形式
    one_hot_labels0 = tf.one_hot(indices=tf.cast(y0, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels1 = tf.one_hot(indices=tf.cast(y1, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels2 = tf.one_hot(indices=tf.cast(y2, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels3 = tf.one_hot(indices=tf.cast(y3, tf.int32), depth=CHAR_SET_LEN)
    # 计算loss
    loss0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits0, labels=one_hot_labels0))
    loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits1, labels=one_hot_labels1))
    loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits2, labels=one_hot_labels2))
    loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits3, labels=one_hot_labels3))
    # 计算总的loss
    total_loss = (loss0 + loss1 + loss2 + loss3) / 4.0
    # 优化total_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(total_loss)
    # 计算准确率
    correct_prediction0 = tf.equal(tf.argmax(one_hot_labels0, 1), tf.argmax(logits0, 1))
    accuracy0 = tf.reduce_mean(tf.cast(correct_prediction0, tf.float32))
    correct_prediction1 = tf.equal(tf.argmax(one_hot_labels1, 1), tf.argmax(logits1, 1))
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
    correct_prediction2 = tf.equal(tf.argmax(one_hot_labels2, 1), tf.argmax(logits2, 1))
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
    correct_prediction3 = tf.equal(tf.argmax(one_hot_labels3, 1), tf.argmax(logits3, 1))
    accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))


# 预处理数据
def prepare_data():
    global image_batch, label_batch0, label_batch1, label_batch2, label_batch3
    # 准备目标文件夹，檢查文件夾是否存在，存在则用户许手动删除才能继续训练，不存在則直接創建
    prepare_file_system(output_model_dir)
    # 获取图片数据和标签
    image, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)
    # 使用shuffle_batch可以随机打乱
    image_batch, label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
        [image, label0, label1, label2, label3], batch_size=BATCH_SIZE,
        capacity=50000, min_after_dequeue=10000, num_threads=1)


with tf.Session() as sess:
    # 计时开始
    start = time.clock()

    # 预处理数据
    prepare_data()

    # 数据处理，训练网络部署
    depoy_net()

    # 用于保存模型
    saver = tf.train.Saver(max_to_keep=1)
    # 初始化
    sess.run(tf.global_variables_initializer())

    # 创建一个协调器，管理线程
    coord = tf.train.Coordinator()
    # 启动QueueRunner, 此时文件名队列已经进队
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(Max_step+1):
        # 获取一个批次的数据和标签
        b_image, b_label0, b_label1, b_label2, b_label3 = sess.run(
            [image_batch, label_batch0, label_batch1, label_batch2, label_batch3])
        # 优化模型
        sess.run(optimizer, feed_dict={x: b_image, y0: b_label0, y1: b_label1, y2: b_label2, y3: b_label3})

        # 每迭代50次计算一次loss和准确率
        if i % 100 == 0:
            # 每迭代500次降低一次学习率
            if i % 600 == 0:
                sess.run(tf.assign(lr, lr*0.8))
            acc0, acc1, acc2, acc3, loss_ = sess.run([accuracy0, accuracy1, accuracy2, accuracy3, total_loss],
                                                     feed_dict={x: b_image,
                                                                y0: b_label0,
                                                                y1: b_label1,
                                                                y2: b_label2,
                                                                y3: b_label3})
            learning_rate = sess.run(lr)
            print("%s: Iter:%d  Loss:%.3f  Accuracy:%.2f,%.2f,%.2f,%.2f  Learning_rate:%.4f" % (datetime.now(),
            i, loss_, acc0, acc1, acc2, acc3, learning_rate))
            # if i == Max_step:
            if acc0 > 0.95 and acc1 > 0.95 and acc2 > 0.95 and acc3 > 0.95:
                model = saver.save(sess, output_model_dir + '/crack_captcha.model', global_step=i)
                print('\n最终训练模型保存到：', model)
                break

        # 保存模型
        # if acc0 > 0.90 and acc1 > 0.90 and acc2 > 0.90 and acc3 > 0.90:
        if i % 600 == 0:
            model = saver.save(sess, output_model_dir + '/crack_captcha.model', global_step=i)
            print('\n临时训练模型保存到：', model)

    # 训练结束，计时结束，并统计运行时长  27min
    end = time.clock()
    print('\n训练总时长：%f s --- %f min' % (end - start, (end - start) / 60.0))

    # 通知其他线程关闭
    coord.request_stop()
    # 其他所有线程关闭之后，这一函数才能返回
    coord.join(threads)