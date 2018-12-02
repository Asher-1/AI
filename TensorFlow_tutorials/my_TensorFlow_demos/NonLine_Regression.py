#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@author: Asher
@time:2018/2/24 16:38
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成300个随机点
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 二次代价函数
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), reduction_indices=[1]))
# 使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())

    # 初始化并设置画布
    fig, ax = plt.subplots()
    ax.scatter(x_data, y_data)
    plt.ion()  # 开启连续绘制
    plt.show()

    for i in range(1000):
        # training
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
        if i % 50 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass

            # 获得预测值
            prediction_value = sess.run(prediction, feed_dict={x: x_data})
            # 画图
            lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
            plt.pause(0.4)
