# -*- coding: UTF-8 -*-

"""
训练 DCGAN
"""

import os
import glob
import numpy as np
from scipy import misc
import tensorflow as tf

from network import *


def train():
    # 确保包含所有图片的 images 文件夹在所有 Python 文件的同级目录下
    # 当然了，你也可以自定义文件夹名和路径
    if not os.path.exists("images"):
        raise Exception("包含所有图片的 images 文件夹不在此目录下，请添加")

    # 获取训练数据
    data = []
    for image in glob.glob("images/*"):
        image_data = misc.imread(image)  # imread 利用 PIL 来读取图片数据
        data.append(image_data)
    input_data = np.array(data)

    # 将数据标准化成 [-1, 1] 的取值, 这也是 Tanh 激活函数的输出范围
    input_data = (input_data.astype(np.float32) - 127.5) / 127.5

    # 构造 生成器 和 判别器
    g = generator_model()
    d = discriminator_model()

    # 构建 生成器 和 判别器 组成的网络模型
    d_on_g = generator_containing_discriminator(g, d)

    # 优化器用 Adam Optimizer
    g_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)
    d_optimizer = tf.keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1)

    # 配置 生成器 和 判别器
    g.compile(loss="binary_crossentropy", optimizer=g_optimizer)
    d_on_g.compile(loss="binary_crossentropy", optimizer=g_optimizer)
    d.trainable = True
    d.compile(loss="binary_crossentropy", optimizer=d_optimizer)

    # 开始训练
    for epoch in range(EPOCHS):
        for index in range(int(input_data.shape[0] / BATCH_SIZE)):
            input_batch = input_data[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]

            # 连续型均匀分布的随机数据（噪声）
            random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            # 生成器 生成的图片数据
            generated_images = g.predict(random_data, verbose=0)

            input_batch = np.concatenate((input_batch, generated_images))
            output_batch = [1] * BATCH_SIZE + [0] * BATCH_SIZE

            # 训练 判别器，让它具备识别不合格生成图片的能力
            d_loss = d.train_on_batch(input_batch, output_batch)

            # 当训练 生成器 时，让 判别器 不可被训练
            d.trainable = False

            # 重新生成随机数据。很关键
            random_data = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

            # 训练 生成器，并通过不可被训练的 判别器 去判别
            g_loss = d_on_g.train_on_batch(random_data, [1] * BATCH_SIZE)

            # 恢复 判别器 可被训练
            d.trainable = True

            # 打印损失
            print("Epoch {}, 第 {} 步, 生成器的损失: {:.3f}, 判别器的损失: {:.3f}".format(epoch, index, g_loss, d_loss))

        # 保存 生成器 和 判别器 的参数
        # 大家也可以设置保存时名称不同（比如后接 epoch 的数字），参数文件就不会被覆盖了
        if epoch % 10 == 9:
            g.save_weights("generator_weight", True)
            d.save_weights("discriminator_weight", True)


if __name__ == "__main__":
    train()
