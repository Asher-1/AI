# -*- coding: UTF-8 -*-

"""
Policy Gradient 算法（REINFORCE）。做决策的部分，相当于机器人的大脑
"""

import numpy as np
import tensorflow as tf

try:
    xrange = xrange  # Python 2
except:
    xrange = range   # Python 3


# 策略梯度 类
class PolicyGradient:
    def __init__(self,
                 lr,      # 学习速率
                 s_size,  # state/observation 的特征数目
                 a_size,  # action 的数目
                 h_size,  # hidden layer（隐藏层）神经元数目
                 discount_factor=0.99  # 折扣因子
    ):
        self.gamma = discount_factor  # Reward 递减率

        # 神经网络的前向传播部分。大脑根据 state 来选 action
        self.state_in = tf.placeholder(shape=[None, s_size], dtype=tf.float32)

        # 第一层全连接层
        hidden = tf.layers.dense(self.state_in, h_size, activation=tf.nn.relu)

        # 第二层全连接层，用 Softmax 来算概率
        self.output = tf.layers.dense(hidden, a_size, activation=tf.nn.softmax)

        # 直接选择概率最大的那个 action
        self.chosen_action = tf.argmax(self.output, 1)

        # 下面主要是负责训练的一些过程
        # 我们给神经网络传递 reward 和 action，为了计算 loss
        # 再用 loss 来调节神经网络的参数
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        # 计算 loss（和平时说的 loss 不一样）有一个负号
        # 因为 TensorFlow 自带的梯度下降只能 minimize（最小化）loss
        # 而 Policy Gradient 里面是要让这个所谓的 loss 最大化
        # 因此需要反一下。对负的去让它最小化，就是让它正向最大化
        self.loss = -tf.reduce_mean(tf.log(self.outputs) * self.reward_holder)

        # 得到可被训练的变量
        train_vars = tf.trainable_variables()
        
        self.gradient_holders = []
        
        for index, var in enumerate(train_vars):
            placeholder = tf.placeholder(tf.float32, name=str(index) + '_holder')
            self.gradient_holders.append(placeholder)

        # 对 loss 以 train_vars 来计算梯度
        self.gradients = tf.gradients(self.loss, train_vars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # apply_gradients 是 minimize 方法的第二部分，应用梯度
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, train_vars))

    # 计算折扣后的 reward
    # 公式： E = r1 + r2 * gamma + r3 * gamma * gamma + r4 * gamma * gamma * gamma ...
    def discount_rewards(self, rewards):
        discounted_r = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(xrange(0, rewards.size)):
            running_add = running_add * self.gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r
