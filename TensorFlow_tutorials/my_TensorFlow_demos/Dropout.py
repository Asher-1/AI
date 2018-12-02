#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@author: Asher
@time:2018/2/24 20:06
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import os

# 参数概要函数
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)  # 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)   # 直方图


# 添加层函数
def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):
    # 添加神经网络层
    '''
    inputs:              输入数据集
    in_size：            当前层的神经元个数
    out_size:            下一层神经元个数
    n_layer:             第几层
    activation_function: 激活函数
    '''
    layer_name = 'layer%s'% n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            # 权重依然使用常用的truncated_normal进行初始化
            Weights = tf.Variable(tf.truncated_normal([in_size, out_size],stddev=0.1), name='W')
            variable_summaries(Weights)
        with tf.name_scope('biases'):
            # 偏置则赋值为0.1
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1, name='b')
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
            tf.summary.histogram('histogram', Wx_plus_b)

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        return outputs


# 数据集路径
data_dir = "MNIST_data"
# 日志路径
log_dir = r'./logs/Dropout'

# 文件路径(不用绝对路径会出错)！！！！！！！
c_dir = os.getcwd()
DIR = r"/".join(c_dir.split("\\")) + "/logs/Dropout/Visual_Output/"


# 载入数据集
mnist = input_data.read_data_sets(data_dir, one_hot=True)
# 运行次数
max_steps = 31
# 图片数量
image_num = 3000
# 学习率
lr = tf.Variable(0.001, dtype=tf.float32)
# 载入图片
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')
# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size


""" 
为了在TensorBoard中展示节点名称，设计网络时会常使用tf.name_scope限制命名空间， 
在这个with下所有的节点都会自动命名为input/xxx这样的格式。 
定义输入x和y的placeholder，并将输入的一维数据变形为28×28的图片存储到另一个tensor， 
这样就可以使用tf.summary.image将图片数据汇总给TensorBoard展示了。 
"""
# 定义两个placeholder
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')
    # keep_prob用于传入当前层神经元保留的百分比
    keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)


# 创建一个简单的神经网络
# 隐藏层1
L1 = add_layer(x, 784, 500, 1, activation_function=tf.nn.tanh)
L1_drop = tf.nn.dropout(L1, keep_prob)
# 隐藏层2
L2 = add_layer(L1_drop, 500, 300, 2, activation_function=tf.nn.tanh)
L2_drop = tf.nn.dropout(L2, keep_prob)
# 输出层(即第三层)
prediction = add_layer(L2_drop, 300, 10, 3, activation_function=tf.nn.softmax)

# 二次代价函数（适用于输出神经元为线性函数）
# loss = tf.reduce_mean(tf.square(y-prediction))

""" 
这里使用tf.nn.softmax_cross_entropy_with_logits()对前面输出层的结果进行softmax处理并计算交叉熵损失cross_entropy。 
计算平均损失，并使用tf.summary.saclar进行统计汇总。 
"""
with tf.name_scope('loss'):
    # 交叉熵代价函数(适用于输出神经元为sigmoid函数和softmax函数)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)

""" 
使用Adma优化器对损失进行优化，同时统计预测正确的样本数并计算正确率accuray， 
再使用tf.summary.scalar对accuracy进行统计汇总。 
"""
with tf.name_scope('train'):
    # 使用梯度下降法
    train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # argmax返回一维张量中最大的值所在的位置
        # 结果存放在一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

    # 求准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
# with tf.Session() as sess:
    # tensorboard --logdir='logs/'
    sess.run(tf.global_variables_initializer())

    projector_writer = tf.summary.FileWriter(DIR + 'projector', sess.graph)
    # 产生metadata文件
    if tf.gfile.Exists(DIR + 'projector/metadata.tsv'):
        tf.gfile.DeleteRecursively(DIR + 'projector/metadata.tsv')
    with open(DIR + 'projector/metadata.tsv', 'w') as f:
        labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
        for i in range(image_num):
            f.write(str(labels[i]) + '\n')

    #  首先，使用tf.train.Saver()创建模型的保存器。
    saver = tf.train.Saver()
    # 配置projector，实现tensorBoard可视化
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = embedding.name
    embed.metadata_path = DIR + 'projector/metadata.tsv'
    embed.sprite.image_path = DIR + 'data/mnist_10k_sprite.png'
    embed.sprite.single_image_dim.extend([28, 28])
    projector.visualize_embeddings(projector_writer, config)

    # 保存信息
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test', sess.graph)

    # 合并所有的summary
    merged = tf.summary.merge_all()
    for epoch in range(max_steps):
        # 更新学习率
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        # 分批次进行训练
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

        # 使用tf.RunOption定义Tensorflow运行选项，其中设置trace_level为FULL——TRACE,
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # 并使用tf.RunMetadata()定义Tensorflow运行的元信息，这样可以记录训练时的运算时间和内存占用等方面的信息
        run_metadata = tf.RunMetadata()

        # 利用更新参数后的网络执行merged数据汇总操作
        train_summaries = sess.run(merged, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0},options=run_options, run_metadata=run_metadata)
        test_summaries = sess.run(merged, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        # 将汇总summary和训练元信息添加到train_writer和test_writer中.
        projector_writer.add_run_metadata(run_metadata, 'step%03d' % epoch)
        projector_writer.add_summary(train_summaries, epoch)
        train_writer.add_summary(train_summaries, epoch)
        test_writer.add_summary(test_summaries, epoch)

        # 将模型保存到log_dir + "/model.ckpt文件
        saver.save(sess, log_dir + "/model.ckpt", epoch)

        # 控制台输出结果
        learning_rate = sess.run(lr)
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        print("Iter " + str(epoch) + ",Testing Accuracy= " + str(test_acc) + ",  Training Accuracy= " + str(train_acc)
              + ",  learning_rate= " + str(learning_rate))

    # 将模型保存到DIR + projector/a_model.ckpt文件
    saver.save(sess, DIR + 'projector/a_model.ckpt', global_step=max_steps)

    # 所有训练全部结束后，关闭train_writer和test_writer。
    projector_writer.close()
    train_writer.close()
    test_writer.close()

# 训练模型可视化1： tensorboard --logdir='logs/'
# 训练模型可视化2： tensorboard --logdir=logs\Dropout\Visual_Output\projector
