from datetime import datetime
import math
import time
import tensorflow as tf


# ILSVRC(ImageNet Large Scale Visual Recognition Challenge)
# ImageNet项目于2007年由斯坦福大学华人教授李飞飞创办，目标是收集大量带有标注信息的图片数据供计算机视觉模型训练
# ImageNet拥有1500万张标注过的高清图片，总共拥有22000类，其中约有100万张标注了图片中主要物体的定位边框
# ImageNet项目最早的灵感来自于人类通过视觉学习世界的方式，如果假定儿童的眼睛是生物照相机，它们平均每200ms就拍照一次
# 眼球转动一次的平均时间200ms，那么3岁大的孩子就已经看过了上亿张真实时间的照片，使用亚马逊的土耳其机器人平台实现标注过程
# 来自世界上167个国家的5万名工作者帮忙一起筛选、标注

# 每年ILSVRC比赛数据集中大概有120万张图片，以及1000类的标注，是ImageNet全部数据的一个子集。比赛采用top-5和top-1分类错误率
# 作为模型性能的评测指标
# AlexNet比赛分类项目的2012年冠军，top5错误率16.4%，8层神经网络
# 神经网络模型AlexNet可以算是LeNet的一种更深更宽的版本！AlexNet中包含了几个比较新的技术点，首次在CNN中成功应用了ReLU、Dropout、
# LRN等Trick，AlexNet包含了6亿3000万个连接，6000多万个参数！和65万个神经元，拥有5个卷积层，3个全连接层，其中3个卷积层后面
# 连接了最大池化层，输出层是1000类的softmax层做分类，LRN应用在了第一卷积层和第二卷积层后面，ReLU激活函数应用在8层每一层后面

# 1，运用ReLU，解决Sigmoid在网络层次较深时的梯度弥散
# 2，训练Dropout，随机忽略一些神经元，避免过拟合
# 3，使用重叠的最大池化，此前CNN普遍平均池化，最大池化避免平均池化的模糊化效果
# 4，提出了LRN层，局部神经元活动创建竞争机制，响应比较大的值变得更大，抑制其他反馈小的神经元，增强泛化能力
# 5，使用CUDA加速深度卷积网络的训练
# 6，数据增强，随机地从256*256的原始图像中截取224*224大小的区域，以及水平翻转的镜像，相当于增加了【（256-224）^2】*2=2048
# 倍的数据量。没有数据增强，紧靠原始的数据量，参数众多的CNN会陷入过拟合中

batch_size = 32
num_batchs = 100


# 定义一个现实网络每一层结构的函数print_actications，展示每一个卷积层或池化层输入tensor的尺寸
# 这个函数接受一个tensor作为输入，并显示其名称 t.op.name 和tensor尺寸 t.get_shape.as_list()
def print_activations(t):
    print(t.op.name, " ", t.get_shape().as_list())


def inference(images):
    parameters = []

    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64],
                                                 dtype=tf.float32, stddev=1e-1), name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]

    # 这里LRN参数基本都是AlexNet论文中的推荐值，不过目前除了AlexNet，其他经典的卷积神经网络基本都放弃了LRN
    # 主要是效果不明显，使用也会使得前馈、反馈的速度整体下降1/3，可以自主选择是否使用LRN
    lrn1 = tf.nn.lrn(conv1, depth_radius=4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    print_activations(pool1)

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192],
                                                 dtype=tf.float32, stddev=1e-1, name='weights'))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192],
                                         dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv2)

    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    print_activations(pool2)

    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                                 dtype=tf.float32, stddev=1e-1, name='weights'))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[384],
                                         dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                                 dtype=tf.float32, stddev=1e-1, name='weights'))
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256],
                                         dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv4)

    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                                 dtype=tf.float32, stddev=1e-1, name='weights'))
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256],
                                         dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)

    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
    print_activations(pool5)

    return pool5, parameters


# 最后我们返回这个池化层的输出pool5，这样inference函数就完成了，它可以创建AlexNet的卷积部分，在正式使用AlexNet来训练或预测时
# 还需要添加3个全连接层，隐含节点数分别为4096，4096，1000，由于最后3个全连接层的计算量很小，就没有房子计算速度评测中
# 大家在使用AlexNet时需要自行加上这3个全连接层

# 评估AlexNet每轮计算时间的函数
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10  # 预热轮数
    total_duration = 0.0
    total_duration_squared = 0.0

    for i in range(num_batchs + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %
                      (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration

    mn = total_duration / num_batchs
    vr = total_duration_squared / num_batchs - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
          (datetime.now(), info_string, num_batchs, mn, sd))


def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size,
                                               3],
                                              dtype=tf.float32,
                                              stddev=1e-1))

        pool5, parameters = inference(images)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        time_tensorflow_run(sess, pool5, "Forward")

        # 模拟训练过程
        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, "Forward-backward")


run_benchmark()






