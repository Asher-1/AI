from datetime import datetime
import math
import time
import tensorflow as tf


# VGGNet是牛津大学计算机视觉组（Visual Geometry Group）和Google DeepMind公司的研究员一起研发的深度卷积神经网络
# VGG探索了卷积神经网络的深度与其性能之间的关系，通过反复堆叠3*3的小型卷积核和2*2的最大池化层，VGG成功构筑了16-19层深的卷积神经网络
# VGG取得了2014年比赛分类项目第二名和定位项目第一名。同时，VGG拓展性很强，迁移到其他图片数据上的泛化性非常好。VGG的结构简洁，
# 整个网络都是使用了同样大小的卷积核尺寸3*3和池化层2*2.
# VGG现在也还经常被用来提取图像特征，VGGNet训练后的模型参数在其官网上开源了，可用来在图像分类任务上进行再训练，相当于提供了非常好的
# 初始化权重

# VGG通过加深层次来提升性能，拥有5段卷积，每一段内有2-3个卷积层，同时每段尾部都会连接一个最大池化层来缩小图片尺寸
# 每段内的卷积核数量一样，越靠后段的卷积核数量越多，64-128-258-512-512
# 其中经常出现多个完全一样的3*3的卷积层堆叠在一起的情况，这其实是非常有用的设计，两个3*3的卷积层串联相当于一个5*5的卷积层
# 即一个像素会和周边5*5的像素产生关联，可以说感受野大小是5*5，而三个3*3的卷积层串联效果相当于一个7*7的卷积层
# 另外，3个串联的卷积层拥有比一个7*7的卷积层更少的参数量，只有后者的 3*3*3/（7*7）= 55%，更重要的是，3个串联的层拥有比1个层
# 更多的非线性变换，前者可以使用三次ReLU激活函数，而后者只有一次，使得CNN对特征的学习能力更强

# VGGNet-16

def conv_op(input_op, name, kernel_height, kernel_width, n_out, stride_height, stride_width, param_list):
    # 获取输入的通道数
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name=name) as scope:
        kernel = tf.get_variable(scope+'w',
                                 shape=[kernel_height, kernel_width, n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())





