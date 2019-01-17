# 有时候需要把模型保持起来，有时候需要做一些checkpoint在训练中
# 以致于如果计算机宕机，我们还可以从之前checkpoint的位置去继续
# TensorFlow使得我们去保存和加载模型非常方便，仅需要去创建Saver节点在构建阶段最后
# 然后在计算阶段去调用save()方法

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

__author__ = 'Asher'

data_path = "D:/develop/workstations/GitHub/datasets/MachineLearning/MNISTClassification/"

# mn.SOURCE_URL = "http://yann.lecun.com/exdb/mnist/"
my_mnist = input_data.read_data_sets(data_path + "MNIST_data_bak/", one_hot=True)

# The MNIST data is split into three parts:
# 55,000 data points of training data (mnist.train)
# 10,000 points of test data (mnist.test), and
# 5,000 points of validation data (mnist.validation).

# Each image is 28 pixels by 28 pixels

# 输入的是一堆图片，None表示不限输入条数，784表示每张图片都是一个784个像素值的一维向量
# 所以输入的矩阵是None乘以784二维矩阵
x = tf.placeholder(dtype=tf.float32, shape=(None, 784))
# 初始化都是0，二维矩阵784乘以10个W值
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# 训练
# labels是每张图片都对应一个one-hot的10个值的向量
y_ = tf.placeholder(dtype=tf.float32, shape=(None, 10))
# 定义损失函数，交叉熵损失函数
# 对于多分类问题，通常使用交叉熵损失函数
# reduction_indices等价于axis，指明按照每行加，还是按照每列加
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),
                                              reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 初始化变量
init = tf.global_variables_initializer()
# 创建Saver()节点
saver = tf.train.Saver()

n_epoch = 1000

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epoch):
        if epoch % 100 == 0:
            save_path = saver.save(sess, data_path + "ckpt/my_model.ckpt")

        batch_xs, batch_ys = my_mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    best_theta = W.eval()
    save_path = saver.save(sess, data_path + "ckpt/my_model_final.ckpt")

