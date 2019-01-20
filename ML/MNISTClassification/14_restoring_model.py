from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

__author__ = 'Asher'

data_path = "D:/develop/workstations/GitHub/Datasets/ML/MNISTClassification/"

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
# labels是每张图片都对应一个one-hot的10个值的向量
y_ = tf.placeholder(dtype=tf.float32, shape=(None, 10))

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, data_path + "ckpt/my_model_final.ckpt")

    # 评估
    # tf.argmax()是一个从tensor中寻找最大值的序号，tf.argmax就是求各个预测的数字中概率最大的那一个
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # 用tf.cast将之前correct_prediction输出的bool值转换为float32，再求平均
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 测试
    print(accuracy.eval({x: my_mnist.test.images, y_: my_mnist.test.labels}))



