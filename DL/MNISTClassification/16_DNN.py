import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tensorflow.contrib.layers import fully_connected

# 构建图阶段
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')


# 构建神经网络层，我们这里两个隐藏层，基本一样，除了输入inputs到每个神经元的连接不同
# 和神经元个数不同
# 输出层也非常相似，只是激活函数从ReLU变成了Softmax而已
def neuron_layer(X, n_neurons, name, activation=None):
    # 包含所有计算节点对于这一层，name_scope可写可不写
    with tf.name_scope(name):
        # 取输入矩阵的维度作为层的输入连接个数
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        # 这层里面的w可以看成是二维数组，每个神经元对于一组w参数
        # truncated normal distribution 比 regular normal distribution的值小
        # 不会出现任何大的权重值，确保慢慢的稳健的训练
        # 使用这种标准方差会让收敛快
        # w参数需要随机，不能为0，否则输出为0，最后调整都是一个幅度没意义
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        w = tf.Variable(init, name='weights')
        b = tf.Variable(tf.zeros([n_neurons]), name='biases')
        # 向量表达的使用比一条一条加和要高效
        z = tf.matmul(X, w) + b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z


# with tf.name_scope("dnn"):
#     hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
#     hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
#     # 进入到softmax之前的结果
#     logits = neuron_layer(hidden2, n_outputs, "outputs")


with tf.name_scope("dnn"):
    # tensorflow使用这个函数帮助我们使用合适的初始化w和b的策略，默认使用ReLU激活函数
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)

with tf.name_scope("loss"):
    # 定义交叉熵损失函数，并且求个样本平均
    # 函数等价于先使用softmax损失函数，再接着计算交叉熵，并且更有效率
    # 类似的softmax_cross_entropy_with_logits只会给one-hot编码，我们使用的会给0-9分类号
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    # 获取logits里面最大的那1位和y比较类别好是否相同，返回True或者False一组值
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# 计算图阶段
data_path = "D:/develop/workstations/GitHub/Datasets/ML/MNISTClassification/"
# mn.SOURCE_URL = "http://yann.lecun.com/exdb/mnist/"
mnist = input_data.read_data_sets(data_path + "MNIST_data_bak/", one_hot=False)
n_epochs = 100
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images,
                                            y: mnist.test.labels})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

    save_path = saver.save(sess, data_path + "ckpt/my_dnn_model_final.ckpt")

# 使用模型预测
with tf.Session as sess:
    saver.restore(sess, data_path + "ckpt/my_dnn_model_final.ckpt")
    X_new_scaled = [...]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)  # 查看最大的类别是哪个

