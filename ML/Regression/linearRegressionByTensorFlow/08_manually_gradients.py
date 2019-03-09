import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


n_epochs = 2000
learning_rate = 0.01

housing = fetch_california_housing(data_home="D:/develop/workstations/GitHub/Datasets/ML/linearRegression/scikit_learn_data", download_if_missing=True)
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
# 可以使用TensorFlow或者Numpy或者sklearn的StandardScaler去进行归一化
# StandardScaler默认就做了方差归一化，和均值归一化，这两个归一化的目的都是为了更快的进行梯度下降
# 你如何构建你的训练集，你训练除了的模型，就具备什么样的功能！
scaler = StandardScaler().fit(housing_data_plus_bias)
scaled_housing_data_plus_bias = scaler.transform(housing_data_plus_bias)

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')

# random_uniform函数创建图里一个节点包含随机数值，给定它的形状和取值范围，就像numpy里面rand()函数
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
# 梯度的公式：(y_pred - y) * xj
gradients = 2/m * tf.matmul(tf.transpose(X), error)
# 赋值函数对于BGD来说就是 theta_new = theta - (learning_rate * gradients)
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()
    print(best_theta)

