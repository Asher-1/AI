import tensorflow as tf

# 让我们修改前面的代码去实现Mini-Batch梯度下降
# 为了去实现这个，我们需要一种方式去取代X和y在每一次迭代中，使用一小批数据
# 最简单的方式去做到这个是去使用placeholder节点
# 这些节点特点是它们不真正的计算，它们只是在执行过程中你要它们输出数据的时候去输出数据
# 它们会传输训练数据给TensorFlow在训练的时候
# 如果在运行过程中你不给它们指定数据，你会得到一个异常

# 需要做的是使用placeholder()并且给输出的tensor指定数据类型，也可以选择指定形状
# 如果你指定None对于某一个维度，它的意思代表任意大小
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5

with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
    B_val_2 = B.eval(feed_dict={A: [["4", 5, 6], [7, 8, 9]]})

print(B_val_1)
print(B_val_2)


