import tensorflow as tf

# 当去计算一个节点的时候，TensorFlow自动计算它依赖的一组节点，并且首先计算依赖的节点
w = tf.constant(3)
x = w + 2
y = x + 5
z = x * 3

with tf.Session() as sess:
    print(y.eval())
    # 这里为了去计算z，又重新计算了x和w，除了Variable值，tf是不会缓存其他比如contant等的值的
    # 一个Variable的生命周期是当它的initializer运行的时候开始，到会话session close的时候结束
    print(z.eval())

# 如果我们想要有效的计算y和z，并且又不重复计算w和x两次，我们必须要求TensorFlow计算y和z在一个图里
with tf.Session() as sess:
    y_val, z_val = sess.run([y, z])
    print(y_val)
    print(z_val)

