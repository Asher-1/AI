import tensorflow as tf

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x*x*y + y + 2

# 可以不分别对每个变量去进行初始化
# 并不立即初始化，在run运行的时候才初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()

print(result)


