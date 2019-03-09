import tensorflow as tf

x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')
f = x*x*y + y + 2

init = tf.global_variables_initializer()

# InteractiveSession和常规的Session不同在于，自动默认设置它自己为默认的session
# 即无需放在with块中了，但是这样需要自己来close session
sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()

# TensorFlow程序会典型的分为两部分，第一部分是创建计算图，叫做构建阶段，
# 这一阶段通常建立表示机器学习模型的的计算图，和需要去训练模型的计算图，
# 第二部分是执行阶段，执行阶段通常运行Loop循环重复训练步骤，每一步训练小批量数据，
# 逐渐的改进模型参数
