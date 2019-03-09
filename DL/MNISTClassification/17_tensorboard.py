import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

max_steps = 1000
learning_rate = 0.001
dropout = 0.9
data_path = "D:/develop/workstations/GitHub/Datasets/ML/MNISTClassification/"
data_dir = data_path + 'MNIST_data_bak'
log_dir = data_path + 'logs/mnist_with_summaries'

mnist = input_data.read_data_sets(data_dir, one_hot=True)
sess = tf.InteractiveSession()

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('input_reshape'):
    # 784维度变形为图片保持到节点
    # -1 代表进来的图片的数量、28，28是图片的高和宽，1是图片的颜色通道
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)


# 定义神经网络的初始化方法
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义Variable变量的数据汇总函数，我们计算出变量的mean、stddev、max、min
# 对这些标量数据使用tf.summary.scalar进行记录和汇总
# 使用tf.summary.histogram直接记录变量var的直方图数据
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# 设计一个MLP多层神经网络来训练数据
# 在每一层中都对模型数据进行汇总
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations


# 我们使用刚刚定义的函数创建一层神经网络，输入维度是图片的尺寸784=28*28
# 输出的维度是隐藏节点数500，再创建一个Dropout层，并使用tf.summary.scalar记录keep_prob
# 然后使用nn_layer定义神经网络输出层，其输入维度为上一层隐含节点数500，输出维度为类别数10
# 同时激活函数为全等映射identity，暂时不使用softmax
hidden1 = nn_layer(x, 784, 500, 'layer1')

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

# 使用tf.nn.softmax_cross_entropy_with_logits()对前面的输出层的结果进行Softmax
# 处理并计算交叉熵损失cross_entropy，计算平均的损失，使用tf.summary.scalar进行统计汇总
with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(diff)
tf.summary.scalar('cross_entropy', cross_entropy)

# 下面使用Adam优化器对损失进行优化，同时统计预测正确的样本数并计算正确率accuracy，汇总
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('accuracy', accuracy)

# 因为我们之前定义了太多的tf.summary汇总操作，逐一执行这些操作太麻烦，
# 使用tf.summary.merge_all()直接获取所有汇总操作，以便后面执行
merged = tf.summary.merge_all()
# 定义两个tf.summary.FileWriter文件记录器再不同的子目录，分别用来存储训练和测试的日志数据
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')
# 同时，将Session计算图sess.graph加入训练过程，这样再TensorBoard的GRAPHS窗口中就能展示
# 整个计算图的可视化效果，最后初始化全部变量
tf.global_variables_initializer().run()


# 定义feed_dict函数，如果是训练，需要设置dropout，如果是测试，keep_prob设置为1
def feed_dict(train):
    if train:
        xs, ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}


# 执行训练、测试、日志记录操作
# 创建模型的保存器
saver = tf.train.Saver(max_to_keep=3)
for i in range(max_steps):
    if i % 10 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
    else:
        if i % 100 == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, 1)
            saver.save(sess, log_dir + 'model.ckpt', i)
            print('Adding run metadata for', i)
        else:
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()

# tensorboard --logdir "D:/develop/workstations/GitHub/Datasets/ML/MNISTClassification/logs" --host=127.0.0.1