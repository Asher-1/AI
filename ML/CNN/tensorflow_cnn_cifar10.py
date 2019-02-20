import tensorflow as tf
import numpy as np
import math
import time
from CNNTools.image.cifar10 import cifar10
from CNNTools.image.cifar10 import cifar10_input


# 本节使用的数据集是CIFAR-10，这是一个经典的数据集，包含60000张32*32的彩色图像，其中训练集50000张，测试集10000张
# 一共标注为10类，每一类图片6000张。10类分别是 airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

# 我们载入一些常用库，比如NumPy和time，并载入TensorFlow Models中自动下载、读取CIFAR-10数据的类
max_steps = 3000
batch_size = 128
# 下载cifar10数据集的默认路径
data_dir = 'D:/develop/workstations/GitHub/Datasets/DL/Images/cifar/cifar10_data/cifar-10-batches-bin'


def variable_with_weight_losses(shape, stddev, wl):
    # 定义初始化weights的函数，和之前一样依然使用tf.truncated_normal截断的正太分布来初始化权值
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        # 给weight加一个L2的loss，相当于做了一个L2的正则化处理
        # 在机器学习中，不管是分类还是回归任务，都可能因为特征过多而导致过拟合，一般可以通过减少特征或者惩罚不重要特征的权重来缓解这个问题
        # 但是通常我们并不知道该惩罚哪些特征的权重，而正则化就是帮助我们惩罚特征权重的，即特征的权重也会成为模型的损失函数的一部分
        # 我们使用w1来控制L2 loss的大小
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        # 我们使用tf.add_to_collection把weight loss统一存到一个collection，这个collection名为"losses"，它会在后面计算神经网络
        # 总体loss时被用上
        tf.add_to_collection("losses", weight_loss)
    return var


# 下载cifar10类下载数据集，并解压，展开到其默认位置
cifar10.maybe_download_and_extract()
# 使用cifar10_input类中的distorted_inputs函数产生训练需要使用的数据，包括特征及其对应的label，这里是封装好的tensor，
# 每次执行都会生成一个batch_size的数量的样本。需要注意的是这里对数据进行了Data Augmentation数据增强
# 具体实现细节查看函数，其中数据增强操作包括随机水平翻转tf.image.random_flip_left_right()
# 随机剪切一块24*24大小的图片tf.random_crop，随机设置亮度和对比度，tf.image.random_brightness、tf.image.random_contrast
# 以及对数据进行标准化，白化 tf.image.per_image_standardization() 减去均值、除以方差，保证数据零均值，方差为1
images_train, labels_train = cifar10_input.distorted_inputs(
    data_dir=data_dir, batch_size=batch_size
)

# 生成测试数据，不过这里不需要进行太多处理，不需要对图片进行翻转或修改亮度、对比度，不过需要裁剪图片正中间的24*24大小的区块，
# 并进行数据标准化操作
images_test, labels_test = cifar10_input.inputs(eval_data=True, data_dir=data_dir, batch_size=batch_size)

# 因为batch_size在之后定义网络结构时被用到了，所以数据尺寸中的第一个值即样本条数需要被预先设定，而不能像以前那样设置为None
# 而数据尺寸中的图片尺寸为24*24即是剪裁后的大小，颜色通道数则设为3
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

# 初始设置第一个卷积层,64个卷积核，卷积核大小是5*5，3通道
weight1 = variable_with_weight_losses(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, filter=weight1, strides=[1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
# 使用尺寸3*3步长2*2的最大池化层处理数据，这里最大池化的尺寸和步长不一样，可以增加数据的丰富性
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
# 使用LRN对结果进行处理
# LRN最早见于Alex那篇用CNN参加ImageNet比赛的论文，Alex在论文中解释LRN层模仿了生物神经系统的"侧抑制"机制，
# 对局部神经元的活动创建竞争环境，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力
# Alex在ImageNet数据集上的实验表明，使用LRN后CNN在Top1的错误率可以降低1.4%，因此其在经典AlexNet中使用了LRN层
# LRN对ReLU这种没有上限边界的激活函数会比较有用，因为它会从附近的多个卷积核的响应中挑选比较大的反馈
# 但不适合Sigmoid这种有固定边界并且能抑制过大值得激活函数
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

# 创建第二个卷积层
# 上面64个卷积核，即输出64个通道，所以本层卷积核尺寸的第三个维度即输入的通道数也需要调整为64
weight2 = variable_with_weight_losses(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding='SAME')
# 还有这里的bias值全部初始化为0.1，而不是0.最后，调换了最大池化层和LRN层的顺序，先进行LRN层处理，再使用最大池化层
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 两个卷积层之后，是全连接层
# 先把第二个卷积层之后的输出结果flatten，使用tf.reshape函数将每个样本都变成一维向量，使用get_shape函数获取数据扁平化之后的长度
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
# 接着初始化权值，隐含节点384个，正太分布的标准差设为0.04，bias的值也初始化为0.1
# 注意这里我们希望这个全连接层不要过拟合，因此设了一个非零的weight loss值0.04，让这一层具有L2正则所约束。
weight3 = variable_with_weight_losses(shape=[dim, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
# 最后我们依然使用ReLU激活函数进行非线性化
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

# 接下来还是全连接层，只是隐含节点只有一半，其他一样
weight4 = variable_with_weight_losses(shape=[384, 192], stddev=0.04, wl=0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

# 最后一层，依然先创建一层weight，其正太分布标准差设为一个隐含层节点数的倒数，并且不用L2正则
# 这里没有用之前的softmax输出最后结果，这里把softmax操作放在了计算loss部分，其实我们不需要对inference的输出进行softmax
# 处理就可以获得最终分类结果（直接比较inference输出的各类的数值大小即可），计算softmax主要是为了计算loss，因此softmax操作整合到后面合理
weight5 = variable_with_weight_losses(shape=[192, 10], stddev=1/192.0, wl=0.0)
bias5 = tf.Variable(tf.constant(0.0, shape=[10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)


# 到这里就完成了整个网络inference的部分，梳理整个网络结构，设计性能良好的CNN是有一定规律可循的，但是想要针对某个问题设计最合适的
# 网络结构，是需要大量实际摸索的
# 完成模型inference的构建，接下来是计算CNN的loss，这里依然是用cross_entropy，这里我们把softmax的计算和cross_entropy的计算
# 合在了一起，即 tf.nn.sparse_softmax_cross_entropy_with_logits()
# 这里使用 tf.reduce_mean() 对 cross entropy计算均值，再使用 tf.add_to_collection()把cross entropy的loss添加到整体
# losses的collection中，最后，使用tf.add_n将整体losses的collection集合中的全部loss求和，得到最终的loss，其中包括
# cross entropy loss, 还有后两个全连接层中weight的L2 loss
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


loss = loss(logits=logits, labels=label_holder)
# 优化器依然选择Adam Optimizer, 学习速率0.001
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 使用 tf.nn.in_top_k()函数求输出结果中 top k的准确率，默认使用top 1，也就是输出分数最高的那一类的准确率
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 前面对图像进行数据增强的操作需要耗费大量CPU时间，因此distorted_inputs使用了16个独立的线程来加速任务，函数内部会产生线程池，
# 在需要使用时会通过TensorFlow queue进行调度
# 启动图片数据增强的线程队列，这里一共使用了16个线程来进行加速，如果不启动线程，那么后续inference以及训练的操作都是无法开始的
tf.train.start_queue_runners()

# 进行训练
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss],
                             feed_dict={image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time
    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)

        format_str = 'step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))


# 评测模型在测试集上的准确率
# 我们依然像训练时那样使用固定的batch_size，然后一个batch一个batch输入测试数据
num_examples = 10000
# 先计算一共要多少个batch才能将全部样本评测完
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict={image_holder: image_batch,
                                                  label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1

precision = true_count / total_sample_count
print('precision @ 1 = %.3f' % precision)

# 最终，在cifar-10数据集上，通过一个短时间小迭代的训练，可以达到大致73%的准确率，持续增加max_steps，可以期望准确率逐渐增加
# 如果max_steps比较大，则推荐使用学习速率衰减decay的SGD进行训练，这样训练过程中能达到的准确率峰值会比较高，大致有86%
# 其中L2正则以及LRN层的使用都对模型准确率有提升作用，它们都可以提升模型的泛化能力

# 数据增强Data Augmentation在我们的训练中作用很大，它可以给单幅图增加多个副本，提高图片的利用率，防止对某一张图片结构的学习过拟合
# 这刚好是利用了图片数据本身的性质，图片的冗余信息量比较大，因此可以制造不同的噪声并让图片依然可以被识别出来。如果神经网络可以克服这些
# 噪声并准确识别，那么他的泛化能力必然很好。数据增强大大增加了样本量，而数据量的大小恰恰是深度学习最看重的，深度学习可以在图像识别上领先
# 其他算法的一大因素就是它对海量数据的利用效率非常高。其他算法，可能在数据量大到一定程度时，准确率就不再上升了，而深度学习只要提供足够
# 多的样本，准确率基本持续提升，所以说它是最适合大数据的算法

















