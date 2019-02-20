#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Asher
@time:2018/3/14 23:08
"""

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from datetime import datetime
import time

# 模型和样本路径的设置
PARENT_PATH = 'D:/develop/workstations/GitHub/Datasets/'
# 下载的谷歌训练好的Inception-v3模型文件目录
MODEL_DIR = PARENT_PATH + 'pretrained_models/inception_model_2015'

# 再训练自己的数据的模型图和分类标签
output_graph = PARENT_PATH + 'DL/trained_outputs/' \
               'flowers_classification_output/output_graph/final_graph.pb'
output_labels = PARENT_PATH + 'DL/trained_outputs/' \
                'flowers_classification_output/output_graph/final_labels.txt'

# intermediate_output_graphs_dir = 'output_graph/'
# 最后输出tensor的名称
final_tensor_name = 'final_result'
# intermediate_store_frequency = 1000

# 用于Tensorboard可视化的再训练模型文件夹名称
# 文件路径(不用绝对路径会出错)！！！！！！！
# c_dir = os.getcwd()
# summaries_dir = r"/".join(c_dir.split("\\")) + "/output_graph/retrain_logs/"

summaries_dir = PARENT_PATH + 'DL/trained_outputs/' \
                'flowers_classification_output/output_graph/retrain_logs/'

# 因为一个训练数据会被使用多次，所以可以将原始图像通过Inception-v3模型计算得
# 到的特征向量保存在文件中，免去重复的计算。下面的变量定义了这些文件的存放地址
CACHE_DIR = PARENT_PATH + 'DL/trained_outputs/' \
            'flowers_classification_output/bottleneck'

# 图片数据文件夹。 在这个文件夹中每个子文件夹代表一个需要区分的类别，
# 每个子文件夹中存放了对应类别的图片
INPUT_DATA = PARENT_PATH + 'DL/Images/flower_photos'

# 模型和样本路径的设置
# 验证的数据百分比
VALIDATION_PERCENTAGE = 10
# 测试的数据百分比
TEST_PERCENTAGE = 10

# 神经网络参数的设置
LEARNING_RATE = 0.01  # 学习率
STEPS = 4000  # 迭代次数
BATCH = 100  # batch大小
max_to_keep = 2


# 从数据文件夹中读取所有的图片列表并按训练、验证、测试数据分开
def create_image_lists(testing_percentage, validation_percentage):
    """
    :param testing_percentage:          指定了测试数据集的大小
    :param validation_percentage:       指定了验证数据集的大小
    :return:                            返回字典的字典(整理好的所有数据)
    """
    # 得到的所有图片都存在result这个字典里。这个字典的key为类别的名称，value也是一个字典，字典里存储了所有的图片名称
    result = {}
    # 获取当前目录下的所有子目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]

    is_root_dir = True
    for sub_dir in sub_dirs:
        # 得到的第一个目录是当前目录不需要考虑
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取当前目录下所有的有效图片文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        # 获取类别文件夹名称,即文件路径的最后一个文件夹名称
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            # glob.glob(file_glob)：获取与file_glob = ../datasets/flower_photos\\dir_name\\*.extension匹配的所有图片的路径
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        # 通过目录名获取类别的名称
        label_name = dir_name.lower()

        # 初始化当前类别的训练数据集、测试数据集和验证数据集
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)

            # 随机将数据分到训练数据集、测试数据集和验证数据集
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        # 将当前类别的数据放入结果字典
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    # 返回整理好的所有数据组成的字典的字典
    return result


# 通过类别名称、所属数据集和图片编号获取一张图片的地址
def get_image_path(image_lists, image_dir, label_name, index, category):
    """
    :param image_lists:     所有图片信息
    :param image_dir:       根目录。存放图片数据的根目录和存放图片特征向量的根目录地址不同
    :param label_name:      类别的名称
    :param index:           需要获取的图片的编号
    :param category:        category参数指定了需要获取的图片是在训练数据集、测试数据集还是验证数据集
    :return:                返回一张图片的地址
    """
    # 获取给定类别中所有图片的信息
    label_lists = image_lists[label_name]
    # 根据所属数据集名称获取集合中的全部图片信息
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    # 获取图片的文件名
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    # 最终的地址为数据根目录的地址 + 类别的文件夹 + 图片的名称
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


# 获取经过Inception-v3模型处理之后的特征向量的文件地址
def get_bottleneck_path(image_lists, label_name, index, category):
    """
    :param image_lists: 所有图片信息
    :param label_name:  类别名称
    :param index:       图片编号
    :param category:    所属数据集
    :return:
    """
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'


# 使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    # 这个过程实际上就是将当前图片作为输入计算瓶颈张量的值。这个瓶颈张量的值就是这张图片的新的特征向量
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # 经过卷积神经网络处理的结果是一个四维数组，需要将这个结果压缩成一个特征向量（一维数组）
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


# 获取一张图片经过Inception-v3模型处理之后的特征向量，先试图寻找已经计算且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    # 获取一张图片对应的特征向量文件路径
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)
    # 如果这个特征向量文件不存在，则通过Inception-v3模型来计算特征向量，并将计算结果存入文件
    if not os.path.exists(bottleneck_path):
        # 获取原始的图片路径
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        # 获取图片内容
        image_data = gfile.FastGFile(image_path, 'rb').read()
        # 通过Inception-v3模型来计算特征向量
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        # 将计算得到的特征向量存入文件
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        # 直接从文件中获取图片相应的特征向量
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    # 返回得到的特征向量
    return bottleneck_values


# 随机获取一个batch的图片作为训练数据
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor,
                                  bottleneck_tensor):
    bottlenecks = []
    ground_truths = []

    for _ in range(how_many):
        # 随机一个类别和图片的编号加入当前的训练数据
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# 获取全部的测试数据。在最终测试的时候需要在所有的测试数据上计算正确率


def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    # 枚举所有的类别和每个类别中的测试图片
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            # 通过Inception-v3模型计算图片对应的特征向量，并将其加入最终数据列表
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor,
                                                  bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


# Attach a lot of summaries to a Tensor (for TensorBoard visualization)


def save_graph_to_file(sess, graph, graph_file_name):
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [final_tensor_name])
    with gfile.FastGFile(graph_file_name, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    return


def ensure_dir_exists(dir_name):
    """
    :param dir_name:    Path string to the folder we want to create.
    :return:            None
    """
    """Makes sure the folder exists on disk.
    Args:
      dir_name: Path string to the folder we want to create.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# Makes sure the folder exists on disk.


# Setup the directory we'll write summaries to for TensorBoard
def prepare_file_system(dir_path):
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(dir_path):
        tf.gfile.DeleteRecursively(dir_path)

    tf.gfile.MakeDirs(dir_path)
    # if intermediate_store_frequency > 0:
    #     ensure_dir_exists(intermediate_output_graphs_dir)
    return


# Given the name of a model architecture, returns information about it
def create_model_info(architecture):
    """
    There are different base image recognition pretrained models that can be
    retrained using transfer learning, and this function translates from the name
    of a model to the attributes that are needed to download and train with it.
    :param architecture:    Name of a model architecture.
    :return:                Dictionary of information about the model, or None if the name isn't recognized
    """
    architecture = architecture.lower()
    if architecture == 'inception_v3':
        # pylint: disable=line-too-long
        data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

        # Inception-v3 模型中代表瓶颈层结果的张量名称。在谷歌提供的Inception-v3模型中，
        # 这个张量的名称就是'pool_3/_reshape:0'。在训练模型时，可以通过tensor.name来获取张量的名称
        # pylint: enable=line-too-long
        bottleneck_tensor_name = 'pool_3/_reshape:0'
        # 图像输入张量所对应的名称
        JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
        bottleneck_tensor_size = 2048
        input_width = 299
        input_height = 299
        input_depth = 3
        resized_input_tensor_name = 'Mul:0'
        # 下载的谷歌训练好的Inception-v3模型文件名
        model_file_name = 'classify_image_graph_def.pb'
        input_mean = 128
        input_std = 128
    elif architecture.startswith('mobilenet_'):
        parts = architecture.split('_')
        if len(parts) != 3 and len(parts) != 4:
            tf.logging.error("Couldn't understand architecture name '%s'",
                             architecture)
            return None
        version_string = parts[1]
        if (version_string != '1.0' and version_string != '0.75' and
                version_string != '0.50' and version_string != '0.25'):
            tf.logging.error(
                """"The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25',
        but found '%s' for architecture '%s'""",
                version_string, architecture)
            return None
        size_string = parts[2]
        if (size_string != '224' and size_string != '192' and
                size_string != '160' and size_string != '128'):
            tf.logging.error(
                """The Mobilenet input size should be '224', '192', '160', or '128',
       but found '%s' for architecture '%s'""",
                size_string, architecture)
            return None
        if len(parts) == 3:
            is_quantized = False
        else:
            if parts[3] != 'quantized':
                tf.logging.error(
                    "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
                    architecture)
                return None
            is_quantized = True
        data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
        data_url += version_string + '_' + size_string + '_frozen.tgz'
        bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
        bottleneck_tensor_size = 1001
        input_width = int(size_string)
        input_height = int(size_string)
        input_depth = 3
        resized_input_tensor_name = 'input:0'
        if is_quantized:
            model_base_name = 'quantized_graph.pb'
        else:
            model_base_name = 'frozen_graph.pb'
        model_dir_name = 'mobilenet_v1_' + version_string + '_' + size_string
        model_file_name = os.path.join(model_dir_name, model_base_name)
        input_mean = 127.5
        input_std = 127.5
    else:
        tf.logging.error("Couldn't understand architecture name '%s'", architecture)
        raise ValueError('Unknown architecture', architecture)

    return {
        'data_url': data_url,
        'bottleneck_tensor_name': bottleneck_tensor_name,
        'bottleneck_tensor_size': bottleneck_tensor_size,
        'input_width': input_width,
        'input_height': input_height,
        'input_depth': input_depth,
        'resized_input_tensor_name': resized_input_tensor_name,
        'model_file_name': model_file_name,
        'input_mean': input_mean,
        'input_std': input_std,
        'image_input_tensor_name': JPEG_DATA_TENSOR_NAME
    }


# Creates a graph from saved GraphDef file and returns a Graph object.
def create_model_graph(model_info):
    """
    :param model_info:  Dictionary containing information about the model architecture.
    :return:            Graph holding the trained Inception network, and various tensors we'll be manipulating.
    """
    # 读取已经训练好的Inception-v3模型。谷歌训练好的模型保存在了GraphDef Protocol Buffer中，里面保存了每个节点取值的计算方法
    # 以及变量的取值。并进行TensorFlow模型持久化操作
    with tf.Graph().as_default() as graph:
        with gfile.FastGFile(os.path.join(MODEL_DIR, model_info['model_file_name']), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            # 加载读取的Inception-v3模型，并返回数据输入所对应的张量以及计算瓶颈层结果所对应的张量
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[
                model_info['bottleneck_tensor_name'], model_info['image_input_tensor_name']])

    return graph, bottleneck_tensor, jpeg_data_tensor


# Adds a new softmax and fully-connected layer for training.
def add_final_training_ops(n_classes, bottleneck_tensor, bottleneck_tensor_size):
    """
      We need to retrain the top layer to identify our new classes, so this function
      adds the right operations to the graph, along with some variables to hold the
      weights, and then sets up all the gradients for the backward pass.

    :param n_classes:               Integer of how many categories of things we're trying to recognize
    :param bottleneck_tensor:       The output of the main CNN graph
    :param bottleneck_tensor_size:  How many entries in the bottleneck vector
    :return:                        The tensors for the training and cross entropy results, and tensors for the
                                     bottleneck input and ground truth input.
    """

    # 定义新的神经网络输入
    with tf.name_scope('input'):
        # 定义新的神经网络输入，这个输入就是新的图片经过Inception-v3模型前向传播到达瓶颈层时的节点取值。
        # 可以将这个过程类似得理解为一种特征提取
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor,
            shape=[None, bottleneck_tensor_size],
            name='BottleneckInputPlaceholder')
        # 定义新的标准答案输入
    ground_truth_input = tf.placeholder(tf.float32, shape=[None, n_classes], name='GroundTruthInput')

    # 定义一层全链接层来解决新的图片分类问题。因为训练好的Inception-v3模型已经将原始的图片抽象为了更容易分类的特征向量了
    # 所以不需要再训练那么复杂的神经网络来完成这个新的任务。
    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            initial_value = tf.truncated_normal([bottleneck_tensor_size, n_classes], stddev=0.001)
            layer_weights = tf.Variable(initial_value, name='final_weights')
            variable_summaries(layer_weights)
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([n_classes]), name='final_biases')
            variable_summaries(layer_biases)
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.summary.histogram('activations', final_tensor)

    # 定义交叉熵损失函数。
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=ground_truth_input, logits=logits)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy, cross_entropy_mean, bottleneck_input, ground_truth_input,
            final_tensor)


# Inserts the operations we need to evaluate the accuracy of our results
def add_evaluation_step(final_tensor, ground_truth_input):
    """
    :param final_tensor:        The new final node that produces results
    :param ground_truth_input:  The node we feed ground truth data
    :return:                    Tuple of (evaluation step, prediction)
    """
    # 计算正确率。
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        with tf.name_scope('accuracy'):
            accuracy_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy_step)
    return accuracy_step


# 运行主程序
def main(_):
    # Needed to make sure the logging output is visible.
    # See https://github.com/tensorflow/tensorflow/issues/3047
    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare necessary directories that can be used during training
    prepare_file_system(os.path.join(summaries_dir, 'train'))
    prepare_file_system(os.path.join(summaries_dir, 'validation'))

    # Gather information about the model architecture we'll be using.
    model_info = create_model_info('inception_v3')
    if not model_info:
        tf.logging.error('Did not recognize architecture flag')
        return -1

    # Set up the pre-trained graph.
    graph, bottleneck_tensor, jpeg_data_tensor = create_model_graph(model_info)

    # 读取所有的图片
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    # 判断待训练数据集是否符合训练要求
    if n_classes == 0:
        tf.logging.error('No valid folders of images found at ' + INPUT_DATA)
        return -1
    if n_classes == 1:
        tf.logging.error('Only one valid folder of images found at ' +
                         INPUT_DATA +
                         ' - multiple classes are needed for classification.')
        return -1

    # 开始训练
    with tf.Session(graph=graph) as sess:
        # 计时开始
        start = time.clock()

        # 加入待训练的新层以适应自己数据的分类类别
        # Add the new layer that we'll be training.
        (train_step, cross_entropy, cross_entropy_mean, bottleneck_input, ground_truth_input, final_tensor) = \
            add_final_training_ops(len(image_lists.keys()),
                                   bottleneck_tensor,
                                   model_info['bottleneck_tensor_size'])
        # Create the operations we need to evaluate the accuracy of our new layer
        accuracy_step = add_evaluation_step(final_tensor, ground_truth_input)

        # Merge all the summaries and write them out to the summaries_dir
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summaries_dir + 'train', sess.graph)
        validation_writer = tf.summary.FileWriter(summaries_dir + 'validation')

        # Set up all our weights to their initial default values.
        init = tf.global_variables_initializer()
        sess.run(init)

        #  首先，使用tf.train.Saver()创建模型的保存器。
        saver = tf.train.Saver(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=max_to_keep)

        # 获取最近一次训练好的模型
        if os.path.exists(os.path.join(summaries_dir, 'checkpoint')):
            ckpt = tf.train.get_checkpoint_state(summaries_dir)
            latest_checkpoint_model = ckpt.model_checkpoint_path
            # 只加载模型数据
            saver.restore(sess, latest_checkpoint_model)
            checkpoint_step = int(latest_checkpoint_model.split('/')[-1].split('-')[-1])
            print('restoring successfully from path: ', latest_checkpoint_model)
            print('from %d step start training...' % int(checkpoint_step + 1))
        else:
            checkpoint_step = -1

        # 训练过程。
        for i in range(checkpoint_step + 1, STEPS):
            # 每次获取一个batch的训练数据
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)

            # Feed the bottlenecks and ground truth into the graph, and run a training
            train_summary, _ = sess.run([merged, train_step], feed_dict={
                bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

            # Capture training summaries for TensorBoard with the `merged` op.
            train_writer.add_summary(train_summary, i)

            # 在验证数据上测试正确率
            if i % 100 == 0 or i + 1 == STEPS:
                # Feed the bottlenecks and ground truth into the graph, and run a training
                train_accuracy, cross_entropy_mean_value = sess.run([accuracy_step, cross_entropy_mean], feed_dict={
                    bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

                # 每次获取一个batch的验证数据
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
                # 获取每100代训练后的验证准确率和交叉熵值信息
                validation_summary, validation_accuracy = sess.run([merged, accuracy_step], feed_dict={
                    bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                # capture training summaries for TensorBoard with the `merged` op.
                validation_writer.add_summary(validation_summary, i)

                # 打印训练信息
                print('%s: Step %d: Train accuracy = %.1f%% (N=%d)'
                      % (datetime.now(), i, train_accuracy * 100, len(train_bottlenecks)))
                print('%s: Step %d: Validation accuracy on random sampled %d examples = %.1f%%'
                      % (datetime.now(), i, BATCH, validation_accuracy * 100))
                print('%s: Step %d: Cross entropy mean = %f (N=%d)'
                      % (datetime.now(), i, cross_entropy_mean_value, len(validation_bottlenecks)))

            if i % 400 == 0:
                # 将模型保存到 summaries_dir + "/model.ckpt文件
                saver.save(sess, summaries_dir + "model.ckpt", global_step=i)
                print('Saving model successfully to  ', summaries_dir + "model.ckpt-" + str(i))
                # Write out the trained graph and labels with the weights stored as constants.
                save_graph_to_file(sess, graph, output_graph)
                with gfile.FastGFile(output_labels, 'w') as f:
                    f.write('\n'.join(image_lists.keys()) + '\n')
                print('Saving graph successfully to  ', output_graph)

            # # Store intermediate results
            # if (intermediate_store_frequency > 0 and (i % intermediate_store_frequency == 0)
            #         and i > 0):
            #     intermediate_file_name = (intermediate_output_graphs_dir +
            #                               'intermediate_' + str(i) + '.pb')
            #     tf.logging.info('Save intermediate result to : ' +
            #                     intermediate_file_name)
            #     save_graph_to_file(sess, graph, intermediate_file_name)

        # 在最后的测试数据上测试正确率。
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(
            sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(accuracy_step, feed_dict={
            bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%% (N=%d)' % (test_accuracy * 100, len(test_bottlenecks)))

        # 将模型保存到 summaries_dir + "/model.ckpt文件
        saver.save(sess, summaries_dir + "model.ckpt", global_step=STEPS)
        print('Saving model successfully to  ', summaries_dir + "model.ckpt-" + str(STEPS))
        # Write out the trained graph and labels with the weights stored as constants.
        save_graph_to_file(sess, graph, output_graph)
        with gfile.FastGFile(output_labels, 'w') as f:
            f.write('\n'.join(image_lists.keys()) + '\n')

    # 训练结束，计时结束，并统计运行时长
    end = time.clock()
    print('\n训练总时长：%f s --- %f min' % (end - start, (end - start) / 60.0))

    train_writer.close()
    validation_writer.close()


if __name__ == '__main__':
    tf.app.run()

# 训练模型可视化： tensorboard --logdir=output_graph/retrain_logs
