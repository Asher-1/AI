# -*- coding: UTF-8 -*-

"""
测试神经网络模型

大家之后可以加上各种的 name_scope（命名空间）
用 TensorBoard 来可视化

==== 一些术语的概念 ====
# Batch size : 批次(样本)数目。一次迭代（Forword 运算（用于得到损失函数）以及 BackPropagation 运算（用于更新神经网络参数））所用的样本数目。Batch size 越大，所需的内存就越大
# Iteration : 迭代。每一次迭代更新一次权重（网络参数），每一次权重更新需要 Batch size 个数据进行 Forward 运算，再进行 BP 运算
# Epoch : 纪元/时代。所有的训练样本完成一次迭代

# 假如 : 训练集有 1000 个样本，Batch_size=10
# 那么 : 训练完整个样本集需要： 100 次 Iteration，1 个 Epoch
# 但一般我们都不止训练一个 Epoch

==== 超参数（Hyper parameter）====
init_scale : 权重参数（Weights）的初始取值跨度，一开始取小一些比较利于训练
learning_rate : 学习率，训练时初始为 1.0
num_layers : LSTM 层的数目（默认是 2）
num_steps : LSTM 展开的步（step）数，相当于每个批次输入单词的数目（默认是 35）
hidden_size : LSTM 层的神经元数目，也是词向量的维度（默认是 650）
max_lr_epoch : 用初始学习率训练的 Epoch 数目（默认是 10）
dropout : 在 Dropout 层的留存率（默认是 0.5）
lr_decay : 在过了 max_lr_epoch 之后每一个 Epoch 的学习率的衰减率，训练时初始为 0.93。让学习率逐渐衰减是提高训练效率的有效方法
batch_size : 批次(样本)数目。一次迭代（Forword 运算（用于得到损失函数）以及 BackPropagation 运算（用于更新神经网络参数））所用的样本数目
（batch_size 默认是 20。取比较小的 batch_size 更有利于 Stochastic Gradient Descent（随机梯度下降），防止被困在局部最小值）
"""

from RNN.utils import *
from RNN.network import *


def test(model_path, test_data, vocab_size, id_to_word):
    # 测试的输入
    test_input = Input(batch_size=20, num_steps=35, data=test_data)

    # 创建测试的模型，基本的超参数需要和训练时用的一致，例如：
    # hidden_size，num_steps，num_layers，vocab_size，batch_size 等等
    # 因为我们要载入训练时保存的参数的文件，如果超参数不匹配 TensorFlow 会报错
    m = Model(test_input, is_training=False, hidden_size=650, vocab_size=vocab_size, num_layers=2)

    # 为了用 Saver 来恢复训练时生成的模型的变量
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Coordinator（协调器），用于协调线程的运行
        coord = tf.train.Coordinator()
        # 启动线程
        threads = tf.train.start_queue_runners(coord=coord)

        # 当前的状态
        # 第二维是 2 是因为测试时指定只有 2 层 LSTM
        # 第二维是 2 是因为对每一个 LSTM 单元有两个来自上一单元的输入：
        # 一个是 前一时刻 LSTM 的输出 h(t-1)
        # 一个是 前一时刻的单元状态 C(t-1)
        current_state = np.zeros((2, 2, m.batch_size, m.hidden_size))

        # 恢复被训练的模型的变量
        saver.restore(sess, model_path)

        # 测试 30 个批次
        num_acc_batches = 30

        # 打印预测单词和实际单词的批次数
        check_batch_idx = 25

        # 超过 5 个批次才开始累加精度
        acc_check_thresh = 5

        # 初始精度的和，用于之后算平均精度
        accuracy = 0

        for batch in range(num_acc_batches):
            if batch == check_batch_idx:
                true, pred, current_state, acc = sess.run([m.input_obj.targets, m.predict, m.state, m.accuracy], feed_dict={m.init_state: current_state})
                pred_words = [id_to_word[x] for x in pred[:m.num_steps]]
                true_words = [id_to_word[x] for x in true[0]]
                print("\n实际的单词:")
                print(" ".join(true_words))  # 真实的单词
                print("预测的单词:")
                print(" ".join(pred_words))  # 预测的单词
            else:
                acc, current_state = sess.run([m.accuracy, m.state], feed_dict={m.init_state: current_state})
            if batch >= acc_check_thresh:
                accuracy += acc

        # 打印平均精度
        print("平均精度: {:.3f}".format(accuracy / (num_acc_batches - acc_check_thresh)))

        # 关闭线程
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    if args.data_path:
        data_path = args.data_path
    if args.load_file:
        load_file = args.load_file
    train_data, valid_data, test_data, vocab_size, id_to_word = load_data(data_path)

    trained_model = save_path + "/" + load_file

    test(trained_model, test_data, vocab_size, id_to_word)
