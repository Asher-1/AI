# -*- coding:utf-8 -*-
# Created Time: Thu 13 Apr 2017 04:07:50 PM CST
# $Author: Taihong Xiao <xiaotaihong@126.com>

import tensorflow as tf
import os
import time
from model import Model
from dataset import config, Dataset
import numpy as np
from scipy import misc
import argparse

# 自定义参数设置
max_to_keep = 2
checkpoint_step = -1


def run(config, dataset, model, gpu):
    global checkpoint_step
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    batch1, batch2 = dataset.input()

    # Make saver.
    saver = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=max_to_keep)

    # image summary
    Ax_op = tf.summary.image('Ax', model.Ax, max_outputs=30)
    Be_op = tf.summary.image('Be', model.Be, max_outputs=30)
    Ax2_op = tf.summary.image('Ax2', model.Ax2, max_outputs=30)
    Be2_op = tf.summary.image('Be2', model.Be2, max_outputs=30)
    Bx_op = tf.summary.image('Bx', model.Bx, max_outputs=30)
    Ae_op = tf.summary.image('Ae', model.Ae, max_outputs=30)

    # G loss summary
    for key in model.G_loss.keys():
        tf.summary.scalar(key, model.G_loss[key])

    loss_G_nodecay_op = tf.summary.scalar('loss_G_nodecay', model.loss_G_nodecay)
    loss_G_decay_op = tf.summary.scalar('loss_G_decay', model.loss_G_decay)
    loss_G_op = tf.summary.scalar('loss_G', model.loss_G)

    # D loss summary
    for key in model.D_loss.keys():
        tf.summary.scalar(key, model.D_loss[key])

    loss_D_op = tf.summary.scalar('loss_D', model.loss_D)

    # learning rate summary
    g_lr_op = tf.summary.scalar('g_learning_rate', model.g_lr)
    d_lr_op = tf.summary.scalar('d_learning_rate', model.d_lr)

    merged_op = tf.summary.merge_all()

    # start training
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(config.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # 读取最近一次训练的参数数据
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('restoring successfully from path: ', ckpt.model_checkpoint_path)
        # 设置爱检查点
        checkpoint_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print('from %d step start training...' % int(checkpoint_step + 1))

    writer = tf.summary.FileWriter(config.log_dir, sess.graph)
    writer.add_graph(sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(checkpoint_step + 1, config.max_iter):
        d_num = 100 if i % 500 == 0 else 1

        # update D with clipping 
        for j in range(d_num):
            _, loss_D_sum, _ = sess.run([model.d_opt, model.loss_D, model.clip_d],
                                        feed_dict={model.Ax: sess.run(batch1),
                                                   model.Be: sess.run(batch2),
                                                   model.g_lr: config.g_lr(epoch=i),
                                                   model.d_lr: config.d_lr(epoch=i)
                                                   })

        # update G 
        _, loss_G_sum = sess.run([model.g_opt, model.loss_G],
                                 feed_dict={model.Ax: sess.run(batch1),
                                            model.Be: sess.run(batch2),
                                            model.g_lr: config.g_lr(epoch=i),
                                            model.d_lr: config.d_lr(epoch=i)
                                            })

        print('iter: {:06d},   g_loss: {}    d_loss: {}'.format(i, loss_D_sum, loss_G_sum))

        if i % 20 == 0:
            merged_summary = sess.run(merged_op,
                                      feed_dict={model.Ax: sess.run(batch1),
                                                 model.Be: sess.run(batch2),
                                                 model.g_lr: config.g_lr(epoch=i),
                                                 model.d_lr: config.d_lr(epoch=i)
                                                 })

            writer.add_summary(merged_summary, i)

        if i % 500 == 0:
            saver.save(sess, os.path.join(config.model_dir, 'model.ckpt-{:06d}'.format(i)))
            print('Saving model successfully to  ', os.path.join(config.model_dir, 'model.ckpt-{:06d}'.format(i)))

            img_Ax, img_Be, img_Ae, img_Bx, img_Ax2, img_Be2 = sess.run(
                [model.Ax, model.Be, model.Ae, model.Bx, model.Ax2, model.Be2],
                feed_dict={model.Ax: sess.run(batch1), model.Be: sess.run(batch2)})

            for j in range(5):
                img = np.concatenate((img_Ax[j], img_Be[j], img_Ae[j], img_Bx[j], img_Ax2[j], img_Be2[j]), axis=1)
                misc.imsave(os.path.join(config.sample_img_dir, 'iter_{:06d}_{}.jpg'.format(i, j)), img)

    writer.close()
    saver.save(sess, os.path.join(config.model_dir, 'model.ckpt'), global_step=config.max_iter)
    print('Saving model successfully to  ', os.path.join(config.model_dir, 'model.ckpt-' + str(config.max_iter)))
    print('Training complete')

    coord.request_stop()
    coord.join(threads)


def main():
    # 所有可训练特征
    features = '5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young '
    features = features.split()
    # 设置训练特征
    # attribute = features[5]  # Bangs
    attribute = 'Young'  # Bangs
    config.set_model_dir_name('feature_' + attribute)

    parser = argparse.ArgumentParser(description='test', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-a', '--attribute',
        default=attribute,
        type=str,
        help='Specify attribute name for training. \ndefault: %(default)s. \nAll attributes can be found in list_attr_celeba.txt'
    )
    parser.add_argument(
        '-g', '--gpu',
        default='0',
        type=str,
        help='Specify GPU id. \ndefault: %(default)s. \nUse comma to seperate several ids, for example: 0,1'
    )
    args = parser.parse_args()

    celebA = Dataset(args.attribute)
    GeneGAN = Model(is_train=True)
    run(config, celebA, GeneGAN, gpu=args.gpu)


if __name__ == "__main__":
    start = time.clock()
    main()
    # 训练结束，计时结束，并统计运行时长
    end = time.clock()
    print('\n训练总时长：%f s --- %f min' % (end - start, (end - start) / 60.0))
