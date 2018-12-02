import sys
sys.path.append('../../')
import numpy as np
import os
import tensorflow as tf
from data_provider.jump_data_fine import JumpData
from model import JumpModel
from tqdm import tqdm
import argparse, cv2

def get_a_test(name):
    posi = name.index('_res')
    img_name = name[:posi] + '.png'
    x, y = name[name.index('_h_') + 3: name.index('_h_') + 6], name[name.index('_w_') + 3: name.index('_w_') + 6]
    x, y = int(x), int(y)
    img = cv2.imread(img_name)
    x1 = x - 160
    x2 = x + 160
    y1 = y - 160
    y2 = y + 160
    x = 160
    y = 160
    if y1 < 0:
        y = 160 + y1
        y1 = 0
        y2 = 320
    if y2 > img.shape[1]:
        y = 160 + y2 - img.shape[1]
        y2 = img.shape[1]
        y1 = y2 - 320
    img = img[x1: x2, y1: y2, :]
    label = np.array([x, y], dtype=np.float32)
    return img[np.newaxis, :, :, :], label.reshape((1, label.shape[0]))


global_step = 100000
batch_size = 10
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', default=0, type=int)
    args = parser.parse_args()

    if args is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    net = JumpModel()
    dataset = JumpData()
    img = tf.placeholder(tf.float32, [None, 320, 320, 3], name='img')
    label = tf.placeholder(tf.float32, [None, 2], name='label')
    is_training = tf.placeholder(np.bool, name='is_training')
    keep_prob = tf.placeholder(np.float32, name='keep_prob')
    lr = tf.placeholder(np.float32, name='lr')

    pred = net.forward(img, is_training, keep_prob)
    loss = tf.reduce_mean(tf.sqrt(tf.pow(pred - label, 2) + 1e-12))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
    saver = tf.train.Saver()

    sess = tf.Session()
    merged = tf.summary.merge_all()
    if not os.path.isdir('./logs'):
        os.mkdir('./logs')
    train_writer = tf.summary.FileWriter(os.path.join('./logs'), sess.graph)
    sess.run(tf.global_variables_initializer())
    if not os.path.isdir('./train_logs'):
        os.mkdir('./train_logs')
    ckpt = tf.train.get_checkpoint_state('./train_logs')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('restore model successfully!')

    best_val_loss = 1e10
    for i in range(global_step):
        # train
        batch = dataset.next_batch(batch_size)
        feed_dict = {
            img: batch['img'],
            label: batch['label'],
            is_training: True,
            keep_prob: 0.75,
            lr: 0.01 - 0.0000001 * i,
        }
        sess.run(train_op, feed_dict=feed_dict)
        if (i + 1) % 10 == 0:
            l, merged_str = sess.run([loss, merged], feed_dict=feed_dict)
            print('batch: %05d, train_loss: %.4f' % (i, l))
            train_writer.add_summary(merged_str, i)
        # if (i + 1) % 100 == 9:
        #     saver.save(sess, os.path.join('./train_logs', 'model.ckpt'), i)
        if (i + 1) % 1000 == 0:
            for idx, name in enumerate(tqdm(dataset.val_name_list)):
                val_img, val_label = get_a_test(name)
                feed_dict = {
                    img: val_img,
                    label: val_label,
                    is_training: False,
                    keep_prob: 1.0,
                    lr: 0.01,
                }
                if idx == 0:
                    val_loss = [sess.run(loss, feed_dict=feed_dict)]
                else:
                    val_loss.append(sess.run(loss, feed_dict=feed_dict))
            val_loss = sum(val_loss) / len(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                saver.save(sess, os.path.join('./train_logs', 'best_model.ckpt'), global_step=i)
            print('val_loss: %.4f, best_val_loss: %.4f' % (val_loss, best_val_loss))

