import os
import scipy.misc
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json

import tensorflow as tf


def main(_):
    flags = tf.app.flags
    flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
    flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
    flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
    flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
    flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
    flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
    flags.DEFINE_integer("input_width", 108,
                         "The size of image to use (will be center cropped). If None, same value as input_height [None]")
    flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
    flags.DEFINE_integer("output_width", 64,
                         "The size of the output images to produce. If None, same value as output_height [None]")
    flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
    flags.DEFINE_string("dataset", "faces_", "The name of dataset [celebA, mnist, lsun]")
    flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
    flags.DEFINE_string("work_dir", ROOT_PATH, "Directory name to work [./]")
    flags.DEFINE_string("checkpoint_dir", CHECK_POINT, "Directory name to save the checkpoints [checkpoint]")
    flags.DEFINE_string("sample_dir", SAMPLE_DIR, "Directory name to save the image samples [samples]")
    flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
    flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
    flags.DEFINE_boolean("visualize", True, "True for visualizing, False for nothing [False]")
    FLAGS = flags.FLAGS

    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        if FLAGS.dataset == 'mnist':
            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                y_dim=10,
                c_dim=1,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                is_crop=FLAGS.is_crop,
                work_dir=FLAGS.work_dir,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir)
        else:
            dcgan = DCGAN(
                sess,
                input_width=FLAGS.input_width,
                input_height=FLAGS.input_height,
                output_width=FLAGS.output_width,
                output_height=FLAGS.output_height,
                batch_size=FLAGS.batch_size,
                c_dim=FLAGS.c_dim,
                dataset_name=FLAGS.dataset,
                input_fname_pattern=FLAGS.input_fname_pattern,
                is_crop=FLAGS.is_crop,
                work_dir=FLAGS.work_dir,
                checkpoint_dir=FLAGS.checkpoint_dir,
                sample_dir=FLAGS.sample_dir)

        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir):
                raise Exception("[!] Train a model first, then run test mode")
        # js_dir = os.path.join(FLAGS.work_dir, 'web/js/')
        # if not os.path.exists(js_dir):
        #     os.makedirs(js_dir)
        # to_json(os.path.join(js_dir, "layers.js"),
        #         [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
        #         [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
        #         [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
        #         [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
        #         [dcgan.h4_w, dcgan.h4_b, None])

        # Below is codes for visualization
        if FLAGS.visualize:
            OPTION = 2
            visualize(sess, dcgan, FLAGS, OPTION)


# python main.py --dataset faces_ --is_train True
if __name__ == '__main__':
    ROOT_PATH = 'D:/develop/workstations/GitHub/Datasets/DL/Images/GAN_db/DCGAN/'
    CHECK_POINT = ROOT_PATH + 'checkpoint'
    SAMPLE_DIR = ROOT_PATH + 'samples'
    tf.app.run()
