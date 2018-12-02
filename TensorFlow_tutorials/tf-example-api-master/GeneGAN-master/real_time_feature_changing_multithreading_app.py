# coding=utf-8

import time
import tensorflow as tf
import numpy as np
from model import Model
# from dataset import Dataset
import os
import cv2
from scipy import misc
import argparse
from queue import Queue
from threading import Thread
from object_detection.utils.app_utils import FPS, WebcamVideoStream, draw_boxes_and_labels


def swap_attribute(src_img, att_img, model_dir, model, gpu):
    '''
    Input
        src_img: the source image that you want to change its attribute
        att_img: the attribute image that has certain attribute
        model_dir: the directory that contains the checkpoint, ckpt.* files
        model: the GeneGAN network that defined in train.py
        gpu: for example, '0,1'. Use '' for cpu mode
    Output
        out1: src_img with attributes
        out2: att_img without attributes
    '''

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        # print(ckpt)
        # print(ckpt.model_checkpoint_path)

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        out2, out1 = sess.run([model.Ae, model.Bx], feed_dict={model.Ax: att_img, model.Be: src_img})
        save_path1 = os.path.join('output', model_path_name, 'swap_out1.jpg')
        save_path2 = os.path.join('output', model_path_name, 'swap_out2.jpg')
        misc.imsave(save_path1, out1[0])
        misc.imsave(save_path2, out2[0])
        # print('结果图保存到：', save_path1, save_path2)
        return out1, out2


def interpolation(src_img, att_img, inter_num, model_dir, model, gpu):
    '''
    Input
        src_img: the source image that you want to change its attribute
        att_img: the attribute image that has certain attribute
        inter_num: number of interpolation points
        model_dir: the directory that contains the checkpoint, ckpt.* files
        model: the GeneGAN network that defined in train.py
        gpu: for example, '0,1'. Use '' for cpu mode
    Output
        out: [src_img, inter1, inter2, ..., inter_{inter_num}]
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(model_dir)
        # print(ckpt)
        # print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        out = src_img[0]
        for i in range(1, inter_num + 1):
            lambda_i = i / float(inter_num)
            model.out_i = model.joiner('G_joiner', model.B, model.x * lambda_i)
            out_i = sess.run(model.out_i, feed_dict={model.Ax: att_img, model.Be: src_img})
            out = np.concatenate((out, out_i[0]), axis=1)
        # print(out.shape)
        save_path = os.path.join('output', model_path_name, 'interpolation.jpg')
        misc.imsave(save_path, out)
        print('结果图保存到：', save_path)


def interpolation2(src_img, att_img, inter_num, model_dir, model, gpu):
    '''
    Input
        src_img: the source image that you want to change its attribute
        att_img: the attribute image that has certain attribute
        inter_num: number of interpolation points
        model_dir: the directory that contains the checkpoint, ckpt.* files
        model: the GeneGAN network that defined in train.py
        gpu: for example, '0,1'. Use '' for cpu mode
    Output
        out: [src_img, inter1, inter2, ..., inter_{inter_num}]
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(model_dir)
        # print(ckpt)
        # print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        B, src_feat = sess.run([model.B, model.e], feed_dict={model.Be: src_img})
        att_feat = sess.run(model.x, feed_dict={model.Ax: att_img})

        out = src_img[0]
        for i in range(1, inter_num + 1):
            lambda_i = i / float(inter_num)
            out_i = sess.run(model.joiner('G_joiner', B, src_feat + (att_feat - src_feat) * lambda_i))
            out = np.concatenate((out, out_i[0]), axis=1)
        # print(out.shape)
        save_path = os.path.join('output', model_path_name, 'interpolation2.jpg')
        misc.imsave(save_path, out)
        print('结果图保存到：', save_path)


def interpolation_matrix(src_img, att_imgs, size, model_dir, model, gpu):
    '''
    Input
        src_img: the source image that you want to change its attribute [1, h, w, c]
        att_imgs: four attribute images that has certain attribute [4, h, w, c]
        size: the size of output matrix
        model_dir: the directory that contains the checkpoint, ckpt.* files
        model: the GeneGAN network that defined in train.py
        gpu: for example, '0,1'. Use '' for cpu mode
    Output
        out1: image matrix
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ckpt = tf.train.get_checkpoint_state(model_dir)
        # print(ckpt)
        # print(ckpt.model_checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        m, n = size
        h, w = model.height, model.width

        rows = [[1 - i / float(m - 1), i / float(m - 1)] for i in range(m)]
        cols = [[1 - i / float(n - 1), i / float(n - 1)] for i in range(n)]
        four_tuple = []
        for row in rows:
            for col in cols:
                four_tuple.append([row[0] * col[0], row[0] * col[1], row[1] * col[0], row[1] * col[1]])

        attributes = [sess.run(model.x, feed_dict={model.Ax: att_imgs[i:i + 1]}) for i in range(4)]
        B = sess.run(model.B, feed_dict={model.Be: src_img})

        cnt = 0
        out = np.zeros((0, w * n, model.channel))
        for i in range(m):
            out_row = np.zeros((h, 0, model.channel))
            for j in range(n):
                four = four_tuple[cnt]
                attribute = sum([four[i] * attributes[i] for i in range(4)])
                # print(attribute.shape)
                img = sess.run(model.joiner('G_joiner', B, attribute))[0]
                out_row = np.concatenate((out_row, img), axis=1)
                cnt += 1
            out = np.concatenate((out, out_row), axis=0)

        first_col = np.concatenate((att_imgs[0], 255 * np.ones(((m - 2) * h, w, 3)), att_imgs[2]), axis=0)

        last_col = np.concatenate((att_imgs[1], 255 * np.ones(((m - 2) * h, w, 3)), att_imgs[3]), axis=0)

        out_canvas = np.concatenate((first_col, out, last_col), axis=1)
        save_path = os.path.join('output', model_path_name, 'four_matrix.jpg')
        misc.imsave(save_path, out_canvas)
        print('结果图保存到：', save_path)


classifier_model = os.path.dirname(__file__) + "/models/classifiers/sex_SVC_classifier.pkl"


# 添加识别标记
def add_overlays(frame, faces, points):
    if faces is not None:
        num = -1
        for face in faces:
            num += 1
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), (0, 255, 0), 2)
            # draw feature points
            cv2.circle(frame, (points[0][num], points[5][num]), 2, (0, 255, 0), -1)
            cv2.circle(frame, (points[1][num], points[6][num]), 2, (0, 255, 0), -1)
            cv2.circle(frame, (points[2][num], points[7][num]), 2, (0, 255, 0), -1)
            cv2.circle(frame, (points[3][num], points[8][num]), 2, (0, 255, 0), -1)
            cv2.circle(frame, (points[4][num], points[9][num]), 2, (0, 255, 0), -1)
            # 绘制所属分类类别
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)

    # cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
    #             thickness=2, lineType=2)


# 线程
def worker(input_q, output_q):
    # 创建脸部识别器
    fps = FPS().start()
    while True:
        fps.update()
        data = input_q.get()
        src_img = data['class_input_img']
        att_img = data['class_target_img']
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        GeneGAN = data['class_model']
        # saver = data['class_saver']
        # GeneGAN = Model(is_train=False, nhwc=[1, 64, 64, 3])
        if args.mode == 'swap':
            src_output, att_output = swap_attribute(src_img, att_img, args.model_dir, GeneGAN, args.gpu)
        elif args.mode == 'interpolation':
            interpolation(src_img, att_img, args.num, args.model_dir, GeneGAN, args.gpu)
        elif args.mode == 'matrix':
            att_imgs = np.array([misc.imresize(misc.imread(img), (GeneGAN.height, GeneGAN.width)) for img in args.targets])
            interpolation_matrix(src_img, att_imgs, args.size, args.model_dir, GeneGAN, args.gpu)
        else:
            raise NotImplementationError()

        # 识别结果放入线程
        output_q.put(dict(class_input_img=src_output, class_target_img=att_output))

    fps.stop()
    sess.close()


if __name__ == '__main__':
    base_dir = '../../../datasets/celebA/align_5p/'
    model_dir = 'G:/develop/PycharmProjects/datasets/result_outputs/GeneGAN_master_output'

    # 用户自定义修改文件
    model_path_name = output_dir_name = 'feature_Eyeglasses'
    if not tf.gfile.Exists(os.path.join('output', model_path_name)):
        tf.gfile.MakeDirs(os.path.join('output', model_path_name))

    output_model_path = os.path.join(model_dir, model_path_name)

    #  Swapping of Attributes
    mode = 'swap'
    m_target = base_dir + '000193.jpg'
    # m_target = base_dir + '022344.jpg'
    m_targets = None
    interpolations_num = '2'
    m_size = [3, 3]

    # # Linear Interpolation of Image Attributes
    # mode = 'interpolation'
    # m_input = base_dir + '182929.jpg'
    # m_target = base_dir + '000193.jpg'
    # m_target = base_dir + '035460.jpg'
    # m_targets = None
    # interpolations_num = '5'
    # m_size = [3, 3]

    # # Matrix Interpolation in Attribute Subspace
    # mode = 'matrix'
    # m_input = base_dir + '182929.jpg'
    # m_target = None
    # m_targets = [base_dir + '035460.jpg', base_dir + '035451.jpg', base_dir + '035463.jpg', base_dir + '035474.jpg']
    # interpolations_num = '2'
    # m_size = [5, 5]

    parser = argparse.ArgumentParser(description='test', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=640, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=480, help='Height of the frames in the video stream.')

    parser.add_argument(
        '-m', '--mode',
        default=mode,
        type=str,
        choices=['swap', 'interpolation', 'matrix'],
        help='Specify mode.'
    )
    parser.add_argument(
        '-t', '--target',
        default=m_target,
        metavar='target image with attributes',
        type=str,
        help='Specify target image name.'
    )
    parser.add_argument(
        '--targets',
        default=m_targets,
        nargs=4,
        type=str,
        help='Specify target image name.'
    )
    parser.add_argument(
        '--model_dir',
        default=os.path.join(output_model_path, 'train_log/model/'),
        type=str,
        help='Specify model_dir. \ndefault: %(default)s.'
    )
    parser.add_argument(
        '-n', '--num',
        default=interpolations_num,
        type=int,
        help='Specify number of interpolations.'
    )
    parser.add_argument(
        '-s', '--size',
        nargs=2,
        default=m_size,
        type=int,
        help='Specify number of interpolations.'
    )
    parser.add_argument(
        '-g', '--gpu',
        default='0',
        type=str,
        help='Specify GPU id. \ndefault: %(default)s. \nUse comma to seperate several ids, for example: 0,1'
    )

    args = parser.parse_args()

    # GeneGAN = Model(is_train=False, nhwc=[1, 64, 64, 3])
    #
    # saver = tf.train.Saver()
    # ckpt = tf.train.get_checkpoint_state(model_dir)

    # input_q = Queue(5)  # fps is better if queue is higher but then more lags
    # output_q = Queue()
    # for i in range(1):
    #     t = Thread(target=worker, args=(input_q, output_q))
    #     t.daemon = True
    #     t.start()
    #
    # video_capture = WebcamVideoStream(src=args.video_source,
    #                                   width=64,
    #                                   height=64).start()
    # fps = FPS().start()

    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0

    GeneGAN = Model(is_train=False, nhwc=[1, 64, 64, 3])
    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=GeneGAN.width,
                                      height=GeneGAN.height).start()
    start_time = time.time()

    while True:
        # Capture frame-by-frame
        frame = video_capture.read()
        src_img = np.expand_dims(misc.imresize(frame, (GeneGAN.width, GeneGAN.height)), axis=0)
        att_img = np.expand_dims(misc.imresize(misc.imread(args.target), (GeneGAN.width, GeneGAN.height)), axis=0)

        if (frame_count % frame_interval) == 0:
            # GeneGAN = Model(is_train=False, nhwc=[1, 64, 64, 3])
            if args.mode == 'swap':
                src_output, att_output = swap_attribute(src_img, att_img, args.model_dir, GeneGAN, args.gpu)
            elif args.mode == 'interpolation':
                interpolation(src_img, att_img, args.num, args.model_dir, GeneGAN, args.gpu)
            elif args.mode == 'matrix':
                att_imgs = np.array(
                    [misc.imresize(misc.imread(img), (GeneGAN.height, GeneGAN.width)) for img in args.targets])
                interpolation_matrix(src_img, att_imgs, args.size, args.model_dir, GeneGAN, args.gpu)
            else:
                raise NotImplementationError()
            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        frame_count += 1
        save_path1 = os.path.join('output', model_path_name, 'swap_out1.jpg')
        cv2.imwrite(os.path.join('output', model_path_name, 'ori_img.jpg'), frame)
        image = cv2.imread(save_path1)
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Video', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()