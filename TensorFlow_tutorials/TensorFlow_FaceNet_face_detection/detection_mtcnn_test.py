# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import cv2
from src import detect_face
import os

# face detection parameters
minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor

root_path = "D:/develop/workstations/GitHub/Datasets/facenet-detection/"

# facenet embedding parameters
model_name = 'model-20170512-110547.ckpt-250000'
# model_name = 'model.ckpt-500000'
check_path = root_path + 'models/20170512-110547/'
# check_path = './model_check_point/'
model_dir = check_path + model_name  # "Directory containing the graph definition and checkpoint files.")
image_size = 96  # "Image size (height, width) in pixels."
pool_type = 'MAX'  # "The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn = False  # "Enables Local Response Normalization after the first layers of the inception network."
seed = 42,  # "Random seed."
batch_size = None  # "Number of images to process in a batch."

# 待测试文件夹
test_image_dir = root_path + 'data/images/'


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


# 生成脸部方框和脸部特征点
def create_boxes_and_features(bounding_boxes, image, points):
    num = -1
    for face_position in bounding_boxes:
        num += 1
        face_position = face_position.astype(int)

        # draw face
        cv2.rectangle(image, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)

        # draw feature points
        cv2.circle(image, (points[0][num], points[5][num]), 2, (0, 255, 0), -1)
        cv2.circle(image, (points[1][num], points[6][num]), 2, (0, 255, 0), -1)
        cv2.circle(image, (points[2][num], points[7][num]), 2, (0, 255, 0), -1)
        cv2.circle(image, (points[3][num], points[8][num]), 2, (0, 255, 0), -1)
        cv2.circle(image, (points[4][num], points[9][num]), 2, (0, 255, 0), -1)


def main():
    # restore mtcnn model
    print('Creating networks and loading parameters')
    gpu_memory_fraction = 0.7
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, check_path)

    # 遍历目录
    for root, dirs, files in os.walk(test_image_dir):
        for file in files:
            # 获取待验证图片
            img_path = os.path.join(os.path.dirname(__file__), root, file)
            image = cv2.imread(img_path)
            # 预处理
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if gray.ndim == 2:
                img = to_rgb(gray)
                # 统一尺寸
                img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)
            # 统一尺寸
            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_CUBIC)

            # 获取脸部方框和脸部特征点
            bounding_boxes, points = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]  # number of faces

            # 生成脸部方框和脸部特征点
            create_boxes_and_features(bounding_boxes, image, points)

            # show result
            cv2.imshow(file, image)
            print('\n图片 %s 中一共找到 %d 张人脸' % (file, nrof_faces))

    cv2.waitKey(0)


if __name__ == '__main__':
    main()