from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
from scipy import misc
import tensorflow as tf
import numpy as np
import os
from utils import file_processing, image_processing
import face_recognition
from tqdm import tqdm
import facenet
import math

THRESHOLD = 1
image_size = 160
margin = 32


def face_recognition_image(model_path, dataset_path, filename, image_path, result_path):
    # 加载数据库的数据
    dataset_emb, names_list = load_dataset(dataset_path, filename)
    # 初始化mtcnn人脸检测
    face_detect = face_recognition.Facedetection()
    # 初始化facenet
    face_net = face_recognition.facenetEmbedding(model_path)
    for path in tqdm(image_path):
        # image = image_processing.read_image(path)
        img_list = []
        try:
            img = misc.imread(os.path.expanduser(path), mode='RGB')
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(path, e)
            print(errorMessage)
        else:
            if img.ndim == 2:
                img = facenet.to_rgb(img)
            img = img[:, :, 0:3]
        # 获取 判断标识 bounding_box crop_image
        if img is None:
            continue
        bounding_boxes, _ = face_detect.detect_face(img)
        det = bounding_boxes[:, 0:4]
        nrof_faces = bounding_boxes.shape[0]
        bbs = []
        if nrof_faces > 0:
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces > 1:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                det_arr.append(np.squeeze(det))
            for i, det in enumerate(det_arr):
                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                bbs.append(bb)
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                img_list.append(facenet.prewhiten(scaled))
        else:
            print("cannot find any face in this ", path)
            continue
        if len(img_list) > 0:
            face_images = np.stack(img_list)
            pred_emb = face_net.get_embedding(face_images)
            pred_name = compare_embadding(pred_emb, dataset_emb, names_list)
            # 在图像上绘制人脸边框和识别的结果
            out_file_name = os.path.join(result_path, os.path.basename(path))
            bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # image_processing.cv_save_image_text(out_file_name, bgr_image, bbs, pred_name)
            image_processing.cv_show_image_text("face_recognition", bgr_image, bbs, pred_name)
            cv2.waitKey(0)


def load_dataset(dataset_path, filename):
    '''
    加载人脸数据库
    :param dataset_path: embedding.npy文件（faceEmbedding.npy）
    :param filename: labels文件路径路径（name.txt）
    :return:
    '''
    compare_emb = np.load(dataset_path)
    names_list = file_processing.read_data(filename)
    return compare_emb, names_list


def compare_embadding(pred_emb, dataset_emb, names_list):
    # 为bounding_box 匹配标签
    pred_num = len(pred_emb)
    dataset_num = len(dataset_emb)
    pred_name = []
    for i in range(pred_num):
        dist_list = []
        for j in range(dataset_num):
            # distance_metric
            # == 0 --> Euclidian distance
            # == 1 --> Distance based on cosine similarity
            dist = distance(pred_emb[i, :], dataset_emb[j, :], distance_metric=0)
            # dist = np.sqrt(np.sum(np.square(np.subtract(pred_emb[i, :], dataset_emb[j, :]))))
            dist_list.append(dist)
        min_value = min(dist_list)
        if min_value > THRESHOLD:
            pred_name.append('unknow')
        else:
            pred_name.append(names_list[dist_list.index(min_value)])
    return pred_name


def distance(embeddings1, embeddings2, distance_metric=0, axis=0):
    if distance_metric == 0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sqrt(np.sum(np.square(diff), axis))
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=axis)
        norm = np.linalg.norm(embeddings1, axis=axis) * np.linalg.norm(embeddings2, axis=axis)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric
    return dist


if __name__ == '__main__':
    ROOT_PATH = "D:/develop/workstations/GitHub/Datasets/facenet-detection/"
    model_path = ROOT_PATH + "models/20180408-102900"
    # model_path = 'D:/develop/workstations/GitHub/Datasets/DL/trained_outputs/facenet_output/models/20190403-193855'
    filename = ROOT_PATH + 'dataset/emb/name.txt'
    dataset_path = ROOT_PATH + 'dataset/emb/faceEmbedding.npy'
    result_path = ROOT_PATH + 'dataset/test_result/'

    filePath_list = []
    image_path = ROOT_PATH + 'dataset/test_images/1.jpg'
    filePath_list.append(image_path)
    face_recognition_image(model_path, dataset_path, filename, filePath_list, result_path)

    # image_path = ROOT_PATH + 'dataset/test_images/'
    # filePath_list = file_processing.get_files_list(image_path, postfix='jpg')
    # print("files nums:{}".format(len(filePath_list)))
    # face_recognition_image(model_path, dataset_path, filename, filePath_list, result_path)
