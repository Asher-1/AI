# -*-coding: utf-8 -*-
"""
    @Project: faceRecognition
    @File   : create_dataset.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2018-12-07 11:31:09
"""
import numpy as np
from utils import image_processing, file_processing
import face_recognition
from create_faces_images import create_face
import os
import math
from tqdm import tqdm


def create_embedding(model_path, images_dir, out_emb_path, out_filename, postfix='jpg'):
    '''
    产生embedding数据库，保存在out_data_path中，这些embedding其实就是人脸的特征
    :param model_path:
    :param emb_face_dir:
    :param out_emb_path:
    :param out_filename:
    :return:
    '''
    face_net = face_recognition.facenetEmbedding(model_path)
    image_list = file_processing.get_files_list(images_dir, postfix=postfix)
    face_images, names_list = create_face(image_list)
    if face_images is None:
        print("dataset is empty ...")
        return

    # Run forward pass to calculate embeddings
    print('Calculating features for images')
    nrof_images = len(face_images)
    assert nrof_images == len(names_list)
    print('Number of images: %d' % nrof_images)
    nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / batch_size))
    emb_array = np.zeros((nrof_images, face_net.embedding_size))
    for i in tqdm(range(nrof_batches_per_epoch)):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, nrof_images)
        faces_batch = face_images[start_index:end_index]
        images = np.stack(faces_batch)
        emb_array[start_index:end_index, :] = face_net.get_embedding(images)

    path_name = os.path.dirname(out_emb_path)
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    np.save(out_emb_path, emb_array)

    # 可以选择保存image_list或者names_list作为人脸的标签
    # 测试时建议保存image_list，这样方便知道被检测人脸与哪一张图片相似
    file_processing.write_data(out_filename, names_list, model='w')


if __name__ == '__main__':
    ROOT_PATH = "D:/develop/workstations/GitHub/Datasets/facenet-detection/"
    images_dir = ROOT_PATH + 'dataset/images'
    model_path = ROOT_PATH + "models/20180408-102900"
    out_emb_path = ROOT_PATH + 'dataset/emb/faceEmbedding.npy'
    out_filename = ROOT_PATH + 'dataset/emb/name.txt'
    batch_size = 90
    create_embedding(model_path, images_dir, out_emb_path, out_filename)
