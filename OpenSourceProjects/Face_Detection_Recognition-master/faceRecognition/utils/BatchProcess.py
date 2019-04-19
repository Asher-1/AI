# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/4/2 10:56
# @Author  : ludahai
# @FileName: BatchProcess.py
# @Software: PyCharm

import image_processing, file_processing
import os
import shutil
from tqdm import tqdm


def classify_by_id(input_face_dir, out_face_dir, out_labels, postfix='jpg'):
    if not os.path.exists(out_face_dir):
        os.mkdir(out_face_dir)

    filePath_list = file_processing.get_files_list(input_face_dir, postfix=postfix)
    print("files nums:{}".format(len(filePath_list)))
    # 获取所有样本标签
    label_list = []
    labels = []
    num_processed = 0
    for filePath in tqdm(filePath_list):
        filename_base, file_extension = os.path.splitext(filePath)
        file_name = os.path.basename(filename_base)
        label = file_name.split("_")[0]
        labels.append(label)
        class_folder_name = out_face_dir + label + "/"
        if not os.path.exists(class_folder_name):
            os.mkdir(class_folder_name)

        output_filename_n = "{}{}{}".format(class_folder_name, file_name, file_extension)
        shutil.copy(filePath, output_filename_n)
        label_list.append((label, output_filename_n))
        num_processed += 1

    with open(out_labels, "w") as text_file:
        for path in label_list:
            text_file.write('%s %s\n' % (path[0], path[1]))
        print("labeles write sucessfully in :{}".format(out_labels))
    print("the number of class:{}".format(len(list(set(labels)))))
    print("the number of successfully processed images:{}".format(num_processed))


if __name__ == '__main__':
    ROOT_PATH = "D:/develop/workstations/resource/"

    # images_dir = ROOT_PATH + 'datasets/face_db/FaceFiles/'
    # out_face_dir = ROOT_PATH + 'datasets/face_db/LabeledFaces/'
    # out_labels = out_face_dir + 'labels.txt'

    images_dir = ROOT_PATH + 'datasets/face_db/test_db/test_images/'
    out_face_dir = ROOT_PATH + 'datasets/face_db/test_db/labeled_test/'
    out_labels = out_face_dir + 'labels.txt'

    classify_by_id(images_dir, out_face_dir, out_labels, 'jpg')
