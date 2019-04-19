# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/4/3 10:24
# @Author  : ludahai
# @FileName: DataAugmentation.py
# @Software: PyCharm


from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import random
import numpy as np
from scipy import misc
import os
from tqdm import tqdm


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def get_dataset(path):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp)
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def DataGenerator(input_dir, max_num_each_class, random_order=True):
    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=10,
                                 width_shift_range=0.1, height_shift_range=0.1, channel_shift_range=10,
                                 brightness_range=[0.1, 2], shear_range=0.2, horizontal_flip=True,
                                 fill_mode='nearest', data_format='channels_last')

    dataset = get_dataset(input_dir)

    sub_name = 0
    for cls in tqdm(dataset):
        output_class_dir = os.path.join(input_dir, cls.name)
        already_exit_num = len(os.listdir(output_class_dir))
        if random_order:
            random.shuffle(cls.image_paths)
        for i in range(max_num_each_class - already_exit_num):
            image_path = random.choice(cls.image_paths)
            filename_base, file_extension = os.path.splitext(image_path)
            base_name = os.path.basename(filename_base)
            # 读取图像
            img = image.load_img(image_path)
            # 将图片转为数组
            x = image.img_to_array(img)
            # 扩充一个维度
            x = np.expand_dims(x, axis=0)
            # 生成图片
            x_batch = next(datagen.flow(x, shuffle=True, batch_size=1))
            output_file_path = os.path.join(output_class_dir,
                                            "{}_{}{}".format(base_name, str(sub_name), file_extension))
            # 保存图像
            misc.imsave(output_file_path, x_batch[0])
            sub_name += 1


if __name__ == "__main__":
    ROOT_PATH = "D:/develop/workstations/resource/"
    INPUT_DIR = ROOT_PATH + 'datasets/face_db/emb_face_182'
    # INPUT_DIR = ROOT_PATH + 'datasets/face_db/test_db/test_emb_face_160'
    max_num_each_person = 40
    DataGenerator(INPUT_DIR, max_num_each_person)
