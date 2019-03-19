# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 20:04:25 2018

@author: huang
"""

"""
1. 课程内容
1.1 人脸探测和人脸标记.
1.2 人脸探测与人脸标记的相关算法.
1.3 学习sklearn, dlb等库函数的安装和使用.



2. 

#人脸探测
from skimage import io
import dlib

# 1.获取图片,转换成数组
file_name = "li_00.jpg"
image = io.imread(file_name)

#3 建立人脸探测器
detector = dlib.get_frontal_face_detector()

#3 运行在图片数据上
detected_faces= detector(image, 1)
print("发现{}张人脸, 于{}图片.".format(len(detected_faces), file_name))
# 4. 建立窗口
win = dlib.image_window()
win.set_image(image)

#5 . 对每张人脸,操作 
# for 循环
for box  in detected_faces:
    win.add_overlay(box)
    dlib.hit_enter_to_continue()
    """

# 人脸标识
from skimage import io
import dlib

FACE_MODEL = "D:/develop/workstations/GitHub/Datasets/DL/Images/face_data/"

# 1.获取图片,转换成数组
file_name = "li_00.jpg"
image = io.imread(file_name)

# 3 建立人脸探测器
detector = dlib.get_frontal_face_detector()

# 4 运行在图片数据上
detected_faces = detector(image, 1)
print("发现{}张人脸, 于{}图片.".format(len(detected_faces), file_name))
# 5.人脸"68点-预测" 模型
model = FACE_MODEL + "shape_predictor_68_face_landmarks.dat"
# 提取特征
predictor = dlib.shape_predictor(model)

# 6. 建立窗口
win = dlib.image_window()
win.set_image(image)

# 5 . 对每张人脸,操作
# for 循环, 实现人脸探测和标识
# enumerate() 返回迭代对象的索引和对应的值
for i, box in enumerate(detected_faces):
    win.add_overlay(box)
    print("第{}张人脸的位置:{},右边位置:{}.".format(i, box.left(), box.right()))
    landmarks = predictor(image, box)
    win.add_overlay(landmarks)
    dlib.hit_enter_to_continue()
