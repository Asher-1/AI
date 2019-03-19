#!usr/bin/env python
# -*- coding:utf-8 -*-

"""
@author: Asher
@file: face_detection.py
@time: 2019/03/18
"""

# 1. 导入库函数
from skimage import io, color
from skimage.feature import hog
import matplotlib.pyplot as plt

# 2. 导入图片
image = io.imread("eg.jpg")
image = color.rgb2gray(image)

# 3. 计算HOG
# hog() 返回值
# hog_image (可用于显示HOG图)

arr, hog_image = hog(image, visualise=True)

# 4. 作图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(image, cmap=plt.cm.gray)
ax2.imshow(hog_image, cmap=plt.cm.gray)
plt.show()
