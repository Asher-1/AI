# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 20:12:58 2018

@author: huang
"""

"""
1.	课程内容
    1.	了解人脸校正和编码的基本过程.
    2.	K-近邻算法.
    3.	学习使用mglearn, skimage, matplotlib, numpy等Python库.
 """

# 六位同学,身高和体重的数据.
# X, 身高, 体重.
# y, 性别.

# 1. 导入模块
import numpy as np
import mglearn
import matplotlib.pyplot as plt

# 2. 生成身高和体重的数据集
# 用numpy数组来表示X, y.
X = np.array([[1.5, 48], [1.56, 51], [1.6, 50], [1.65, 65], [1.70, 66], [1.8, 72],
              [1.71, 65], [1.72, 63]])
y = np.array([1, 1, 1, 0, 0, 0, 1, 0])

# 3. 对数据集作散点图.
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

plt.legend(["Men", "Women"])
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()

# k-近邻算法.
# k: 邻居的数目.
# 增大k的值可以减小误差.
# k也不能太大!否则也不准确.
# [1,0,1]
from sklearn.neighbors import NearestNeighbors

s = [[0, 0, 0], [0, 0.5, 0], [1, 1, 5], [10, 10, 0]]

neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(s)
X = [[1, 0, 1], [1, 9, 4]]
print(neigh.kneighbors(X))
