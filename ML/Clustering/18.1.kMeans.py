# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
import sklearn.datasets as ds
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_mutual_info_score,\
    adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans


def expand(a, b):
    d = (b - a) * 0.1
    return a-d, b+d


if __name__ == "__main__":
    N = 400
    centers = 4
    data, y = ds.make_blobs(N, n_features=2, centers=centers, random_state=2)
    data2, y2 = ds.make_blobs(N, n_features=2, centers=centers, cluster_std=(1,2.5,0.5,2), random_state=2)
    data3 = np.vstack((data[y == 0][:], data[y == 1][:50], data[y == 2][:20], data[y == 3][:5]))
    y3 = np.array([0] * 100 + [1] * 50 + [2] * 20 + [3] * 5)
    m = np.array(((1, 1), (1, 3)))
    data_r = data.dot(m)

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    cm = matplotlib.colors.ListedColormap(list('rgbm'))
    data_list = data, data, data_r, data_r, data2, data2, data3, data3
    y_list = y, y, y, y, y2, y2, y3, y3
    titles = '原始数据', 'KMeans++聚类', '旋转后数据', '旋转后KMeans++聚类',\
             '方差不相等数据', '方差不相等KMeans++聚类', '数量不相等数据', '数量不相等KMeans++聚类'

    model = KMeans(n_clusters=4, init='k-means++', n_init=5)
    plt.figure(figsize=(8, 9), facecolor='w')
    for i, (x, y, title) in enumerate(zip(data_list, y_list, titles), start=1):
        plt.subplot(4, 2, i)
        plt.title(title)
        if i % 2 == 1:
            y_pred = y
        else:
            y_pred = model.fit_predict(x)
        print(i)
        print('Homogeneity：', homogeneity_score(y, y_pred))
        print('completeness：', completeness_score(y, y_pred))
        print('V measure：', v_measure_score(y, y_pred))
        print('AMI：', adjusted_mutual_info_score(y, y_pred))
        print('ARI：', adjusted_rand_score(y, y_pred))
        print('Silhouette：', silhouette_score(x, y_pred), '\n')
        plt.scatter(x[:, 0], x[:, 1], c=y_pred, s=30, cmap=cm, edgecolors='none')
        x1_min, x2_min = np.min(x, axis=0)
        x1_max, x2_max = np.max(x, axis=0)
        x1_min, x1_max = expand(x1_min, x1_max)
        x2_min, x2_max = expand(x2_min, x2_max)
        plt.xlim((x1_min, x1_max))
        plt.ylim((x2_min, x2_max))
        plt.grid(b=True, ls=':')
    plt.tight_layout(2, rect=(0, 0, 1, 0.97))
    plt.suptitle('数据分布对KMeans聚类的影响', fontsize=18)
    plt.show()
