# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

iris_feature = '花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'


def expand(a, b, rate=0.05):
    d = (b - a) * rate
    return a-d, b+d


if __name__ == '__main__':
    path = '..\\..\\9.Regression\\iris.data'
    data = pd.read_csv(path, header=None)
    x_prime = data[np.arange(4)]
    y = pd.Categorical(data[4]).codes

    n_components = 3
    feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    plt.figure(figsize=(8, 6), facecolor='w')
    for k, pair in enumerate(feature_pairs, start=1):
        x = x_prime[pair]
        m = np.array([np.mean(x[y == i], axis=0) for i in range(3)])  # 均值的实际值
        print('实际均值 = \n', m)

        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
        gmm.fit(x)
        print('预测均值 = \n', gmm.means_)
        print('预测方差 = \n', gmm.covariances_)
        y_hat = gmm.predict(x)
        order = pairwise_distances_argmin(m, gmm.means_, axis=1, metric='euclidean')
        print('顺序：\t', order)

        n_sample = y.size
        n_types = 3
        change = np.empty((n_types, n_sample), dtype=np.bool)
        for i in range(n_types):
            change[i] = y_hat == order[i]
        for i in range(n_types):
            y_hat[change[i]] = i
        acc = '准确率：%.2f%%' % (100*np.mean(y_hat == y))
        print(acc)

        cm_light = mpl.colors.ListedColormap(['#FF8080', '#77E0A0', '#A0A0FF'])
        cm_dark = mpl.colors.ListedColormap(['r', 'g', '#6060FF'])
        x1_min, x2_min = x.min()
        x1_max, x2_max = x.max()
        x1_min, x1_max = expand(x1_min, x1_max)
        x2_min, x2_max = expand(x2_min, x2_max)
        x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
        grid_test = np.stack((x1.flat, x2.flat), axis=1)
        grid_hat = gmm.predict(grid_test)

        change = np.empty((n_types, grid_hat.size), dtype=np.bool)
        for i in range(n_types):
            change[i] = grid_hat == order[i]
        for i in range(n_types):
            grid_hat[change[i]] = i

        grid_hat = grid_hat.reshape(x1.shape)
        plt.subplot(2, 3, k)
        plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
        plt.scatter(x[pair[0]], x[pair[1]], s=20, c=y, marker='o', cmap=cm_dark, edgecolors='k')
        xx = 0.95 * x1_min + 0.05 * x1_max
        yy = 0.1 * x2_min + 0.9 * x2_max
        plt.text(xx, yy, acc, fontsize=10)
        plt.xlim((x1_min, x1_max))
        plt.ylim((x2_min, x2_max))
        plt.xlabel(iris_feature[pair[0]], fontsize=11)
        plt.ylabel(iris_feature[pair[1]], fontsize=11)
        plt.grid(b=True, ls=':', color='#606060')
    plt.suptitle('EM算法无监督分类鸢尾花数据', fontsize=14)
    plt.tight_layout(1, rect=(0, 0, 1, 0.95))
    plt.show()
