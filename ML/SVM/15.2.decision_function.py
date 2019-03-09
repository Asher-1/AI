#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
from scipy import stats
from sklearn.metrics import accuracy_score
import matplotlib as mpl
import matplotlib.pyplot as plt


def extend(a, b, r=0.01):
    return a * (1 + r) - b * r, -a * r + b * (1 + r)


if __name__ == "__main__":
    np.random.seed(0)
    N = 200
    x = np.empty((4*N, 2))
    means = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
    sigmas = [np.eye(2), 2*np.eye(2), np.diag((1,2)), np.array(((3, 2), (2, 3)))]
    for i in range(4):
        mn = stats.multivariate_normal(means[i], sigmas[i]*0.1)
        x[i*N:(i+1)*N, :] = mn.rvs(N)
    a = np.array((0,1,2,3)).reshape((-1, 1))
    y = np.tile(a, N).flatten()
    clf = svm.SVC(C=1, kernel='rbf', gamma=1, decision_function_shape='ovr')
    # clf = svm.SVC(C=1, kernel='linear', decision_function_shape='ovr')
    clf.fit(x, y)
    y_hat = clf.predict(x)
    acc = accuracy_score(y, y_hat)
    np.set_printoptions(suppress=True)
    print('预测正确的样本个数：%d，正确率：%.2f%%' % (round(acc*4*N), 100*acc))
    # decision_function
    print(clf.decision_function(x))
    print(y_hat)

    x1_min, x2_min = np.min(x, axis=0)
    x1_max, x2_max = np.max(x, axis=0)
    x1_min, x1_max = extend(x1_min, x1_max)
    x2_min, x2_max = extend(x2_min, x2_max)
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
    x_test = np.stack((x1.flat, x2.flat), axis=1)
    y_test = clf.predict(x_test)
    y_test = y_test.reshape(x1.shape)
    cm_light = mpl.colors.ListedColormap(['#FF8080', '#80FF80', '#8080FF', '#F0F080'])
    cm_dark = mpl.colors.ListedColormap(['r', 'g', 'b', 'y'])
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_test, cmap=cm_light)
    plt.contour(x1, x2, y_test, levels=(0,1,2), colors='k', linestyles='--')
    plt.scatter(x[:, 0], x[:, 1], s=20, c=y, cmap=cm_dark, edgecolors='k', alpha=0.7)
    plt.xlabel('$X_1$', fontsize=11)
    plt.ylabel('$X_2$', fontsize=11)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.grid(b=True)
    plt.tight_layout(pad=2.5)
    plt.title('SVM多分类方法：One/One or One/Other', fontsize=14)
    plt.show()
