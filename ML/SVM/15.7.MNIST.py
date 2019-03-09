#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn import svm
import matplotlib.colors
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from time import time
from pprint import pprint


def save_image(im, i):
    im = 255 - im.values.reshape(28, 28)
    a = im.astype(np.uint8)
    output_path = '.\\HandWritten'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    Image.fromarray(a).save(output_path + ('\\%d.png' % i))


def save_result(model):
    data_test_pred = model.predict(data_test)
    data_test['Label'] = data_test_pred
    data_test.to_csv('Prediction.csv', header=True, index=True, columns=['Label'])


if __name__ == "__main__":
    classifier_type = 'RF'

    print('载入训练数据...')
    t = time()
    data = pd.read_csv('MNIST.train.csv', header=0, dtype=np.int)
    print('载入完成，耗时%f秒' % (time() - t))
    x, y = data.iloc[:, 1:], data['label']
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=1)
    print(x.shape, x_valid.shape)
    print('图片个数：%d，图片像素数目：%d' % x.shape)

    print('载入测试数据...')
    t = time()
    data_test = pd.read_csv('MNIST.test.csv', header=0, dtype=np.int)
    print('载入完成，耗时%f秒' % (time() - t))

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15, 9), facecolor='w')
    for index in range(16):
        image = x.iloc[index, :]
        plt.subplot(4, 8, index + 1)
        plt.imshow(image.values.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('训练图片: %i' % y[index])
    for index in range(16):
        image = data_test.iloc[index, :]
        plt.subplot(4, 8, index + 17)
        plt.imshow(image.values.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
        save_image(image.copy(), index)
        plt.title('测试图片')
    plt.tight_layout(2)
    plt.show()

    if classifier_type == 'SVM':
        model = svm.SVC(C=1000, kernel='rbf', gamma=1e-10)
        print('SVM开始训练...')
    else:
        model = RandomForestClassifier(100, criterion='gini', min_samples_split=2, min_impurity_decrease=1e-10)
        print('随机森林开始训练...')
    t = time()
    model.fit(x_train, y_train)
    t = time() - t
    print('%s训练结束，耗时%d分钟%.3f秒' % (classifier_type, int(t/60), t - 60*int(t/60)))
    t = time()
    y_train_pred = model.predict(x_train)
    t = time() - t
    print('%s训练集准确率：%.3f%%，耗时%d分钟%.3f秒' % (classifier_type, accuracy_score(y_train, y_train_pred)*100, int(t/60), t - 60*int(t/60)))
    t = time()
    y_valid_pred = model.predict(x_valid)
    t = time() - t
    print('%s测试集准确率：%.3f%%，耗时%d分钟%.3f秒' % (classifier_type, accuracy_score(y_valid, y_valid_pred)*100, int(t/60), t - 60*int(t/60)))
    save_result(model)

    err = (y_valid != y_valid_pred)
    err_images = x_valid[err]
    err_y_hat = y_valid_pred[err]
    err_y = y_valid[err]
    print(err_y_hat)
    print(err_y)
    plt.figure(figsize=(10, 8), facecolor='w')
    for index in range(12):
        image = err_images.iloc[index, :]
        plt.subplot(3, 4, index + 1)
        plt.imshow(image.values.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('错分为：%i，真实值：%i' % (err_y_hat[index], err_y.values[index]), fontsize=12)
    plt.suptitle('数字图片手写体识别：分类器%s' % classifier_type, fontsize=15)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
