#!/usr/bin/python
# -*- coding:utf-8 -*-

import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pprint import pprint


if __name__ == "__main__":
    path = '..\\Advertising.csv'
    # # 手写读取数据
    # f = file(path)
    # x = []
    # y = []
    # for i, d in enumerate(f):
    #     if i == 0:
    #         continue
    #     d = d.strip()
    #     if not d:
    #         continue
    #     d = map(float, d.split(','))
    #     x.append(d[1:-1])
    #     y.append(d[-1])
    # pprint(x)
    # pprint(y)
    # x = np.array(x)
    # y = np.array(y)

    # Python自带库
    # f = file(path, 'r')
    # print f
    # d = csv.reader(f)
    # for line in d:
    #     print line
    # f.close()

    # # numpy读入
    # p = np.loadtxt(path, delimiter=',', skiprows=1)
    # print p
    # print '\n\n===============\n\n'

    # pandas读入
    data = pd.read_csv(path)    # TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio', 'Newspaper']]
    # x = data[['TV', 'Radio']]
    y = data['Sales']
    print(x)
    print(y)

    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 绘制1
    plt.figure(facecolor='w')
    plt.plot(data['TV'], y, 'ro', label='TV')
    plt.plot(data['Radio'], y, 'g^', label='Radio')
    plt.plot(data['Newspaper'], y, 'mv', label='Newspaer')
    plt.legend(loc='lower right')
    plt.xlabel('广告花费', fontsize=16)
    plt.ylabel('销售额', fontsize=16)
    plt.title('广告花费与销售额对比数据', fontsize=18)
    plt.grid(b=True, ls=':')
    plt.show()

    # 绘制2
    plt.figure(facecolor='w', figsize=(9, 10))
    plt.subplot(311)
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')
    plt.grid(b=True, ls=':')
    plt.subplot(312)
    plt.plot(data['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid(b=True, ls=':')
    plt.subplot(313)
    plt.plot(data['Newspaper'], y, 'b*')
    plt.title('Newspaper')
    plt.grid(b=True, ls=':')
    plt.tight_layout()
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    print(type(x_test))
    print(x_train.shape, y_train.shape)
    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    print(model)
    print(linreg.coef_, linreg.intercept_)

    order = y_test.argsort(axis=0)
    y_test = y_test.values[order]
    x_test = x_test.values[order, :]
    y_hat = linreg.predict(x_test)
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print('MSE = ', mse, end=' ')
    print('RMSE = ', rmse)
    print('R2 = ', linreg.score(x_train, y_train))
    print('R2 = ', linreg.score(x_test, y_test))

    plt.figure(facecolor='w')
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='预测数据')
    plt.legend(loc='upper left')
    plt.title('线性回归预测销量', fontsize=18)
    plt.grid(b=True, ls=':')
    plt.show()
