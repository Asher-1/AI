#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":
    # pandas读入
    data = pd.read_csv('..\\Advertising.csv')    # TV、Radio、Newspaper、Sales
    print(data)
    x = data[['TV', 'Radio', 'Newspaper']]
    # x = data[['TV', 'Radio']]
    y = data['Sales']
    print(x)
    print(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.8)
    # model = Lasso()
    model = Ridge()
    alpha_can = np.logspace(-3, 2, 10)
    np.set_printoptions(suppress=True)
    print('alpha_can = ', alpha_can)
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    lasso_model.fit(x_train, y_train)
    print('超参数：\n', lasso_model.best_params_)

    order = y_test.argsort(axis=0)
    y_test = y_test.values[order]
    x_test = x_test.values[order, :]
    y_hat = lasso_model.predict(x_test)
    print(lasso_model.score(x_test, y_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print(mse, rmse)

    t = np.arange(len(x_test))
    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(t, y_test, 'r-', linewidth=2, label='真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='预测数据')
    plt.title('线性回归预测销量', fontsize=18)
    plt.legend(loc='upper left')
    plt.grid(b=True, ls=':')
    plt.show()
