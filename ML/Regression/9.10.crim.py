#!/usr/bin/python
# -*- encoding: utf-8

import numpy as np
import matplotlib as mpl
import pandas as pd
from sklearn.preprocessing import StandardScaler, minmax_scale
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, ElasticNetCV
import warnings
from sklearn.exceptions import ConvergenceWarning


if __name__ == "__main__":
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    data = pd.read_excel('crim.xlsx', sheetname='Sheet1', header=0)
    print 'data.head() = \n', data.head()
    columns = [c for c in data.columns]      # 列标题
    print data.columns
    data.sort_values(by=data.columns[1], inplace=True)
    data = data.values
    x = data[:, 2:].astype(np.float)
    y = data[:, 1].astype(np.int)
    columns = columns[2:]

    ss = StandardScaler()
    x = ss.fit_transform(x)

    # 增加一列全1
    t = np.ones(x.shape[0]).reshape((-1, 1))
    print t.shape, x.shape
    x = np.hstack((t, x))

    # model = ElasticNetCV(alphas=np.logspace(-3, 2, 50), l1_ratio=[.1, .5, .7, .9, .95, .99, 1], fit_intercept=False)
    model = LassoCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False)
    model.fit(x, y)
    y_hat = model.predict(x)
    y_hat[y_hat < 0] = 0
    print 'model.alpha = \t', model.alpha_
    # print 'model.l1_ratio = \t', model.l1_ratio_
    print 'model.coef_ = \n', model.coef_
    print 'model.predict(x) = \n', y_hat
    print 'Acture = \n', y
    print 'RMSE:\t', np.sqrt(np.mean((y_hat-y)**2))
    print 'R2:\t', model.score(x, y)
    for theta, col in zip(model.coef_[1:], columns):
        if theta > 0.01:
            print col, theta

    plt.figure(facecolor='w')
    t = np.arange(len(y))
    plt.plot(t, y_hat, 'go', label=u'预测值')
    plt.plot(t, y, 'r-', lw=2, label=u'实际值')
    plt.grid(b=True)
    plt.legend(loc='upper left')
    plt.title(u'北京市犯罪率与特征相关性回归分析', fontsize=18)
    plt.show()
