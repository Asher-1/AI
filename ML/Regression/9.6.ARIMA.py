# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
from statsmodels.tools.sm_exceptions import HessianInversionWarning


def extend(a, b):
    return 1.05*a-0.05*b, 1.05*b-0.05*a


def date_parser(date):
    return pd.datetime.strptime(date, '%Y-%m')


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=HessianInversionWarning)
    pd.set_option('display.width', 100)
    np.set_printoptions(linewidth=100, suppress=True)

    data = pd.read_csv('..\\AirPassengers.csv', header=0, parse_dates=['Month'], date_parser=date_parser, index_col=['Month'])
    data.rename(columns={'#Passengers': 'Passengers'}, inplace=True)
    print(data.dtypes)
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    x = data['Passengers'].astype(np.float)
    x = np.log(x)
    print(x.head(10))

    show = 'prime'   # 'diff', 'ma', 'prime'
    d = 1
    diff = x - x.shift(periods=d)
    ma = x.rolling(window=12).mean()
    xma = x - ma

    p = 2
    q = 2
    model = ARIMA(endog=x, order=(p, d, q))     # 自回归函数p,差分d,移动平均数q
    arima = model.fit(disp=-1)                  # disp<0:不输出过程
    prediction = arima.fittedvalues
    print(type(prediction))
    y = prediction.cumsum() + x[0]
    mse = ((x - y)**2).mean()
    rmse = np.sqrt(mse)

    plt.figure(facecolor='w')
    if show == 'diff':
        plt.plot(x, 'r-', lw=2, label='原始数据')
        plt.plot(diff, 'g-', lw=2, label='%d阶差分' % d)
        #plt.plot(prediction, 'r-', lw=2, label=u'预测数据')
        title = '乘客人数变化曲线 - 取对数'
    elif show == 'ma':
        #plt.plot(x, 'r-', lw=2, label=u'原始数据')
        #plt.plot(ma, 'g-', lw=2, label=u'滑动平均数据')
        plt.plot(xma, 'g-', lw=2, label='ln原始数据 - ln滑动平均数据')
        plt.plot(prediction, 'r-', lw=2, label='预测数据')
        title = '滑动平均值与MA预测值'
    else:
        plt.plot(x, 'r-', lw=2, label='原始数据')
        plt.plot(y, 'g-', lw=2, label='预测数据')
        title = '对数乘客人数与预测值(AR=%d, d=%d, MA=%d)：RMSE=%.4f' % (p, d, q, rmse)
    plt.legend(loc='upper left')
    plt.grid(b=True, ls=':')
    plt.title(title, fontsize=16)
    plt.tight_layout(2)
    # plt.savefig('%s.png' % title)
    plt.show()
