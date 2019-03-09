# coding:utf-8

import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings


if __name__ == '__main__':
    np.random.seed(0)
    np.set_printoptions(linewidth=300)
    N = 9
    # x = np.linspace(0, 6, N) + np.random.randn(N)
    # x = np.sort(x)
    x = np.arange(-3, 3, 1)
    y = 2*x - 5 + np.random.randn(len(x))*0.1
    print x
    print y
    x.shape = -1, 1
    y.shape = -1, 1
    model = LinearRegression(fit_intercept=True)
    model.fit(x, y)
    print model.coef_
    print model.intercept_
