# -*-coding:utf-8-*-

import numbers
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from numpy import interp
from sklearn import metrics
from itertools import cycle


if __name__ == '__main__':
    np.random.seed(0)
    pd.set_option('display.width', 300)
    np.set_printoptions(suppress=True)
    data = pd.read_csv('iris.data', header=None)
    iris_types = data[4].unique()
    for i, iris_type in enumerate(iris_types):
        data.set_value(data[4] == iris_type, 4, i)
    x = data.iloc[:, :2]
    n, features = x.shape
    print x
    y = data.iloc[:, -1].astype(np.int)
    c_number = np.unique(y).size
    x, x_test, y, y_test = train_test_split(x, y, train_size=0.6, random_state=0)
    y_one_hot = label_binarize(y_test, classes=np.arange(c_number))
    alpha = np.logspace(-2, 2, 20)
    models = [
        ['KNN', KNeighborsClassifier(n_neighbors=7)],
        ['LogisticRegression', LogisticRegressionCV(Cs=alpha, penalty='l2', cv=3)],
        ['SVM(Linear)', GridSearchCV(SVC(kernel='linear', decision_function_shape='ovr'), param_grid={'C': alpha})],
        ['SVM(RBF)', GridSearchCV(SVC(kernel='rbf', decision_function_shape='ovr'), param_grid={'C': alpha, 'gamma': alpha})]]
    colors = cycle('gmcr')
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(7, 6), facecolor='w')
    for (name, model), color in zip(models, colors):
        model.fit(x, y)
        if hasattr(model, 'C_'):
            print(model.C_)
        if hasattr(model, 'best_params_'):
            print(model.best_params_)
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(x_test)
        else:
            y_score = model.decision_function(x_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_one_hot.ravel(), y_score.ravel())
        auc = metrics.auc(fpr, tpr)
        print(auc)
        plt.plot(fpr, tpr, c=color, lw=2, alpha=0.7, label=u'%s，AUC=%.3f' % (name, auc))
    plt.plot((0, 1), (0, 1), c='#808080', lw=2, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    # plt.legend(loc='lower right', fancybox=True, framealpha=0.8, edgecolor='#303030', fontsize=12)
    plt.title(u'鸢尾花数据不同分类器的ROC和AUC', fontsize=17)
    plt.show()
