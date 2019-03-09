# /usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import scipy as sp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import os
from sklearn.externals import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == '__main__':
    pd.set_option('display.width', 400)
    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_columns', 70)
    column_names = 'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', \
                   'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    model_file = 'adult.pkl'
    if os.path.exists(model_file):
        model = joblib.load(model_file)
    else:
        show_result = True
        print('读入数据')
        data = pd.read_csv('adult.data', header=None, names=column_names)
        # print data
        for name in data.columns:
            data[name] = pd.Categorical(data[name]).codes
        x = data[data.columns[:-1]]
        y = data[data.columns[-1]]
        # print x.describe()
        # print y.describe()
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, random_state=0)
        print(y_train.mean())
        print(y_valid.mean())
        base_estimator = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=5)
        model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, learning_rate=0.1)
        model.fit(x_train, y_train)
        joblib.dump(model, model_file)
        if show_result:
            y_train_pred = model.predict(x_train)
            print('训练集准确率：', accuracy_score(y_train, y_train_pred))
            print('\t训练集查准率：', precision_score(y_train, y_train_pred))
            print('\t训练集召回率：', recall_score(y_train, y_train_pred))
            print('\t训练集F1：', f1_score(y_train, y_train_pred))

            y_valid_pred = model.predict(x_valid)
            print('验证集准确率：', accuracy_score(y_valid, y_valid_pred))
            print('\t验证集查准率：', precision_score(y_valid, y_valid_pred))
            print('\t验证集召回率：', recall_score(y_valid, y_valid_pred))
            print('\t验证集F1：', f1_score(y_valid, y_valid_pred))

    data_test = pd.read_csv('adult.test', header=None, skiprows=1, names=column_names)
    for name in data_test.columns:
        data_test[name] = pd.Categorical(data_test[name]).codes
    x_test = data_test[data_test.columns[:-1]]
    y_test = data_test[data_test.columns[-1]]
    y_test_pred = model.predict(x_test)
    print('测试集准确率：', accuracy_score(y_test, y_test_pred))
    print('\t测试集查准率：', precision_score(y_test, y_test_pred))
    print('\t测试集召回率：', recall_score(y_test, y_test_pred))
    print('\t测试集F1：', f1_score(y_test, y_test_pred))

    y_test_proba = model.predict_proba(x_test)
    # print y_test_proba
    y_test_proba = y_test_proba[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_proba)
    auc = metrics.auc(fpr, tpr)
    print('AUC = ', auc)
    # 或直接调用roc_auc_score
    # print 'AUC = ', metrics.roc_auc_score(y_test, y_test_proba)

    mpl.rcParams['font.sans-serif'] = 'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(facecolor='w')
    plt.plot(fpr, tpr, 'r-', lw=2, alpha=0.8, label='AUC=%.3f' % auc)
    plt.plot((0, 1), (0, 1), c='b', lw=1.5, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.grid(b=True)
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=14)
    plt.title('Adult数据的ROC曲线和AUC值', fontsize=17)
    plt.show()
