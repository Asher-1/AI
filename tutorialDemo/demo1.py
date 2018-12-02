# -*- coding:utf-8 -*-
# 利用Python的机器学习库sklearn: SkLearnExample.py
from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()


iris = datasets.load_iris()


print iris

knn.fit(iris.data, iris.target)

predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])

print predictedLabel