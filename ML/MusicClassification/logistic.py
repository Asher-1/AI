# coding:utf-8

import numpy as np
import random
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import fft
from scipy.io import wavfile

"""
n = 40
# hstack使得十足拼接
# rvs是Random Variates随机变量的意思
# 在模拟X的时候使用了两个正态分布,分别制定各自的均值,方差,生成40个点
X = np.hstack((norm.rvs(loc=2, size=n, scale=2), norm.rvs(loc=8, size=n, scale=3)))
# zeros使得数据点生成40个0,ones使得数据点生成40个1
y = np.hstack((np.zeros(n),np.ones(n)))
# 创建一个 10 * 4 点（point）的图，并设置分辨率为 80
plt.figure(figsize=(10, 4),dpi=80)
# 设置横轴的上下限
plt.xlim((-5, 20))
# scatter散点图
plt.scatter(X, y, c=y)
plt.xlabel("feature value")
plt.ylabel("class")
plt.grid(True, linestyle='-', color='0.75')
plt.savefig("D:/workspace/scikit-learn/logistic_classify.png", bbox_inches="tight")
"""

"""
# linspace是在-5到15的区间内找10个数
xs=np.linspace(-5,15,10)

#---linear regression----------
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
# reshape重新把array变成了80行1列二维数组,符合机器学习多维线性回归格式
clf.fit(X.reshape(n * 2, 1), y)
def lin_model(clf, X):
    return clf.intercept_ + clf.coef_ * X

#---logistic regression--------
from sklearn.linear_model import LogisticRegression
logclf = LogisticRegression()
# reshape重新把array变成了80行1列二维数组,符合机器学习多维线性回归格式
logclf.fit(X.reshape(n * 2, 1), y)
def lr_model(clf, X):
    return 1.0 / (1.0 + np.exp(-(clf.intercept_ + clf.coef_ * X)))

#----plot---------------------------    
plt.figure(figsize=(10, 5))
# 创建一个一行两列子图的图像中第一个图
plt.subplot(1, 2, 1)
plt.scatter(X, y, c=y)
plt.plot(X, lin_model(clf, X),"o",color="orange")
plt.plot(xs, lin_model(clf, xs),"-",color="green")
plt.xlabel("feature value")
plt.ylabel("class")
plt.title("linear fit")
plt.grid(True, linestyle='-', color='0.75')
# 创建一个一行两列子图的图像中第二个图
plt.subplot(1, 2, 2)
plt.scatter(X, y, c=y)
plt.plot(X, lr_model(logclf, X).ravel(),"o",color="c")
plt.plot(xs, lr_model(logclf, xs).ravel(),"-",color="green")
plt.xlabel("feature value")
plt.ylabel("class")
plt.title("logistic fit")
plt.grid(True, linestyle='-', color='0.75')

plt.tight_layout(pad=0.4, w_pad=0, h_pad=1.0)     
plt.savefig("D:/workspace/scikit-learn/logistic_classify2.png", bbox_inches="tight")
"""

"""
使用logistic regression处理音乐数据,音乐数据训练样本的获得和使用快速傅里叶变换（FFT）预处理的方法需要事先准备好
1. 把训练集扩大到每类100个首歌而不是之前的10首歌,类别仍然是六类:jazz,classical,country, pop, rock, metal
2. 同时使用logistic回归和KNN作为分类器
3. 引入一些评价的标准来比较Logistic和KNN在测试集上的表现 
"""

# 准备音乐数据

'''
def create_fft(g, n):
    rad = "d:/genres/"+g+"/converted/"+g+"."+str(n).zfill(5)+".au.wav"
    sample_rate, X = wavfile.read(rad)
    fft_features = abs(fft(X)[:1000])
    sad = "d:/trainset/"+g+"."+str(n).zfill(5) + ".fft"
    np.save(sad, fft_features)
    
# -------create fft--------------


genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]
for g in genre_list:
    for n in range(100):
        create_fft(g, n)
'''

# 加载训练集数据,分割训练集以及测试集,进行分类器的训练
# 构造训练集！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
# -------read fft--------------
print('Starting load fft file...')

dataset_path = "D:/develop/workstations/GitHub/datasets/MachineLearning/MusicClassification/trainset/"
genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]
X = []
Y = []
for g in genre_list:
    for n in range(100):
        rad = dataset_path + g + "." + \
              str(n).zfill(5) + ".fft" + ".npy"
        fft_features = np.load(rad)
        X.append(fft_features)
        Y.append(genre_list.index(g))

X = np.array(X)
Y = np.array(Y)
# 首先我们要将原始数据分为训练集和测试集，这里是随机抽样80%做测试集，剩下20%做训练集 

randomIndex = random.sample(range(len(Y)), int(len(Y) * 8 / 10))
trainX = []
trainY = []
testX = []
testY = []
for i in range(len(Y)):
    if i in randomIndex:
        trainX.append(X[i])
        trainY.append(Y[i])
    else:
        testX.append(X[i])
        testY.append(Y[i])

# 接下来，我们使用sklearn，来构造和训练我们的两种分类器

print('Starting trainning  model...')

# ------train logistic classifier--------------
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver="liblinear", multi_class='auto', max_iter=1000)
model.fit(X, Y)
# model.fit(trainX, trainY)
# predictYlogistic = map(lambda x: model.predict(x)[0], testX)

# 可以采用Python内建的持久性模型 pickle 来保存scikit的模型
"""
>>> import pickle
>>> s = pickle.dumps(clf)
>>> clf2 = pickle.loads(s)
>>> clf2.predict(X[0])
"""

# ----train knn classifier-----------------------
"""
from sklearn.neighbors import NearestNeighbors

neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(trainX)
predictYknn = map(lambda x: trainY[neigh.kneighbors(x, return_distance=False)[0][0]], testX)

# 将predictYlogistic以及predictYknn与testY对比，我们就可以知道两者的判定正确率 
a = np.array(predictYlogistic) - np.array(testY)
print(a, np.count_nonzero(a), len(a))
accuracyLogistic = 1 - np.count_nonzero(a) / (len(a) * 1.0)
b = np.array(predictYknn) - np.array(testY)
print(b, np.count_nonzero(b), len(b))
accuracyKNN = 1 - np.count_nonzero(b) / (len(b) * 1.0)

print("%f" % accuracyLogistic)
print("%f" % accuracyKNN)
"""

print('Starting read test wavfile...')
# prepare test data By using logistic classifier-------------------
sample_rate, test = wavfile.read(dataset_path + "sample/heibao-wudizirong-remix.wav")
testdata_fft_features = abs(fft(test))[:1000]
# print(sample_rate, testdata_fft_features, len(testdata_fft_features))
type_index = model.predict([testdata_fft_features])[0]
print(type_index)
print(genre_list[type_index])

"""
from sklearn.metrics import confusion_matrix
cmlogistic = confusion_matrix(testY, predictYlogistic)
cmknn = confusion_matrix(testY, predictYknn)

def plotCM(cm,title,colorbarOn,givenAX):
    ncm=cm/cm.max()
    plt.matshow(ncm, fignum=False, cmap='Blues', vmin=0, vmax=2.0)
    if givenAX=="":
        ax=plt.axes()
    else:
        ax = givenAX
    ax.set_xticks(range(len(genre_list)))
    ax.set_xticklabels(genre_list)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(len(genre_list)))
    ax.set_yticklabels(genre_list)
    plt.title(title,size=12)
    if colorbarOn=="on":
        plt.colorbar()
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(i,j,cm[i,j],size=15)

plt.figure(figsize=(10, 5))  
fig1=plt.subplot(1, 2, 1)          
plotCM(cmlogistic,"confusion matrix: FFT based logistic classifier","off",fig1.axes)   
fig2=plt.subplot(1, 2, 2)     
plotCM(cmknn,"confusion matrix: FFT based KNN classifier","off",fig2.axes) 
plt.tight_layout(pad=0.4, w_pad=0, h_pad=1.0)     

plt.savefig("d:/confusion_matrix.png", bbox_inches="tight")
"""
