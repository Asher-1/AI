#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@author: Asher
@time:2018/2/24 20:06
'''

import tensorflow as tf
import os
import tarfile
import requests

# inception模型下载地址
data_url = 'http://download.tensorflow.org/example_images/flower_photos.tgz'

# 模型存放地址
data_dir = "datasets"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 获取文件名，以及文件路径
filename = data_url.split('/')[-1]
filepath = os.path.join(data_dir, filename)

# 下载模型
if not os.path.exists(filepath):
    print("download: ", filename)
    r = requests.get(data_url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
print("finish: ", filename)

# 获取解压目标文件夹的名称
folername = os.path.exists(os.path.join(data_dir, filename.split('.')[0]))
# 判断压缩文件是否解压过了
if not folername:
    # 解压文件
    tarfile.open(filepath, 'r:gz').extractall(data_dir)

print("数据解压完毕！！！")

