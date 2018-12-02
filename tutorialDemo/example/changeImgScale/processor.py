#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
from PIL import Image

myPath = r'./img/'
outPath = r'destdir/'

def processImage(filesource, destsource, name, imgtype):
    '''
    filesource是存放待转换图片的目录
    destsource是存放输出转换后图片的目录
    name是文件名
    imgtype是文件类型
    '''
    imgtype = 'jpeg' if imgtype == '.jpg' else 'png'
    #打开图片
    im = Image.open(filesource + name)
    # 缩放比例
    rate = max(im.size[0]/320.0 if im.size[0] > 320 else 0, im.size[1]/540.0 if im.size[1] > 540 else 0)
    if rate:
        im.thumbnail((im.size[0]/rate, im.size[1]/rate))
    im.save(destsource + name, imgtype)

def run(myPath):
    # 切换到源目录，遍历源目录下所有图片
    os.chdir(myPath)
    absInputPath = os.getcwd() + '\\'
    absoutPutPath = os.getcwd().replace('img', '') + outPath
    for i in os.listdir(os.getcwd()):
        # 检查后缀
        postfix = os.path.splitext(i)[1]
        if postfix == '.jpg' or postfix == '.png':
            processImage(absInputPath, absoutPutPath, i, postfix)

if __name__ == '__main__':
    run(myPath)
