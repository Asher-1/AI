#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Asher
@time:2018/8/6 17:07
"""
import numpy as np
import cv2
import os
im = cv2.imread('src1.png')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)

cv2.imshow("original", thresh)
cv2.waitKey(0)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
for i in range(0,len(contours)):

    x, y, w, h = cv2.boundingRect(contours[i])
    cv2.rectangle(image, (x,y), (x+w,y+h), (153,153,0), 5)

    newimage = image[y + 2:y + h - 2, x + 2:x + w - 2]  # 先用y确定高，再用x确定宽
    nrootdir = ("./cut_image/")
    if not os.path.isdir(nrootdir):
        os.makedirs(nrootdir)
    cv2.imwrite(nrootdir + str(i) + ".jpg", newimage)
    print(i)

