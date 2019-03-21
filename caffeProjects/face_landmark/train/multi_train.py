#!usr/bin/env python
# -*- coding:utf-8 -*-

"""
@author: Asher
@file: multi_train.py
@time: 2019/03/20

This file train Caffe CNN models
"""

import os, sys
import caffe
import multiprocessing
import sys

sys.path.insert(0, "..")
from common import createDir

ROOT_PATH = "D:/develop/workstations/GitHub/Datasets/DL/Images/face_data/deep_landmark/"
MODEL_OUTPUT_PATH = ROOT_PATH + "mymodel/{0}_{1}/"

pool_on = False

models = [
    ['F', 'EN', 'NM'],
    ['LE1', 'LE2', 'RE1', 'RE2', 'N1', 'N2', 'LM1', 'LM2', 'RM1', 'RM2'],
    ['LE1', 'LE2', 'RE1', 'RE2', 'N1', 'N2', 'LM1', 'LM2', 'RM1', 'RM2'], ]


def runCaffe(solver_prototxt):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_prototxt)
    # 预设权重，或者从已经训练好的文件中重新开始训练
    # solver.net.copy_from(ROOT_PATH + "alexnet_iter_50000_full_conv.caffemodel")
    # 加载已训练的模型，继续训练的时候可以使用这个
    # solver.restore(ROOT_PATH + "alexnet_iter_50000_full_conv.caffemodel")
    # 直接训练，直接得到一个最终的训练models
    solver.solve()


def train(level=1):
    """
        train caffe model
    """
    prototxts = []
    for t in models[level - 1]:
        createDir(MODEL_OUTPUT_PATH.format(level, t))
        prototxt = (ROOT_PATH + 'my_prototxt/{0}_{1}_solver.prototxt').format(level, t)
        prototxts.append(prototxt)
    # we train level-2 and level-3 with mutilprocess (we may train two level in parallel)
    if pool_on:
        pool_size = 3
        pool = multiprocessing.Pool(processes=pool_size, maxtasksperchild=2)
        pool.map(runCaffe, prototxts)
        pool.close()
        pool.join()
    else:
        for pototxt in prototxts:
            runCaffe(pototxt)


if __name__ == '__main__':
    pool_on = True
    for level in range(1, 4):
        train(level)
