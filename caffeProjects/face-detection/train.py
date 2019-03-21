#!usr/bin/env python
# -*- coding:utf-8 -*-

"""
@author: Asher
@file: train.py
@time: 2019/03/19
"""

import caffe

ROOT_PATH = "D:/develop/workstations/GitHub/Datasets/DL/Images/face_data/caffe_face_data/model/"

if __name__ == "__main__":
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(ROOT_PATH + 'solver.prototxt')

    # 预设权重，或者从已经训练好的文件中重新开始训练
    solver.net.copy_from(ROOT_PATH + "alexnet_iter_50000_full_conv.caffemodel")

    # 加载已训练的模型，继续训练的时候可以使用这个
    # solver.restore(ROOT_PATH + "alexnet_iter_50000_full_conv.caffemodel")
    # 直接训练，直接得到一个最终的训练models
    solver.solve()
