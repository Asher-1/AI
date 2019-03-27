#!usr/bin/env python
# -*- coding:utf-8 -*-

"""
@author: Asher
@file: train.py
@time: 2019/03/20
"""

import caffe

import sys
sys.path.insert(0, "..")
from common import createDir

ROOT_PATH = "D:/develop/workstations/GitHub/Datasets/DL/Images/face_data/deep_landmark/"
MODEL_OUTPUT_PATH = ROOT_PATH + "mymodel/{0}_{1}/"

Pretrained_model_path = "D:/develop/workstations/GitHub/Datasets/DL/trained_outputs/face_system_output/model/VGG_FACE.caffemodel"

nameDict = {
    's0': ['F'],
    's1': ['EN', 'NM'],
    's3': ['LE1', 'LE2', 'RE1', 'RE2', 'N1', 'N2', 'LM1', 'LM2', 'RM1', 'RM2'], }

level = 3
target_name = nameDict['s3'][-1]

MODEL_OUTPUT_PATH = MODEL_OUTPUT_PATH.format(level, target_name)

if __name__ == "__main__":
    caffe.set_device(0)
    caffe.set_mode_gpu()

    createDir(MODEL_OUTPUT_PATH)

    solver_file = (ROOT_PATH + 'my_prototxt/{0}_{1}_solver.prototxt').format(level, target_name)
    solver = caffe.SGDSolver(solver_file)

    # 预设权重，或者从已经训练好的文件中重新开始训练
    # solver.net.copy_from(Pretrained_model_path)

    # 加载已训练的模型，继续训练的时候可以使用这个
    # solver.restore(ROOT_PATH + "")
    # 直接训练，直接得到一个最终的训练models
    solver.solve()

