#!/usr/bin/env python2.7
# coding: utf-8
"""
    This file generate prototxt file for LEVEL-2 and LEVEL-3
"""

import sys
import os
from os.path import join, exists

ROOT_PATH = "D:/develop/workstations/GitHub/Datasets/DL/Images/face_data/"
PROTOTXT_TEMPLATE_PATH = ROOT_PATH + "deep_landmark/prototxt_template/"
PROTOTXT_OUTPUT = ROOT_PATH + "deep_landmark/my_prototxt/"

if not exists(PROTOTXT_OUTPUT):
    os.makedirs(PROTOTXT_OUTPUT)
assert (exists(PROTOTXT_TEMPLATE_PATH) and exists(PROTOTXT_OUTPUT))

nameDict = {
    's0': ['F'],
    's1': ['EN', 'NM'],
    's3': ['LE1', 'LE2', 'RE1', 'RE2', 'N1', 'N2', 'LM1', 'LM2', 'RM1', 'RM2'], }


def generate(network, level, names, mode='GPU'):
    """
        Generate template
        network: CNN type
        level: LEVEL
        names: CNN names
        mode: CPU or GPU
    """
    assert (mode == 'GPU' or mode == 'CPU')

    types = ['train', 'solver', 'deploy']
    for name in names:
        for t in types:
            tempalteFile = (PROTOTXT_TEMPLATE_PATH + '{0}_{1}.prototxt.template').format(network, t)
            with open(tempalteFile, 'r') as fd:
                template = fd.read()
                outputFile = (PROTOTXT_OUTPUT + '{0}_{1}_{2}.prototxt').format(level, name, t)
                with open(outputFile, 'w') as fd:
                    fd.write(template.format(level=level, name=name, mode=mode))


def generate_train(network, level, names):
    for name in names:
        tempalteFile = PROTOTXT_TEMPLATE_PATH + 'train.template'
        with open(tempalteFile, 'r') as fd:
            template = fd.read()
            outputFile = PROTOTXT_OUTPUT + '{0}_{1}_train.sh'.format(level, name)
            with open(outputFile, 'w') as fd:
                fd.write(template.format(level=level, name=name))


if __name__ == '__main__':
    mode = 'GPU'  # 'GPU'

    generate('s0', 1, nameDict['s0'], mode)
    generate('s1', 1, nameDict['s1'], mode)
    generate('s3', 2, nameDict['s3'], mode)
    generate('s3', 3, nameDict['s3'], mode)

    generate_train('s0', 1, nameDict['s0'])
    generate_train('s1', 1, nameDict['s1'])
    generate_train('s3', 2, nameDict['s3'])
    generate_train('s3', 3, nameDict['s3'])
    # Done
