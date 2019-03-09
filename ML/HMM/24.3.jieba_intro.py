# !/usr/bin/python
# -*- coding:utf-8 -*-

import sys
import jieba
import jieba.posseg


if __name__ == "__main__":
    reload(sys)
    sys.setdefaultencoding('utf-8')
    f = open('.\\novel.txt')
    str = f.read().decode('utf-8')
    f.close()

    seg = jieba.posseg.cut(str)
    for word, flag in seg:
        print word, flag, '|',
