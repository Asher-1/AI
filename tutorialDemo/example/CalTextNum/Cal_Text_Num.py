# -*- coding:utf-8 -*-
import re
import os
from collections import Counter

FILESOURCE = r'./media/abc.txt'
FILE_PATH = r'./media'

def getCounter(articlefilesource):
    '''输入一个英文的纯文本文件，统计其中的单词出现的个数'''
    pattern = r'''[A-Za-z]+|\$?\d+%?$'''
    with open(articlefilesource) as f:
        r = re.findall(pattern, f.read())
        return Counter(r)

def getMostCommonWord(articlefilesource):
    '''输入一个英文的纯文本文件，统计其中的单词出现的个数'''
    pattern = r'''[A-Za-z]+|\$?\d+%?$'''
    with open(articlefilesource) as f:
        r = re.findall(pattern, f.read())
        return Counter(r).most_common()

# 过滤词
stop_word = ['the', 'in', 'of', 'and', 'to', 'has', 'that', 's', 'is',
             'are', 'a', 'with', 'as', 'an']

def run(FILE_PATH):
    # 切换到目标文件夹所在目录
    os.chdir(FILE_PATH)
    # 遍历该目录下的txt文件
    total_counter = Counter()
    for i in os.listdir(os.getcwd()):
        if os.path.splitext(i)[1] == '.txt':
            total_counter += getCounter(i)

    # 排除stopword的影响
    for i in stop_word:
        total_counter[i] = 0

    word = total_counter.most_common()[0][0]
    print ('出现次数最多的单词为：' + word + '-----其个数为：' + str(total_counter.most_common()[0][1]))

if __name__ == '__main__':
    help('Cal_Text_Num.getMostCommonWord')
    print (getMostCommonWord(FILESOURCE))

    run(FILE_PATH)
