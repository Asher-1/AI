#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import jieba
import re
import os


def load_stopwords():
    f = open('stopword.txt')
    for w in f:
        stopwords.add(w.strip().decode('GB18030'))
    f.close()


def segment_one_file(input_file_name, output_file_name):
    f = open(input_file_name, mode='r')
    f_output = open(output_file_name, mode='w')
    pattern = re.compile('<content>(.*?)</content>')
    for line in f:
        line = line.decode('GB18030')
        news = re.findall(pattern=pattern, string=line)
        for one_news in news:
            words_list = []
            words = jieba.cut(one_news.strip())
            for word in words:
                word = word.strip()
                if word not in stopwords:
                    words_list.append(word)
            if len(words_list) > 10:
                s = u' '.join(words_list)
                f_output.write(s.encode('utf-8') + '\n')
    f.close()
    f_output.close()


if __name__ == "__main__":
    stopwords = set()
    load_stopwords()
    input_dir = '.\\200806\\'
    output_dir = '.\\200806_segment'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for file_name in os.listdir(input_dir):
        if os.path.splitext(file_name)[-1] == '.txt':
            print file_name
            segment_one_file(os.path.join(input_dir, file_name), os.path.join(output_dir, file_name))
