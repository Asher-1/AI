#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Asher
@time:2018/3/14 23:08
"""

# 验证码生成库
from captcha.image import ImageCaptcha  # pip install captcha
# import numpy as np
# from PIL import Image
import random
import sys
import os
# 递归删除包
# import shutil

dis_dir = 'captcha/images/'

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']


# ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# 随机获取验证码文本
def random_captcha_text(char_set=number + alphabet, captcha_size=4):
    # 验证码列表
    captcha_text = []
    for i in range(captcha_size):
        # 随机选择
        c = random.choice(char_set)
        # 加入验证码列表
        captcha_text.append(c)
    return captcha_text


# 递归删除文件夹和子文件夹以及文件
def remove_dir(dir_path):
    if not os.path.isdir(dir_path):
        print(u"删除目标不是文件夹")
        return

    files = os.listdir(dir_path)
    try:
        for file in files:
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                remove_dir(file_path)
        os.rmdir(dir_path)
    except Exception as err:
        print("Exception：", err)


# Setup the directory
def prepare_file_system(dir_name):
    if os.path.exists(dir_name):
        # 如果原始文件夹存在则递归删除(方法1：调用remove_dir(dir_name)；方法2：调用shutil.rmtree(dir_name))
        remove_dir(dir_name)
        # shutil.rmtree(dir_name)
        os.makedirs(dir_name)
    # Makes sure the folder exists on disk
    else:
        os.makedirs(dir_name)
    return


# 生成字符对应的验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    # 获得随机生成的验证码
    captcha_text = random_captcha_text()
    # 把验证码列表转为字符串
    captcha_text = ''.join(captcha_text)
    # 生成验证码
    captcha = image.generate(captcha_text)
    image.write(captcha_text, dis_dir + captcha_text + '.jpg')  # 写到文件


# 数量少于10000，因为重名
num = 10000
if __name__ == '__main__':
    # 准备目标文件夹，檢查文件夾是否存在，存在则删除并创建，不存在則直接創建
    prepare_file_system(dis_dir)
    for i in range(num):
        # 生成验证码图片
        gen_captcha_text_and_image()
        sys.stdout.write('\r>> Creating image %d/%d' % (i + 1, num))
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()

    print("生成完毕")
