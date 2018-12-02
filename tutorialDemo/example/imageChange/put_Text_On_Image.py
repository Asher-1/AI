# -*- coding:utf-8 -*-
import os
import random
from PIL import ImageFont, Image, ImageDraw

ImagefolderPath = r'./img'
fontfolderPath = r'./font'
outputFileName = "output.jpg"
OutPath = r'./outPut/'

def getRandomColor():
    return (random.randint(30, 100), random.randint(30, 100), random.randint(30,100))

# create lists of fonts and images
fontlist = []
for filename in os.listdir(fontfolderPath):
    if os.path.splitext(filename)[1] == '.tff' or '.tcc' or '.TCC' or '.TFF':
        fontlist.append(os.path.join(fontfolderPath, filename))

imlist = []
for filename in os.listdir(ImagefolderPath):
    if os.path.splitext(filename)[1] == '.jpg':
        imlist.append(os.path.join(ImagefolderPath, filename))

# 计算图片张数和字体种类数
img_num = len(imlist)
font_num = len(fontlist)
print '处理图片总数为 ' + str(img_num) + ' 张'

# 循环图片列表，批处理图片
for i in range(img_num):
    # 打开图片
    im = Image.open(imlist[i])
    draw = ImageDraw.Draw(im)
    fontsize = min(im.size) / 4

    # 字体设置选择
    if i <= font_num - 1:
        font = ImageFont.truetype(fontlist[i], fontsize)
    else:
        font = ImageFont.truetype(fontlist[0], fontsize)

    # 绘制文字
   # draw.text((im.size[0] - fontsize, 0), str(i), font=font, fill=(0, 255, 0))
    draw.text((im.size[0] - fontsize, 0), str(i), font=font, fill= getRandomColor())
    # 保存修改后的图片
    im.save(OutPath + str(i) + '_' + outputFileName, 'jpeg')
