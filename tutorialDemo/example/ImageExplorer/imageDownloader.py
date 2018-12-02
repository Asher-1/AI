#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
@author: Asher
@time:2018/3/7 17:27
'''
import requests


def download(url, filename):
    req = requests.get(url)
    with open(filename, 'wb') as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)


if __name__ == '__main__':
    url = 'http://d.hiphotos.baidu.com/image/pic/' \
          'item/8601a18b87d6277fcdb9b01d24381f30e924fc68.jpg'

    url2 = 'https://timgsa.baidu.com/timg?image&quality=80&size=' \
          'b9999_10000&sec=1520425528283&di=64afd815fc5876effc6afb' \
          'b69ea0922b&imgtype=0&src=http%3A%2F%2Fscimg.jb51.net%2Fal' \
          'limg%2F170116%2F106-1F116114312L2.jpg'

    filename = 'out.jpg'
    download(url, filename)



