#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@author: Asher
@time:2018/2/25 19:20
'''
import json
import requests

# myparams = {'q': 'linux'}
# r1 = requests.get('http://c.itcast.cn')
# r3 = requests.get('http://www.haosou.com/s', params=myparams)
# print r3.text
# print r3.content


# mydata = {'name': 'xwp', 'wd': 'linux'}
# r = requests.post('http://httpbin.org/post', data=mydata)
#
# print r.text

mydata = {'name': 'xwp', 'wd': 'linux'}
r = requests.post('http://httpbin.org/post', data=json.dumps(mydata))

print r.text


# myfile = {'file': open('xwp.jpg', 'rb')}
# r2 = requests.post('http://httpbin.org/post', files=myfile)
# print r2.text
