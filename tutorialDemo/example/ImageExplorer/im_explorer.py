#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@author: Asher
@time:2018/3/7 17:09
'''

import tkinter as tk
import os
from PIL import ImageTk, Image


class Application(tk.Frame):
    def __init__(self, master, path):
        self.path = path
        self.files = os.listdir(self.path)
        print(self.files)
        self.index = 0
        self.img = ImageTk.PhotoImage(Image.open(self.path + '\\' + self.files[self.index]))

        # self.img = tk.PhotoImage(file=self.path+'\\'+self.files[self.index])
        print(self.path + '\\' + self.files[self.index])
        tk.Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

    def createWidgets(self):
        self.lblImage = tk.Label(self)
        self.lblImage['image'] = self.img
        self.lblImage.pack()
        self.f = tk.Frame()
        self.f.pack()
        self.btnPrev = tk.Button(self.f, text='上一张', command=self.prev)
        self.btnPrev.pack(side=tk.LEFT)
        self.btnNext = tk.Button(self.f, text='下一张', command=self.next)
        self.btnNext.pack(side=tk.LEFT)

    def prev(self):
        self.showfile(-1)

    def next(self):
        self.showfile(1)

    def showfile(self, n):
        self.index += n
        if self.index < 0:
            self.index = len(self.files)
        if self.index > len(self.files) - 1:
            self.index = 0
        self.img = ImageTk.PhotoImage(Image.open(self.path + '\\' + self.files[self.index]))
        # self.img = tk.PhotoImage(file=self.path+'\\'+self.files[self.index])
        self.lblImage['image'] = self.img


if __name__ == '__main__':
    root = tk.Tk()
    root.title('图片浏览器')
    path = r'.\img'
    app = Application(root, path)
    app.mainloop()