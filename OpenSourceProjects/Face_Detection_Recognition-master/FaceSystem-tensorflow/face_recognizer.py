from PyQt4 import QtCore

try:
    from PyQt4.QtCore import QString
except ImportError:
    QString = str
import numpy as np
import glob
import sklearn.metrics.pairwise
import cv2
import sys
import os

src = "D:/develop/workstations/GitHub/AI/OpenSourceProjects/Face_Detection_Recognition-master/faceRecognition/"
sys.path.insert(0, src)
from utils import file_processing, image_processing
import face_recognition

ROOT_PATH = "D:/develop/workstations/GitHub/Datasets/facenet-detection/"
MODEL_PATH = ROOT_PATH + "models/20180408-102900"
DATABASE_PATH = ROOT_PATH + 'dataset/emb/faceEmbedding.npy'
filename = ROOT_PATH + 'dataset/emb/name.txt'


class Face_recognizer(QtCore.QThread):
    def __init__(self, textBrowser):
        super(Face_recognizer, self).__init__()
        # 初始化facenet
        self.face_net = face_recognition.facenetEmbedding(MODEL_PATH)
        # 加载数据库的数据
        dataset_emb, names_list = self.load_dataset(DATABASE_PATH, filename)
        self.dataset_emb = dataset_emb
        self.names_list = names_list
        self.recognizing = False
        self.textBrowser = textBrowser
        self.threshold = 0
        self.label = ['Stranger']

    def load_dataset(self, dataset_path, filename):
        '''
        加载人脸数据库
        :param dataset_path: embedding.npy文件（faceEmbedding.npy）
        :param filename: labels文件路径路径（name.txt）
        :return:
        '''
        compare_emb = np.load(dataset_path)
        names_list = file_processing.read_data(filename)
        return compare_emb, names_list

    def face_recognize(self, face_info):
        if self.recognizing:
            pred_name = []
            bounding_box = face_info[0][0][0]
            face_images = face_info[0][0][2]
            if len(bounding_box) > 0 and len(face_images) > 0:
                pred_emb = self.face_net.get_embedding(face_images)
                pred_name = self.compare_embadding(pred_emb, self.dataset_emb, self.names_list)
            assert (len(bounding_box) == len(pred_name))

            # writ on GUI
            if len(pred_name) == 0:
                msg = QString("there is no face!!! ")
            else:
                msg = QString("Face Recognition Pred: <span style='color:red'>{}</span>".format(
                    ' '.join([name for name in pred_name])))
            self.textBrowser.append(msg)

            for i, name in enumerate(pred_name):
                if name != "unknow":
                    pred_name[i] = os.path.basename(os.path.dirname(name))
            # emit signal when detection finished
            self.emit(QtCore.SIGNAL('face_id(PyQt_PyObject)'), [bounding_box, pred_name])

    def compare_embadding(self, pred_emb, dataset_emb, names_list):
        # 为bounding_box 匹配标签
        pred_num = len(pred_emb)
        dataset_num = len(dataset_emb)
        pred_name = []
        for i in range(pred_num):
            dist_list = []
            for j in range(dataset_num):
                dist = np.sqrt(np.sum(np.square(np.subtract(pred_emb[i, :], dataset_emb[j, :]))))
                dist_list.append(dist)
            min_value = min(dist_list)
            if (min_value > self.threshold):
                pred_name.append('unknow')
            else:
                pred_name.append(names_list[dist_list.index(min_value)])
        return pred_name

    def set_threshold(self, th):
        self.threshold = th
        self.textBrowser.append('Threshold is changed to: {}'.format(self.threshold))

    def startstopfacerecognize(self, checkbox):
        if checkbox.isChecked():
            self.recognizing = True
        else:
            self.recognizing = False
