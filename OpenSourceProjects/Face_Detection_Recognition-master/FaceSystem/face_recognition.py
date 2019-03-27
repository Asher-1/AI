from PyQt4 import QtCore
import numpy as np
import glob
import caffe
import dlib
import sklearn.metrics.pairwise
import cv2

ROOT_PATH = "D:/develop/workstations/GitHub/Datasets/DL/trained_outputs/face_system_output/"
MODEL_PATH = ROOT_PATH + "model/"
DATABASE_PATH = ROOT_PATH + "db"


class Face_recognizer(QtCore.QThread):
    def __init__(self, textBrowser):
        super(Face_recognizer, self).__init__()

        # load face model
        caffemodel = MODEL_PATH + 'VGG_FACE.caffemodel'
        deploy_file = MODEL_PATH + 'VGG_FACE_deploy.prototxt'
        # 加载数据库的数据
        dataset_emb, names_list = load_dataset(dataset_path, filename)
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

    def face_recognition(self, face_info):
        if self.recognizing:
            img = []
            bounding_box = face_info[0]
            face_images = face_info[2]
            pred_emb = face_net.get_embedding(face_images)
            self.pred_name = compare_embadding(pred_emb, self.dataset_emb, self.names_list)
            assert (len(bounding_box) == len(pred_name))
            # dist = sklearn.metrics.pairwise.cosine_similarity(fea, self.db)
            # print('dist = {}'.format(dist))
            # pred = np.argmax(dist, 1)
            # dist = np.max(dist, 1)

            # print('pred = {}'.format(pred))
            # print('maxdist = {}'.format(dist))
            # print('threshold = {}'.format(self.threshold/100.0))
            # pred = [0 if dist[i] < self.threshold / 100.0 else pred[i] + 1 for i in range(len(dist))]

            # print('pred(after threshold) = {}'.format(pred))

            # writ on GUI

            msg = QtCore.QString("Face Recognition Pred: <span style='color:red'>{}</span>".format(
                ' '.join([name for name in self.pred_name])))
            self.textBrowser.append(msg)
            # emit signal when detection finished
            self.emit(QtCore.SIGNAL('face_id(PyQt_PyObject)'), [bounding_box, self.pred_name])

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
