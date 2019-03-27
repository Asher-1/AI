import dlib
from PyQt4 import QtCore
import MyGui
import numpy as np
import time
import cv2

DLIB_MODEL_PATH = "D:/develop/workstations/GitHub/Datasets/pretrained_models/dlib_model/"


class Face_detector(QtCore.QThread):
    def __init__(self, textBrowser):
        super(Face_detector, self).__init__()
        # 初始化mtcnn人脸检测
        self.face_detector = face_recognition.Facedetection()
        self.face_info = {}
        self.textBrowser = textBrowser
        self.detecting = True  # flag of if detect face
        self.ldmarking = False  # flag of if detect landmark
        self.total = 0

    def detect_face(self, img):
        if self.detecting:
            self.face_info = {}
            # det_start_time = time.time()
            if img is None:
                return None

            # 进行人脸检测，获得bounding_box
            bounding_box, points = self.face_detector.detect_face(img)
            bounding_box = bounding_box[:, 0:4].astype(int)
            # 获得人脸区域
            face_images = image_processing.get_crop_images(img, bounding_box, resize_height, resize_width,
                                                           whiten=True)
            if len(face_images) > 0:
                self.textBrowser.append('Number of face detected: {}'.format(len(face_images)))
                if self.ldmarking:
                    self.face_info[0] = (bounding_box, points, face_images)
                else:
                    self.face_info[0] = (bounding_box, [], face_images)
            # emit signal when detection finished
            self.emit(QtCore.SIGNAL('det(PyQt_PyObject)'), [self.face_info, img])

    def startstopdet(self, checkbox):
        if checkbox.isChecked():
            self.detecting = True
        else:
            self.detecting = False

    def startstopldmark(self, checkbox):
        if checkbox.isChecked():
            self.ldmarking = True
        else:
            self.ldmarking = False
