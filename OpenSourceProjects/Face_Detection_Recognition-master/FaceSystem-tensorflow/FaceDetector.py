from PyQt4 import QtCore
import MyGui
import numpy as np
import time
import cv2
import sys

src = "D:/develop/workstations/GitHub/AI/OpenSourceProjects/Face_Detection_Recognition-master/faceRecognition/"
sys.path.insert(0, src)
from utils import file_processing, image_processing
import face_recognition

resize_height = 160
resize_width = 160

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

            if len(img.shape) == 2:  # 若是灰度图则转为三通道
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
            self.rgb_image = np.asanyarray(rgb_image)

            # 进行人脸检测，获得bounding_box
            bounding_box, points = self.face_detector.detect_face(self.rgb_image)
            bounding_box = bounding_box[:, 0:4].astype(int)

            # 获得人脸区域
            face_images = image_processing.get_crop_images(self.rgb_image, bounding_box, resize_height, resize_width, whiten=True)
            if len(face_images) > 0:
                self.textBrowser.append('Number of face detected: {}'.format(len(face_images)))
                if self.ldmarking:
                    self.face_info[0] = (bounding_box, points, face_images)
                else:
                    self.face_info[0] = (bounding_box, [], face_images)
            else:
                self.face_info[0] = ([], [], face_images)
            # emit signal when detection finished
            self.emit(QtCore.SIGNAL('det(PyQt_PyObject)'), [self.face_info, self.rgb_image])

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
