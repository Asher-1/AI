from PyQt4 import QtGui
from PyQt4.QtWebKit import *
import sys

from MyGui import *
from capture import *
from FaceDetector import *
from face_recognizer import *
from functools import partial

# qt dark theme of the GUI
import qdarkstyle


ROOT_PATH = "D:/develop/workstations/GitHub/Datasets/facenet-detection/dataset/test_videos/"
VIDEO_PATH = ROOT_PATH + "huge.mp4"

def main():
    app = QtGui.QApplication(['Face_Demo'])
    app.setStyleSheet(qdarkstyle.load_stylesheet(pyside=False))

    # Create Gui Form
    form = MyGUi()

    # Create video capture thread and run
    capture = Capture(0)
    # capture = Capture(VIDEO_PATH)
    capture.start()
    # connect GUI widgets
    form.pushButton.clicked.connect(capture.quitCapture)
    form.pushButton_2.clicked.connect(capture.startCapture)
    form.pushButton_3.clicked.connect(capture.endCapture)

    # Create face detector thread and run
    face_detector = Face_detector(form.textBrowser)
    face_detector.connect(capture, QtCore.SIGNAL("getFrame(PyQt_PyObject)"), face_detector.detect_face)
    # Connect GUI widgets
    enable_slot_det = partial(face_detector.startstopdet, form.checkBox_2)
    form.checkBox_2.stateChanged.connect(lambda x: enable_slot_det())

    enable_slot_ldmark = partial(face_detector.startstopldmark, form.checkBox)
    form.checkBox.stateChanged.connect(lambda x: enable_slot_ldmark())
    form.connect(face_detector, QtCore.SIGNAL('det(PyQt_PyObject)'), form.drawFace)

    # Create deep net for face recognition
    form.dial.setRange
    form.dial.setValue(2)
    face_network = Face_recognizer(form.textBrowser)
    face_network.connect(face_detector, QtCore.SIGNAL('det(PyQt_PyObject)'), face_network.face_recognize)

    # Connect GUI Widgets
    enable_slot_identity = partial(face_network.startstopfacerecognize, form.checkBox_4)
    form.checkBox_4.stateChanged.connect(lambda x: enable_slot_identity())
    form.dial.valueChanged.connect(face_network.set_threshold)
    form.connect(face_network, QtCore.SIGNAL('face_id(PyQt_PyObject)'), form.drawIdentity)

    form.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
