from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import QObject
import mainwindow


class MyGUi(QtGui.QMainWindow, mainwindow.Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyGUi, self).__init__(parent)
        self.setupUi(self)

        # set LCD number display according to dial
        QObject.connect(self.dial, QtCore.SIGNAL("valueChanged(int)"), self.lcdNumber.display)

        # bind check box buttons
        self.checkBox_2.stateChanged.connect(lambda: self.check1_state(self.checkBox, self.checkBox_3))
        self.frame = None
        self.painter = QtGui.QPainter()
        self.gender_pred = [[], []]
        self.gender_label = ['Female', 'Male']

        self.identity_pred = [[], [], []]
        self.face_labels = []

    def check1_state(self, check2, check3):
        if not self.checkBox_2.isChecked():
            check2.setChecked(False)
            check3.setChecked(False)

    def setFrame(self, frame):
        self.frame = frame
        pixmap = QtGui.QPixmap.fromImage(self.frame)
        scaledpixmap = pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio)
        self.label.setPixmap(scaledpixmap)

    def drawFace(self, face_info):
        self.frame = face_info[1]
        face_info = face_info[0]

        image = QtGui.QImage(self.frame.tostring(), self.frame.shape[1], self.frame.shape[0],
                             QtGui.QImage.Format_RGB888).rgbSwapped()

        self.painter.begin(image)
        self.painter.setRenderHint(QtGui.QPainter.Antialiasing)
        self.painter.setBrush(QtGui.QBrush(QtGui.QColor(192, 192, 255)))

        bounding_boxes = face_info[0]
        landmarks = face_info[1]
        assert(len(bounding_boxes) == len(landmarks))
        for i in len(bounding_boxes):
            x, y = bounding_boxes[i][0], bounding_boxes[i][1]
            w = bounding_boxes[i][2] - x
            h = bounding_boxes[i][3] - y
            self.drawLines(x, y, w, h)
            self.drawPoints(landmarks[i])

        identity_pos = self.identity_pred[0]
        identity_str = self.identity_pred[1]
        # draw identity for every face
        for i in range(len(identity_str)):
            font = self.painter.font()
            font.setPixelSize(20)
            self.painter.setFont(font)
            self.painter.drawText(int(identity_pos[i][0]), int(identity_pos[i][1]),
                                  QtCore.QString(identity_str[i]))

        self.identity_pred = [[], [], []]
        self.painter.end()

        pixmap = QtGui.QPixmap.fromImage(image)
        # scaledpixmap = pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio)
        scaledpixmap = pixmap.scaled(self.label.size())
        self.label.setPixmap(scaledpixmap)

    def drawLines(self, x, y, w, h):
        pen = QtGui.QPen(QtCore.Qt.green, 5, QtCore.Qt.SolidLine)
        sz = 0.1
        self.painter.setPen(pen)
        self.painter.drawLine(x, y, x + w * sz, y)
        self.painter.drawLine(x, y, x, y + h * sz)

        self.painter.drawLine(x + w, y, x + w, y + h * sz)
        self.painter.drawLine(x + w, y, x + w * (1 - sz), y)

        self.painter.drawLine(x, y + h, x, y + h * (1 - sz))
        self.painter.drawLine(x, y + h, x + w * sz, y + h)

        self.painter.drawLine(x + w, y + h, x + w * (1 - sz), y + h)
        self.painter.drawLine(x + w, y + h, x + w, y + h * (1 - sz))

        brush = QtGui.QBrush(QtCore.Qt.SolidPattern)
        brush.setColor(QtCore.Qt.blue)
        self.painter.setBrush(brush)
        brush.setStyle(QtCore.Qt.NoBrush)
        self.painter.setBrush(brush)
        pen = QtGui.QPen(QtCore.Qt.darkBlue, 2, QtCore.Qt.DotLine)
        self.painter.setPen(pen)
        self.painter.drawRect(x, y, w, h)

    def drawPoints(self, landmarks):
        pen = QtGui.QPen(QtCore.Qt.darkBlue, 4, QtCore.Qt.SolidLine)
        self.painter.setPen(pen)
        for mk in landmarks:
            self.painter.drawPoint(mk[0], mk[1])

    def drawIdentity(self, identity_info):
        self.identity_pred = identity_info
