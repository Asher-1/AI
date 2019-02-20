# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import time
import cv2
from src import face

root_path = "D:/develop/workstations/GitHub/Datasets/facenet-detection/"

classifier_model = root_path + "models/classifiers/sex_SVC_classifier.pkl"
# classifier_model = os.path.dirname(__file__) + "/models/classifiers/sex_SVC_classifier.pkl"


def add_overlays(frame, faces, points, frame_rate):
    if faces is not None:
        num = -1
        for face in faces:
            num += 1
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), (0, 255, 0), 2)
            # draw feature points
            cv2.circle(frame, (points[0][num], points[5][num]), 2, (0, 255, 0), -1)
            cv2.circle(frame, (points[1][num], points[6][num]), 2, (0, 255, 0), -1)
            cv2.circle(frame, (points[2][num], points[7][num]), 2, (0, 255, 0), -1)
            cv2.circle(frame, (points[3][num], points[8][num]), 2, (0, 255, 0), -1)
            cv2.circle(frame, (points[4][num], points[9][num]), 2, (0, 255, 0), -1)
            # 绘制所属分类类别
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)


def main():
    frame_interval = 3  # Number of frames after which to run face detection
    fps_display_interval = 5  # seconds
    frame_rate = 0
    frame_count = 0

    video_capture = cv2.VideoCapture(0)
    face_recognition = face.Recognition(classifier_model)
    start_time = time.time()

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if (frame_count % frame_interval) == 0:
            faces, points = face_recognition.identify(frame)
            # Check our current fps
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0

        # 添加识别标记
        add_overlays(frame, faces, points, frame_rate)

        frame_count += 1
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
