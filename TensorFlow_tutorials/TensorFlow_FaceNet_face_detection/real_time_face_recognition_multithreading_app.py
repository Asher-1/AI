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
import argparse
from queue import Queue
from threading import Thread
from src.app_utils import FPS, WebcamVideoStream, draw_boxes_and_labels
from src import face

classifier_model = os.path.dirname(__file__) + "/models/classifiers/sex_SVC_classifier.pkl"


# 添加识别标记
def add_overlays(frame, faces, points):
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

    # cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
    #             thickness=2, lineType=2)


# 获取脸部区域和脸部特征点
def get_faces_points(face_recognition, frame):
    faces, points = face_recognition.identify(frame)
    return dict(class_faces=faces, class_points=points)


# 线程
def worker(input_q, output_q):
    # 创建脸部识别器
    face_recognition = face.Recognition(classifier_model)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 识别结果放入线程
        output_q.put(get_faces_points(face_recognition, frame_rgb))

    fps.stop()
    sess.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=int,
                        default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int,
                        default=640, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int,
                        default=480, help='Height of the frames in the video stream.')
    args = parser.parse_args()

    input_q = Queue(5)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

    video_capture = WebcamVideoStream(src=args.video_source,
                                      width=args.width,
                                      height=args.height).start()
    fps = FPS().start()

    while True:
        # Capture frame-by-frame
        frame = video_capture.read()
        input_q.put(frame)

        t = time.time()
        if output_q.empty():
            pass  # fill up queue
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            data = output_q.get()
            faces = data['class_faces']
            points = data['class_points']
            # 添加识别标记
            add_overlays(frame, faces, points)
            # 显示识别结果
            cv2.imshow('Video', frame)

        # 更新窗口
        fps.update()
        # 打印画面延迟信息
        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
        # 响应按键消息
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
