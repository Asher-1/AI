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

import logging
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from object_detection.utils.app_utils import FPS, WebcamVideoStream, draw_boxes_and_labels

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0
e = None


# 线程
def worker(input_q, output_q):
    fps = FPS().start()
    while True:
        fps.update()
        frame_rgb = input_q.get()
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # logger.debug('image process+')
        humans = e.inference(frame_rgb)
        # logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(frame_rgb, humans, imgcopy=False)

        # 识别结果放入线程
        output_q.put(dict(class_images=image))

    fps.stop()
    sess.close()


def main():
    global e, fps_time
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')

    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='640x480', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')

    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    logger.debug('cam read+')

    input_q = Queue(3)  # fps is better if queue is higher but then more lags
    output_q = Queue()
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

    video_capture = WebcamVideoStream(src=args.camera,
                                      width=w,
                                      height=h).start()
    frame = video_capture.read()
    logger.info('cam image=%dx%d' % (frame.shape[1], frame.shape[0]))

    fps = FPS().start()

    while True:
        # Capture frame-by-frame
        image = video_capture.read()
        # logger.debug('image preprocess+')

        if args.zoom < 1.0:
            canvas = np.zeros_like(image)
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            image = canvas
        elif args.zoom > 1.0:
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (img_scaled.shape[1] - image.shape[1]) // 2
            dy = (img_scaled.shape[0] - image.shape[0]) // 2
            image = img_scaled[dy:image.shape[0], dx:image.shape[1]]

        input_q.put(image)

        t = time.time()
        if output_q.empty():
            pass  # fill up queue
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            data = output_q.get()
            target_image = data['class_images']

            # 显示识别结果
            # logger.debug('show+')
            cv2.putText(target_image,
                        "FPS: %f" % (1.0 / (time.time() - fps_time)),
                        (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', target_image)
            fps_time = time.time()

        # 更新窗口
        fps.update()
        # 打印画面延迟信息
        print('[INFO] elapsed time: {:.2f}'.format(time.time() - t))
        # 响应按键消息
        if cv2.waitKey(1) == 27:
            logger.debug('finished+')
            break

    fps.stop()
    print('[INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

    # When everything is done, release the capture
    video_capture.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
