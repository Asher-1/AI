### This repo implements a custom toy detection model using the Tensorflow Object Detection API. 

一、添加自己的数据集步骤：
1.添加标注文件到annotations文件夹
2.添加图片到images文件夹
3.修改my_data文件夹下的own_data_label_map.pbtxt 添加自己的类别id 如下格式：
item {
  id: 1
  name: 'toy'
}

item {
  id: 2
  name: 'raccoon'
}

4.修改my_data文件夹下的faster_rcnn_inception_v2_coco.config的类别数目，即num_classes: ？


二、用自己的数据集制作tf-record数据：
1.运行xml_to_csv.py 生成toy_train_labels.csv和toy_val_labels.csv
2.运行generate_tfrecord.py生成toy_train.record和toy_val.record

三、开始训练：
1.运行 train.py
2.运行export_inference_graph.py生成frozen_inference_graph.pb
3.利用frozen_inference_graph.pb，运行object_detection_faster_rcnn_inception_v2_tutorial.py展示成果
（PS: 需要修改类别数目NUM_CLASSES = ?）