"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2

train_csv_input = 'my_data/toy_train_labels.csv'
val_csv_input = 'my_data/toy_val_labels.csv'
train_output_path = 'my_data/toy_train.record'
val_output_path = 'my_data/toy_val.record'
m_label_map_path = 'my_data/own_data_label_map.pbtxt'

flags = tf.app.flags
flags.DEFINE_string('train_csv_input', train_csv_input, 'Path to the train CSV input')
flags.DEFINE_string('train_output_path', train_output_path, 'Path to train output TFRecord')
flags.DEFINE_string('val_csv_input', val_csv_input, 'Path to the val CSV input')
flags.DEFINE_string('val_output_path', val_output_path, 'Path to val output TFRecord')
flags.DEFINE_string('m_label_map_path', m_label_map_path, 'Path to label_map.pbtxt')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label, label_map, use_display_name=False):
    label_map_dict = {}
    for key in label_map.keys():
        if row_label == key:
            return label_map[key]
        else:
            None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(label_map, group, path):
    file_path = os.path.join(path, '{}'.format(group.filename.replace('png', 'jpg')))
    with tf.gfile.GFile(file_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class'], label_map))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def _validate_label_map(label_map):
    """Checks if a label map is valid.

    Args:
      label_map: StringIntLabelMap to validate.

    Raises:
      ValueError: if label map is invalid.
    """
    for item in label_map.item:
        if item.id < 0:
            raise ValueError('Label map ids should be >= 0.')
        if item.id == 0 and item.name != 'background':
            raise ValueError('Label map id 0 is reserved for the background label')


def get_label_map_dict(label_map_path, use_display_name=False):
    """Reads a label map and returns a dictionary of label names to id.

    Args:
      label_map_path: path to label_map.
      use_display_name: whether to use the label map items' display names as keys.

    Returns:
      A dictionary mapping label names to id.
    """

    with tf.gfile.GFile(label_map_path, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)
    _validate_label_map(label_map)

    label_map_dict = {}
    for item in label_map.item:
        if use_display_name:
            label_map_dict[item.display_name] = item.id
        else:
            label_map_dict[item.name] = item.id
    return label_map_dict


def main(_):
    input_list = [FLAGS.train_csv_input, FLAGS.val_csv_input]
    csv_output_path_list = [FLAGS.train_output_path, FLAGS.val_output_path]
    for csv_input, output_path in zip(input_list, csv_output_path_list):

        writer = tf.python_io.TFRecordWriter(output_path)
        path = os.path.join(os.getcwd(), 'my_data', 'images')

        examples = pd.read_csv(csv_input)
        grouped = split(examples, 'filename')
        # 读取标签类别图
        label_map_dict = get_label_map_dict(FLAGS.m_label_map_path)
        for group in grouped:
            tf_example = create_tf_example(label_map_dict, group, path)
            writer.write(tf_example.SerializeToString())

        writer.close()
        output_path = os.path.join(os.getcwd(), output_path)
        print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
