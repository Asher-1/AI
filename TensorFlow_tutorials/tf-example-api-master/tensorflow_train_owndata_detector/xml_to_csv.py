import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import random

train_csv_path = 'my_data/toy_train_labels.csv'
val_csv_path = 'my_data/toy_val_labels.csv'
# 验证集和数据集比例
train_val_rate = 0.7

def xml_to_csv(examples_list):
    xml_list = []
    for xml_file in examples_list:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'my_data', 'annotations')
    examples_list = glob.glob(image_path + '/*.xml')
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(train_val_rate * num_examples)
    train_examples_list = examples_list[:num_train]
    val_examples_list = examples_list[num_train:]
    # 转化训练集数据
    xml_df = xml_to_csv(train_examples_list)
    xml_df.to_csv(train_csv_path, index=None)
    print('Successfully converted xml to %s.' % train_csv_path)
    # 转化验证集数据
    xml_df = xml_to_csv(val_examples_list)
    xml_df.to_csv(val_csv_path, index=None)
    print('Successfully converted xml to %s.' % val_csv_path)


main()
