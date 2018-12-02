import sys
import os
sys.path.append("..")

import numpy as np
import json
import pandas as pd

import h5py
import cv2
from config import GetConfig
from addins.head_counter_config import HeadCounterConfig
from model import get_testing_model
from glob import glob

toloka_dir = os.path.abspath('/opt/home/anatolix/YaDisk/Приложения/Яндекс.Толока')
results_dir_mask = os.path.join(toloka_dir, "Results/Heads/assignments_*.tsv.csv")
img_dir = os.path.join(toloka_dir, "Yandex.Toloka/pochta")


task = sys.argv[1]
assert task == "predict" or task == "render"
config_name = sys.argv[2]
model = sys.argv[3]

config = GetConfig(config_name)

val_size = 19

val_ids = None
train_ids = None


def prepare(config, model_file):

    model = get_testing_model(np_branch1=config.paf_layers, np_branch2=config.heat_layers + 1)

    print("using model:", model_file)
    model.load_weights(model_file)

    return model

def load_results():

    global val_ids
    global train_ids

    files = glob(results_dir_mask)
    data = [pd.read_csv(file, sep="\t", skip_blank_lines=True) for file in files]
    data = pd.concat(data)
    data = data[['INPUT:image', 'OUTPUT:result']]
    data = data.dropna()

    camera_sets = data['INPUT:image'].str.split('/', expand=True)[3].unique()
    camera_sets = np.sort(camera_sets)

    assert '-' in camera_sets[0]

    val_ids = camera_sets[:val_size]
    train_ids = camera_sets[val_size:]

    print(val_ids)
    print(train_ids)

    return data.itertuples()


def load_resize_images(config, scales = (0.5, 1, 1.5, 2)):

    for n, img_id, head_rects in load_results():

        raw_anno = json.loads(head_rects)

        assert img_id[:19] == '/YandexDisk/pochta/'
        img_id = img_id[19:]
        img_set = img_id.split('/')[0]


        img_path = os.path.join(img_dir, img_id)

        orig_img = cv2.imread(img_path)
        h, w, c = orig_img.shape

        multipliers = [x * config.height / w for x in scales]

        results = []

        for m in multipliers:

            new_height = int(h * m)
            new_width = int(w * m)

            if new_height % config.stride != 0:
                new_height += 8 - (new_height % config.stride)

            if new_width % config.stride != 0:
                new_width += 8 - (new_width % config.stride)

            M = np.array([[m,0.,0.],[0.,m,0.]])

            img = cv2.warpAffine(orig_img, M, (new_height, new_width), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))

            img = np.transpose(np.float32(img[:, :, :, np.newaxis]), (3, 0, 1, 2))

            results += [img]

        print(img_id, [r.shape for r in results] )

        raw_anno = {'toloka': raw_anno, 'validation':img_set in val_ids }

        yield img_id, results, raw_anno



model = prepare(config, model)

with h5py.File("results.h5", 'w') as h5:

    for name, imgs, annotation in load_resize_images(config):
        for n,i in enumerate(imgs):
            blobs = model.predict(i)
            blobs = np.concatenate(blobs, axis=3)
            ds = h5.create_dataset(name + "/image" + str(n), data=i, chunks=None)
            ds = h5.create_dataset(name + "/res" + str(n), data=blobs, chunks=None)
            ds.attrs['meta'] = json.dumps(annotation)
            print(name)


