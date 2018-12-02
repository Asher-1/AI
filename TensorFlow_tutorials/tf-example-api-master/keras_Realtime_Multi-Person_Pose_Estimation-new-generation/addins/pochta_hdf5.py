#!/usr/bin/env python


import glob
import os
import pandas as pd
import numpy as np
import json
import cv2
import h5py

dataset_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'dataset'))
toloka_dir = os.path.abspath('/opt/home/anatolix/YaDisk/Приложения/Яндекс.Толока')
results_dir_mask = os.path.join(toloka_dir, "Results/Heads/assignments_*.tsv.csv")
img_dir = os.path.join(toloka_dir, "Yandex.Toloka/pochta")

tr_hdf5_path = os.path.join(dataset_dir, "pochta_train_dataset.h5")
val_hdf5_path = os.path.join(dataset_dir, "pochta_val_dataset.h5")

val_size = 19

val_ids = None
train_ids = None

def load_results():

    global val_ids
    global train_ids

    files = glob.glob(results_dir_mask)
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

def load_image(img_dir, img_id):

    img_path = os.path.join(img_dir, img_id)
    img = cv2.imread(img_path)

    return img

def process_image(raw_anno, pic_width, pic_height):

    results = { 'joints':[], 'scale_provided':[], 'objpos':[], 'head':[] }

    for item in raw_anno:

        if item['type']=='rectangle':
            p1 = item['data']['p1']
            p2 = item['data']['p2']
            points = [p1,p2]
        elif item['type']=='polygon':
            points = item['data']
        else:
            assert False, item['type']

        Xs = [p['x'] for p in points]
        Ys = [p['y'] for p in points]

        min_x = min(Xs) * pic_width
        max_x = max(Xs) * pic_width
        min_y = min(Ys) * pic_height
        max_y = max(Ys) * pic_height

        if max_x-min_x<5 or max_y-min_y<5:
            continue

        results['objpos'].append([(min_x+max_x)/2., (min_y+max_y)/2.])
        results['joints'].append([[(min_x + max_x) / 2., (min_y + max_y) / 2., 1.]])
        results['head'].append([min_x, min_y, max_x-min_x, max_y-min_y])
        results['scale_provided'].append((max_y-min_y)*4/368)  # why 4 - xz, looks nice

    for pp in range(len(results['joints'])):

        yield_item = {}

        yield_item['scale_provided'] = results['scale_provided'][pp:] + results['scale_provided'][:pp]
        yield_item['joints'] = results['joints'][pp:] + results['joints'][:pp]
        yield_item['objpos'] = results['objpos'][pp:] + results['objpos'][:pp]
        yield_item['head'] = results['head'][pp:] + results['head'][:pp]

        yield yield_item


def writeImage(grp, img_grp, raw_anno, processed_anno, img, count, image_id):

    raw_anno = { 'toloka':raw_anno }
    raw_anno['count'] = count
    raw_anno['image'] = image_id

    #print(anno)

    if not image_id in img_grp:
        print('Writing image %s' % image_id)
        _, compressed_image = cv2.imencode(".jpg", img)
        img_ds = img_grp.create_dataset(image_id, data=compressed_image, chunks=None)

    key = '%07d' % count
    processed_anno['image'] = image_id

    ds = grp.create_dataset(key, data=json.dumps(processed_anno), chunks=None)
    ds.attrs['meta'] = json.dumps(raw_anno)

    print('Writing sample %d' % count)



def process():

    tr_h5 = h5py.File(tr_hdf5_path, 'w')
    tr_grp = tr_h5.create_group("dataset")
    tr_write_count = 0
    tr_grp_img = tr_h5.create_group("images")

    val_h5 = h5py.File(val_hdf5_path, 'w')
    val_grp = val_h5.create_group("dataset")
    val_write_count = 0
    val_grp_img = val_h5.create_group("images")

    count = 0

    img = None
    cached_img_id = None
    img_set = None

    for n, img_id, head_rects in load_results():

        raw_anno = json.loads(head_rects)

        assert img_id[:19] == '/YandexDisk/pochta/'
        img_id = img_id[19:]

        if cached_img_id != img_id:
            cached_img_id = img_id
            img = load_image(img_dir, cached_img_id)
            img_set = img_id.split('/')[0]

        h, w, c = img.shape

        for processed in process_image(raw_anno, w, h):
            count += 1

            if img_set in val_ids:
                writeImage(val_grp, val_grp_img, raw_anno, processed, img, val_write_count, cached_img_id)
                val_write_count += 1
            elif img_set in train_ids:
                writeImage(tr_grp, tr_grp_img, raw_anno, processed, img, tr_write_count, cached_img_id)
                tr_write_count += 1
            else:
                assert False, img_set

    tr_h5.close()
    val_h5.close()

    print("Total:", count)


if __name__ == '__main__':
    process()
