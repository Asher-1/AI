#!/usr/bin/env python

import numpy as np
import cv2
import os
import os.path
import h5py
import json


dataset_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'dataset'))
brainwash_dir = '/opt/home/anatolix/iidf-data/brainwash'
jsons = [ os.path.join(brainwash_dir, j) for j in ('test_boxes.json', 'val_boxes.json', 'train_boxes.json') ]
jsons_val_split = [True, True, False]

tr_hdf5_path = os.path.join(dataset_dir, "brainwash_train_dataset.h5")
val_hdf5_path = os.path.join(dataset_dir, "brainwash_val_dataset.h5")


def load_results():

    for n, f in enumerate(jsons):

        val = jsons_val_split[n]

        with open(f) as file:
            data = json.load(file)

        for d in data:
            yield d['image_path'], d, val


def load_image(img_dir, img_id):

    img_path = os.path.join(img_dir, img_id)
    img = cv2.imread(img_path)

    return img


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

def process_image(raw_anno):

    results = { 'joints':[], 'scale_provided':[], 'objpos':[], 'head':[] }

    for item in raw_anno['rects']:

        min_x = item['x1']
        max_x = item['x2']
        min_y = item['y1']
        max_y = item['y2']

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

    for img_id, raw_anno, val in load_results():

        if cached_img_id != img_id:
            cached_img_id = img_id
            img = load_image(brainwash_dir, cached_img_id)

        for processed in process_image(raw_anno):
            count += 1

            if val:
                writeImage(val_grp, val_grp_img, raw_anno, processed, img, val_write_count, cached_img_id)
                val_write_count += 1
            else:
                writeImage(tr_grp, tr_grp_img, raw_anno, processed, img, tr_write_count, cached_img_id)
                tr_write_count += 1

    tr_h5.close()
    val_h5.close()

    print("Total:", count)



if __name__ == '__main__':
    process()
