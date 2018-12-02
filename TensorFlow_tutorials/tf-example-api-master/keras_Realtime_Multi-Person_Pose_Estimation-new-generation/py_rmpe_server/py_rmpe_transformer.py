#!/usr/bin/env python

import numpy as np
from math import cos, sin, pi
import cv2
import random

class AugmentSelection:

    def __init__(self, flip=False, degree = 0., crop = (0,0), scale = 1.):
        self.flip = flip
        self.degree = degree #rotate
        self.crop = crop #shift actually
        self.scale = scale

    @staticmethod
    def random(transform_params):
        flip = random.uniform(0.,1.) > transform_params.flip_prob
        degree = random.uniform(-1.,1.) * transform_params.max_rotate_degree
        scale = (transform_params.scale_max - transform_params.scale_min)*random.uniform(0.,1.)+transform_params.scale_min \
            if random.uniform(0.,1.) > transform_params.scale_prob else 1. # TODO: see 'scale improbability' TODO above
        x_offset = int(random.uniform(-1.,1.) * transform_params.center_perterb_max);
        y_offset = int(random.uniform(-1.,1.) * transform_params.center_perterb_max);

        return AugmentSelection(flip, degree, (x_offset,y_offset), scale)

    @staticmethod
    def unrandom():
        flip = False
        degree = 0.
        scale = 1.
        x_offset = 0
        y_offset = 0

        return AugmentSelection(flip, degree, (x_offset,y_offset), scale)

    def affine(self, center, scale_self, config):

        # the main idea: we will do all image transformations with one affine matrix.
        # this saves lot of cpu and make code significantly shorter
        # same affine matrix could be used to transform joint coordinates afterwards


        A = self.scale * cos(self.degree / 180. * pi )
        B = self.scale * sin(self.degree / 180. * pi )

        scale_size = config.transform_params.target_dist / scale_self * self.scale

        (width, height) = center
        center_x = width + self.crop[0]
        center_y = height + self.crop[1]

        center2zero = np.array( [[ 1., 0., -center_x],
                                 [ 0., 1., -center_y ],
                                 [ 0., 0., 1. ]] )

        rotate = np.array( [[ A, B, 0 ],
                           [ -B, A, 0 ],
                           [  0, 0, 1. ] ])

        scale = np.array( [[ scale_size, 0, 0 ],
                           [ 0, scale_size, 0 ],
                           [  0, 0, 1. ] ])

        flip = np.array( [[ -1 if self.flip else 1., 0., 0. ],
                          [ 0., 1., 0. ],
                          [ 0., 0., 1. ]] )

        center2center = np.array( [[ 1., 0., config.width//2],
                                   [ 0., 1., config.height//2 ],
                                   [ 0., 0., 1. ]] )

        # order of combination is reversed
        combined = center2center.dot(flip).dot(scale).dot(rotate).dot(center2zero)

        return combined[0:2]

class Transformer:

    def __init__(self, config):

        self.config = config

    def transform(self, img, mask, meta, aug = None):

        if aug is None:
            aug = AugmentSelection.random(self.config.transform_params)

        # warp picture and mask
        M = aug.affine(meta['objpos'][0], meta['scale_provided'][0], self.config)

        # TODO: need to understand this, scale_provided[0] is height of main person divided by 368, caclulated in generate_hdf5.py
        # print(img.shape)
        img = cv2.warpAffine(img, M, (self.config.height, self.config.width), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(127,127,127))
        mask = cv2.warpAffine(mask, M, (self.config.height, self.config.width), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        mask = cv2.resize(mask, self.config.mask_shape, interpolation=cv2.INTER_CUBIC)  # TODO: should be combined with warp for speed
        mask = mask.astype(np.float) / 255.

        # warp key points
        #TODO: joint could be cropped by augmentation, in this case we should mark it as invisible.
        #update: may be we don't need it actually, original code removed part sliced more than half totally, may be we should keep it
        original_points = meta['joints'].copy()
        original_points[:,:,2]=1  # we reuse 3rd column in completely different way here, it is hack
        converted_points = np.matmul(M, original_points.transpose([0,2,1])).transpose([0,2,1])
        meta['joints'][:,:,0:2]=converted_points

        # we just made image flip, i.e. right leg just became left leg, and vice versa

        if aug.flip:
            tmpLeft = meta['joints'][:, self.config.leftParts, :]
            tmpRight = meta['joints'][:, self.config.rightParts, :]
            meta['joints'][:, self.config.leftParts, :] = tmpRight
            meta['joints'][:, self.config.rightParts, :] = tmpLeft


        return img, mask, meta

