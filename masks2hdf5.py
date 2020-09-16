#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import cv2
import sys
import h5py
import argparse
import numpy as np

from glob import glob
from tqdm import tqdm

"""

This script stores image masks from a directory in a compressed hdf5 file.

Example:
$ python masks2hdf5.py ./mask_data ./data/masks.hdf5

"""

# You can run below commented code to verify your generated masks with the masks SMPL generated
# my code is not generating perfect masks, but it will work

'''
masks = h5py.File("../male2/masks.hdf5", 'r')['masks']
num_frames = masks.shape[0]

keypoints = masks[0]
blank_image = np.zeros((1080,1080), np.uint8)

keypoints[keypoints == 0] = 100

blank_image[:] = keypoints[:]

cv2.imwrite("smpl-mask.jpg", blank_image) 

masks = h5py.File("./data/masks.hdf5", 'r')['masks']
num_frames = masks.shape[0]

keypoints = masks[0]
blank_image = np.zeros((1080,1080), np.uint8)

keypoints[keypoints == 0] = 100

blank_image[:] = keypoints[:]

cv2.imwrite("my-mask.jpg", blank_image) 

exit()
'''

parser = argparse.ArgumentParser()
parser.add_argument('src', type=str)
parser.add_argument('target', type=str)

args = parser.parse_args()

out_file = args.target
mask_dir = args.src
mask_files = sorted(glob(os.path.join(mask_dir, '*.png')) + glob(os.path.join(mask_dir, '*.jpg')))

with h5py.File(out_file, 'w') as f:
    dset = None

    #for i, silh_file in enumerate(tqdm(mask_files)):
    for i in range(len(mask_files)):
        #print(i)
        silh_file = mask_dir +'/'+str(i)+'.jpg'
        silh = cv2.imread(silh_file, cv2.IMREAD_GRAYSCALE)

        if dset is None:
            dset = f.create_dataset("masks", (len(mask_files), silh.shape[0], silh.shape[1]), 'b', chunks=True, compression="lzf")

        _, silh = cv2.threshold(silh, 5, 255, cv2.THRESH_BINARY)
        dset[i] = silh.astype(np.bool)
        #print(dset[i])
