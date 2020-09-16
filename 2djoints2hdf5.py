#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import cv2
import h5py
import json
import argparse
import numpy as np

from glob import glob
from tqdm import tqdm


"""
This script stores OpenPose 2D keypoints from json files in the given directory in a compressed hdf5 file.

Example:
$ python 2djoints2hdf5.py ./json_data ./data/keypoints.hdf5

"""

def writepoints(filename, keypoints):
	count = 0
	blank_image = np.zeros((1080,1080,3), np.uint8)
	blank_image[:] = (0,0,0)

	for point in range(0, len(keypoints), 3):
		center = (int(keypoints[point]), int(keypoints[point+1]))
		color = (0,0,255)
		cv2.putText(blank_image, str(count), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
		count += 1
		# print(count)
		# print(keypoints[point+2])
		# print(center)
		# print("-------------------")

	cv2.imwrite(filename, blank_image)

# You can run the below code to verify your code with SMPL data

'''
masks = h5py.File("../male2/keypoints.hdf5", 'r')['keypoints']
keypoints = masks[0]
writepoints("smpl-keypoints.jpg", keypoints)

masks = h5py.File("./data/keypoints.hdf5", 'r')['keypoints']
keypoints = masks[0]
writepoints("my-keypoints.jpg", keypoints)

exit()

'''

parser = argparse.ArgumentParser()
parser.add_argument('src_folder', type=str)
parser.add_argument('target', type=str)

args = parser.parse_args()

out_file = args.target
pose_dir = args.src_folder
pose_files = sorted(glob(os.path.join(pose_dir, '*.json')))
#print(pose_files)

#print(os.listdir(pose_dir))

with h5py.File(out_file, 'w') as f:
    poses_dset = f.create_dataset("keypoints", (len(pose_files), 54), 'f', chunks=True, compression="lzf")

    for i in range(len(pose_files)):
        pose_file = pose_dir +'/'+str(i)+'.json'
        with open(pose_file) as fp:
            pose = np.array(json.load(fp))
            poses_dset[i] = pose
