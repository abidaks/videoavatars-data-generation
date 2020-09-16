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
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

"""

This script stores image masks from a directory in a compressed hdf5 file.

Example:
$ python masks2hdf5.py dataset/subject/masks masks.hdf5

"""

# masks = h5py.File("../male/keypoints.hdf5", 'r')['keypoints']
# num_frames = masks.shape[0]

# keypoints = masks[0]
# count = 0
# blank_image = np.zeros((1300,1300,3), np.uint8)
# blank_image[:] = (0,0,0)

# for point in range(0, len(keypoints), 3):
#     print(point)
#     center = (int(keypoints[point]), int(keypoints[point+1]))
#     color = (0,0,255)
#     cv2.putText(blank_image, str(count), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
#     count += 1

# cv2.imwrite("image.jpg", blank_image) 

# print(num_frames)
# print(len(keypoints))


# masks = h5py.File("../new/masks.hdf5", 'r')['masks']
# num_frames = masks.shape[0]

# keypoints = masks[0]
# blank_image = np.zeros((1080,1080,3), np.uint8)
# print(keypoints[0])
# #blank_image[:] = keypoints[:]

# #cv2.imwrite("image.jpg", blank_image) 

# print(keypoints.shape)
# print(len(keypoints))



# #define the vertical filter
# vertical_filter = [[-1,-2,-1], [0,0,0], [1,2,1]]

# #define the horizontal filter
# horizontal_filter = [[-1,0,1], [-2,0,2], [-1,0,1]]

# #read in the pinwheel image
# img = plt.imread("../male/images/156.jpg")

# #get the dimensions of the image
# n,m,d = img.shape

# #initialize the edges image
# edges_img = img.copy()

# #loop over all pixels in the image
# for row in range(3, n-2):
#     for col in range(3, m-2):
        
#         #create little local 3x3 box
#         local_pixels = img[row-1:row+2, col-1:col+2, 0]
        
#         #apply the vertical filter
#         vertical_transformed_pixels = vertical_filter*local_pixels
#         #remap the vertical score
#         vertical_score = vertical_transformed_pixels.sum()/4
        
#         #apply the horizontal filter
#         horizontal_transformed_pixels = horizontal_filter*local_pixels
#         #remap the horizontal score
#         horizontal_score = horizontal_transformed_pixels.sum()/4
        
#         #combine the horizontal and vertical scores into a total edge score
#         edge_score = (vertical_score**2 + horizontal_score**2)**.5
        
#         #insert this edge score into the edges image
#         edges_img[row, col] = [edge_score]*3

# #remap the values in the 0-1 range in case they went out of bounds
# edges_img = edges_img/edges_img.max()

# cv2.imwrite("testing3.jpg", edges_img)

# imgplot = plt.imshow(edges_img)
# plt.show()

# exit()


masks = h5py.File("../male2/masks.hdf5", 'r')['masks']
num_frames = masks.shape[0]

keypoints = masks[0]
blank_image = np.zeros((1080,1080), np.uint8)

keypoints[keypoints == 0] = 100

blank_image[:] = keypoints[:]

cv2.imwrite("image.jpg", blank_image) 

# # print(keypoints.shape)
# # print(len(keypoints))

# keypoint = np.copy(keypoints)

# #print(type(keypoint))

# keypoint[keypoint == 0] = 100

# #print(keypoint[579])

# # img = cv2.imread("../male/images/156.jpg") #load rgb image
# # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert it to hsv
# # hsv[:,:,2] += 30
# # img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# # cv2.imwrite("image_processed.jpg", img)

# # exit()

# cv2.imwrite("testing.jpg", keypoint)

# print("here")
# exit()
# with h5py.File("test.hdf5", 'w') as f:
# 	dset = f.create_dataset("masks", (2, 1080, 1080), 'b', chunks=True, compression="lzf")

# 	# silh = cv2.imread("../male/images/156.png", cv2.IMREAD_GRAYSCALE)
# 	# #print(silh[579])
# 	# a, silh = cv2.threshold(silh, 93, 255, cv2.THRESH_BINARY)

	
# 	# dset[0] = silh.astype(np.bool)
# 	# #print(dset[0][579])

# 	# keypoint = np.copy(dset[0])
# 	# keypoint[keypoint == 0] = 100
# 	# cv2.imwrite("testing3.png", keypoint)

# 	silh = cv2.imread("../male2/images/156.jpg")
# 	for i in range(1080):
# 		for j in range(1080):
# 			if silh[i][j][0] < 80 and silh[i][j][0] > 28 \
# 				and silh[i][j][1] > 75 and silh[i][j][1] < 130 \
# 				and silh[i][j][2] < 80 and silh[i][j][2] > 28:
# 				silh[i][j][0] = 0
# 				silh[i][j][1] = 0
# 				silh[i][j][2] = 0

# 	cv2.imwrite("testing3.png", silh)

# 	#print(silh[100])
silh = cv2.imread("mask_data/0.jpg", cv2.IMREAD_GRAYSCALE)
a, silh = cv2.threshold(silh, 10, 255, cv2.THRESH_BINARY)
silh[silh == 0] = 100
silh[silh == 255] = 0
cv2.imwrite("0.jpg", silh)
print(silh[500])
exit()

parser = argparse.ArgumentParser()
parser.add_argument('src', type=str)
parser.add_argument('target', type=str)

args = parser.parse_args()

out_file = args.target
mask_dir = args.src
mask_files = sorted(glob(os.path.join(mask_dir, '*.png')) + glob(os.path.join(mask_dir, '*.jpg')))

with h5py.File(out_file, 'w') as f:
    dset = None

    for i, silh_file in enumerate(tqdm(mask_files)):
        silh = cv2.imread(silh_file, cv2.IMREAD_GRAYSCALE)

        if dset is None:
            dset = f.create_dataset("masks", (len(mask_files), silh.shape[0], silh.shape[1]), 'b', chunks=True, compression="lzf")

        _, silh = cv2.threshold(silh, 100, 255, cv2.THRESH_BINARY)
        dset[i] = silh.astype(np.bool)
        #print(dset[i])
