#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import cv2
import math
import numpy as np

KeyPointNames = {
  'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder',
  'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist',
  'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'
}

NUM_KEYPOINTS = len(KeyPointNames)

ConnectedKeyPointsNames = {
    'leftHipleftShoulder':(0,0,255), 'leftShoulderleftHip':(0,0,255),
    'leftElbowleftShoulder':(255,0,0), 'leftShoulderleftElbow':(255,0,0),
    'leftElbowleftWrist':(0,255,0), 'leftWristleftElbow':(0,255,0),
    'leftHipleftKnee':(0,0,255), 'leftKneeleftHip':(0,0,255),
    'leftKneeleftAnkle':(255,255,0), 'leftAnkleleftKnee':(255,255,0),
    'rightHiprightShoulder':(0,255,0), 'rightShoulderrightHip':(0,255,0),
    'rightElbowrightShoulder':(255,0,0), 'rightShoulderrightElbow':(255,0.0),
    'rightElbowrightWrist':(255,255,0), 'rightWristrightElbow':(255,255,0),
    'rightHiprightKnee':(255,0,0), 'rightKneerightHip':(255,0,0),
    'rightKneerightAnkle':(255,0,0), 'rightAnklerightKnee':(255,0,0),
    'leftShoulderrightShoulder':(0,255,0), 'rightShoulderleftShoulder':(0,255,0),
    'leftHiprightHip':(0,0,255), 'rightHipleftHip':(0,0,255)
}

poseChain = [
  ['nose', 'leftEye'], ['leftEye', 'leftEar'], ['nose', 'rightEye'],
  ['rightEye', 'rightEar'], ['nose', 'leftShoulder'],
  ['leftShoulder', 'leftElbow'], ['leftElbow', 'leftWrist'],
  ['leftShoulder', 'leftHip'], ['leftHip', 'leftKnee'],
  ['leftKnee', 'leftAnkle'], ['nose', 'rightShoulder'],
  ['rightShoulder', 'rightElbow'], ['rightElbow', 'rightWrist'],
  ['rightShoulder', 'rightHip'], ['rightHip', 'rightKnee'],
  ['rightKnee', 'rightAnkle']
]

confidence_threshold = 0.1
def drawKeypoints(body, img, color):
    for keypoint in body['keypoints']:
        if keypoint['score'] >= confidence_threshold:
            center = (int(keypoint['position']['x']), int(keypoint['position']['y']))
            radius = 3
            color = color
            cv2.circle(img, center, radius, color, -1, 8)
    return None

def drawPositions(body, img, color):
    count = 0
    for point in range(0, len(body), 3):
        center = (int(body[point]), int(body[point+1]))
        color = (0,0,255)
        if body[point+2] >= confidence_threshold:
            cv2.putText(img, str(count), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        count += 1

    return None


HeaderPart = {'nose', 'leftEye', 'leftEar', 'rightEye', 'rightEar'}
def drawSkeleton(body, img):
    valid_name = set()
    keypoints = body['keypoints']
    thickness = 2
    for idx in range(len(keypoints)):
        src_point = keypoints[idx]
        if src_point['part'] in HeaderPart or src_point['score'] < confidence_threshold:
            continue
        for dst_point in keypoints[idx:]:
            if dst_point['part'] in HeaderPart or dst_point['score'] < confidence_threshold:
                continue
            name = src_point['part'] + dst_point['part']
            def check_and_drawline(name):
                if name not in valid_name and name in ConnectedKeyPointsNames:
                    color = (255,255,0)#ConnectedKeyPointsNames[name]
                    cv2.line(img,
                             (int(src_point['position']['x']), int(src_point['position']['y'])),
                             (int(dst_point['position']['x']), int(dst_point['position']['y'])),
                             color, thickness)
                    valid_name.add(name)
            name = src_point['part'] + dst_point['part']
            check_and_drawline(name)
            name = dst_point['part'] + src_point['part']
            check_and_drawline(name)
    return None

# def drawSkeleton2(body, img):
#     valid_name = set()
#     keypoints = body['keypoints']
#     thickness = 2
#     for idx in range(len(keypoints)):
#         src_point = keypoints[idx]
#         if src_point['part'] in HeaderPart or src_point['score'] < confidence_threshold:
#             continue
#         for dst_point in keypoints[idx:]:
#             if dst_point['part'] in HeaderPart or dst_point['score'] < confidence_threshold:
#                 continue
#             name = src_point['part'] + dst_point['part']
#             def check_and_drawline(name):
#                 if name not in valid_name and name in ConnectedKeyPointsNames:
#                     color = (255,255,0)#ConnectedKeyPointsNames[name]
#                     cv2.line(img,
#                              (int(src_point['position']['x']), int(src_point['position']['y'])),
#                              (int(dst_point['position']['x']), int(dst_point['position']['y'])),
#                              color, thickness)
#                     valid_name.add(name)
                    
#                     x_origin ,y_origin , faceWidth , faceHight = getFaceDim(self, keypoints)
#             name = src_point['part'] + dst_point['part']
#             check_and_drawline(name)
#             name = dst_point['part'] + src_point['part']
#             check_and_drawline(name)
#     return None

def getFaceDim(self,keypoints):
        keypoints_ = keypoints["keypoints"]
        leftEar = keypoints_[3]
        leftEarCoordinates = leftEar["position"]
        rightEar = keypoints_[4]
        rightEarCoordinates = rightEar["position"]
        distBetweenEars = math.sqrt(abs((leftEarCoordinates["x"] - rightEarCoordinates["x"])**2 + (leftEarCoordinates["y"] - rightEarCoordinates["y"])**2))
        faceWidth = distBetweenEars * 1.25
        faceHight = faceWidth * 1.62
        nose = keypoints_[0]
        nosePosition = nose["position"]
        y_origin = nosePosition["y"] -  (faceHight / 2)
        x_origin = nosePosition["x"] - (faceWidth / 2)
       
        return x_origin ,y_origin , faceWidth , faceHight
        
