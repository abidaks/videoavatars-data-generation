#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import os
import cv2
import yaml
import time
import json
import math
import struct
import numpy as np
import tensorflow.compat.v1 as tf

from decode_multi_pose import decodeMultiplePoses
from draw import drawKeypoints, drawSkeleton, drawPositions

tf.disable_v2_behavior()

color_table = [(0,255,0), (255,0,0), (0,0,255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]

isUser = False

userName = ""

soundStartTime = time.time()
firstSoundPlay = True
helloUser = False
helloUserStartTime = time.time()
moving_commands = ["walk" , "dance" , "jump"]
soundsDurations = {"hello" : 2 , "notUser" : 3 , "onePerson" : 4 , "again" : 3}

class poseNet:
    def __init__(self, loc, video):
        self.file_location = loc
        self.video_file = video
        with open('config.yaml') as f:
            cfg = yaml.load(f)
            checkpoints = cfg['checkpoints']
            imageSize = cfg['imageSize']
            chk = cfg['chk']
            self.outputStride = cfg['outputStride']
            chkpoint = checkpoints[chk]
            self.mobileNetArchitectures = self.architecture(chkpoint, cfg)
            self.width = imageSize
            self.height = imageSize
            self.layers = self.toOutputStridedLayers()
            self.variables(chkpoint)
            
            self.wav_obj = None
            self.tobeplayed = []
            self.waitTime = 2;
            self.waitTimeCorordin = 2;
            self.startAvatar = True

    def variables(self, chkpoint):
        with open(os.path.join('./waits/', chkpoint, "manifest.json")) as f:
            self.variables = json.load(f)
            # with tf.variable_scope(None, 'MobilenetV1'):
            for x in self.variables:
                filename = self.variables[x]["filename"]
                with open(os.path.join('./waits/', chkpoint, filename), 'rb') as fp:
                    byte = fp.read()
                    fmt = str(int(len(byte) / struct.calcsize('f'))) + 'f'
                    d = struct.unpack(fmt, byte)
                    d = tf.cast(d, tf.float32)
                    d = tf.reshape(d, self.variables[x]["shape"])
                    self.variables[x]["x"] = tf.Variable(d, name=x)
        return None
    
    def architecture(self, chkpoint, cfg):
        if chkpoint == 'mobilenet_v1_050':
            mobileNetArchitectures = cfg['mobileNet50Architecture']
        elif chkpoint == 'mobilenet_v1_075':
            mobileNetArchitectures = cfg['mobileNet75Architecture']
        else:
            mobileNetArchitectures = cfg['mobileNet100Architecture']
        return mobileNetArchitectures

    def toOutputStridedLayers(self):
        currentStride = 1
        rate = 1
        blockId = 0
        buff = []
        for _a in self.mobileNetArchitectures:
            convType = _a[0]
            stride = _a[1]
            if (currentStride == self.outputStride):
                layerStride = 1
                layerRate = rate
                rate *= stride
            else:
                layerStride = stride
                layerRate = 1
                currentStride *= stride
            buff.append({'blockId': blockId, \
                         'convType': convType, \
                         'stride': layerStride, \
                         'rate': layerRate, \
                         'outputStride': currentStride})
            blockId += 1
        return buff

    def convToOutput(self, mobileNetOutput, outputLayerName):
        w = tf.nn.conv2d(mobileNetOutput, \
                         self.weights(outputLayerName), \
                         [1,1,1,1], padding='SAME')
        w = tf.nn.bias_add(w, self.biases(outputLayerName), name=outputLayerName)
        return w

    def conv(self, inputs, stride, blockId):
        return tf.nn.relu6(
            tf.nn.conv2d(inputs, \
                         self.weights("Conv2d_" + str(blockId)), \
                         stride, padding='SAME')
            + self.biases("Conv2d_" + str(blockId)))

    def weights(self, layerName):
        return self.variables["MobilenetV1/" + layerName + "/weights"]['x']

    def biases(self, layerName):
        return self.variables["MobilenetV1/" + layerName + "/biases"]['x']

    def depthwiseWeights(self, layerName):
        return self.variables["MobilenetV1/" + layerName + "/depthwise_weights"]['x']

    def separableConv(self, inputs, stride, blockID, dilations):
        if (dilations == None):
            dilations = [1,1]
        dwLayer = "Conv2d_" + str(blockID) + "_depthwise"
        pwLayer = "Conv2d_" + str(blockID) + "_pointwise"
        w = tf.nn.depthwise_conv2d(inputs, \
                                   self.depthwiseWeights(dwLayer), \
                                   stride, 'SAME', rate=dilations, data_format='NHWC')
        w = tf.nn.bias_add(w, self.biases(dwLayer))
        w = tf.nn.relu6(w)
        w = tf.nn.conv2d(w, self.weights(pwLayer), [1,1,1,1], padding='SAME')
        w = tf.nn.bias_add(w, self.biases(pwLayer))
        w = tf.nn.relu6(w)

        return w


    
    def get_only_poses_with_valid_score(self,poses):
        """
        Parameters
        ----------
        poses : list
            All poses list.

        Returns
        -------
        successPoses : list
            array of pose with score is > 0.2.

        """
        successPoses = []
        for pose in poses:
            if pose["score"] > 0.2:
                successPoses.append(pose)
        return successPoses


    
    def getWeights(self):
        weights = np.ones((1,2,17))
        
        weights[0][0][0] = 0
        weights[0][1][0] = 0
        weights[0][0][7] = 0
        weights[0][1][7] = 0
        weights[0][0][9] = 0
        weights[0][1][9] = 0
    
        return weights

    def load_model(self):
        weights = self.getWeights()
        self.image = tf.placeholder(tf.float32, shape=[1, self.width, self.height, 3],name='image')
        x = self.image
        rate = [1,1]
        buff = []
        with tf.variable_scope(None, 'MobilenetV1'):
            for m in self.layers:
                strinde = [1,m['stride'],m['stride'],1]
                rate = [m['rate'],m['rate']]
                if (m['convType'] == "conv2d"):
                    x = self.conv(x,strinde,m['blockId'])
                    buff.append(x)
                elif (m['convType'] == "separableConv"):
                    x = self.separableConv(x,strinde,m['blockId'],rate)
                    buff.append(x)
        self.heatmaps = self.convToOutput(x, 'heatmap_2')
        self.offsets = self.convToOutput(x, 'offset_2')
        self.displacementFwd = self.convToOutput(x, 'displacement_fwd_2')
        self.displacementBwd = self.convToOutput(x, 'displacement_bwd_2')
        self.heatmaps = tf.sigmoid(self.heatmaps, 'heatmap')


    def start(self):
        isUser = False

        cap = cv2.VideoCapture(self.video_file) # 0 Read camera
        cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width_factor =  cap_width/self.width
        height_factor = cap_height/self.height
        
        config = tf.ConfigProto(device_count = {'GPU': 0})

        with tf.Session(config = config) as sess:

            init = tf.compat.v1.global_variables_initializer()#tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()
            save_dir = './checkpoints'
            save_path = os.path.join(save_dir, 'model.ckpt')
            saver.save(sess, save_path)
            flag, frame = cap.read()
            count = 0
            drawnsk = 0
            addedNotUser = False
            recognizeAgain = False
            startime = time.time()
            first_check = True

            coordinatesTime = None 
            startcoordinatesTime = time.time()

            leftHandpoint = []
            rectCenters = []
            objects = []

            leftHandT = []
            rightHandT = []
            
            while flag:

                orig_image = frame
                frame = cv2.resize(frame, (self.width, self.height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(float)
                frame = frame * (2.0 / 255.0) - 1.0
                frame = np.array(frame, dtype=np.float32)
                frame = frame.reshape(1, self.width, self.height, 3)
                heatmaps_result, offsets_result, displacementFwd_result, displacementBwd_result \
                    = sess.run([self.heatmaps, \
                                self.offsets, \
                                self.displacementFwd, \
                                self.displacementBwd], feed_dict={self.image: frame } )
                '''
                poses = decode_single_pose(heatmaps_result, offsets_result, 16, width_factor, height_factor)
                '''
                poses = decodeMultiplePoses(heatmaps_result, offsets_result, \
                                            displacementFwd_result, \
                                            displacementBwd_result, \
                                            width_factor, height_factor)
        
                x = self.get_only_poses_with_valid_score(poses)

                keypoints = []

                if len(x) > 0:
                    color = color_table[0]
                    b = np.array(x[0]['keypoints'])
                    keypoints = [b[0]['position']['x'], b[0]['position']['y'], b[0]['score'], \
                                    ((b[5]['position']['x'] + b[6]['position']['x'])/2), ((b[5]['position']['y'] + b[6]['position']['y'])/2), b[6]['score'], \
                                    b[6]['position']['x'], b[6]['position']['y'], b[6]['score'], \
                                    b[8]['position']['x'], b[8]['position']['y'], b[8]['score'], \
                                    b[10]['position']['x'], b[10]['position']['y'], b[10]['score'], \
                                    b[5]['position']['x'], b[5]['position']['y'], b[5]['score'], \
                                    b[7]['position']['x'], b[7]['position']['y'], b[7]['score'], \
                                    b[9]['position']['x'], b[9]['position']['y'], b[9]['score'], \
                                    b[12]['position']['x'], b[12]['position']['y'], b[12]['score'], \
                                    b[14]['position']['x'], b[14]['position']['y'], b[14]['score'], \
                                    b[16]['position']['x'], b[16]['position']['y'], b[16]['score'], \
                                    b[11]['position']['x'], b[11]['position']['y'], b[11]['score'], \
                                    b[13]['position']['x'], b[13]['position']['y'], b[13]['score'], \
                                    b[15]['position']['x'], b[15]['position']['y'], b[15]['score'], \
                                    b[2]['position']['x'], b[2]['position']['y'], b[2]['score'], \
                                    b[1]['position']['x'], b[1]['position']['y'], b[1]['score'], \
                                    b[4]['position']['x'], b[4]['position']['y'], b[4]['score'], \
                                    b[3]['position']['x'], b[3]['position']['y'], b[3]['score']]

                    drawPositions(keypoints, orig_image, color)
                        
                cv2.imshow("Hand Gesture Tracking", orig_image)

                flag, frame = cap.read()

                mkey = cv2.waitKey(1)
                countr = 0
                
                # I have to make face points as saved in the keypoints of the test data.
                # If probability is less than 0.12 then the values are zero.
                if keypoints[2] < 0.2:
                    print(str(keypoints[0]) + " - "+str(keypoints[1])  + " - "+str(keypoints[2]) )
                    print(str(keypoints[3]) + " - "+str(keypoints[4])  + " - "+str(keypoints[5]) )
                
                for point in range(0, len(keypoints), 3):
                    if count > 13 and keypoints[point+2] < 0.12:
                        keypoints[point] = 0
                        keypoints[point+1] = 0
                        keypoints[point+2] = 0.0
                    
                    center = (int(keypoints[point]), int(keypoints[point+1]))
                    color = (0,0,255)
                    cv2.putText(orig_image, str(countr), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    countr += 1


                with open(self.file_location + '/'+str(count)+'.json', 'w') as f:
                    json.dump([np.asscalar(a) if isinstance(a, np.float32) else a for a in keypoints], f)

                count = count + 1

                if mkey == ord('q'):
                    flag = False
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        
