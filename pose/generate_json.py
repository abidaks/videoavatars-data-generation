#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from posenet import poseNet
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--location', '-l',
        default='../json_data',
        help='Path to save json data')

    parser.add_argument(
        '--video', '-v',
        required=True,
        help='Video File')

    args = parser.parse_args()
    
    app = poseNet(args.location, args.video)
    app.load_model()
    app.start()
