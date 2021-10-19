"""
    online_pose_classification.py
    Author: Park Jaehun
    Purpose
        Online realtime inference for human pose classification
"""
#!/usr/bin/env python3

import os
import sys
import logging

import cv2

from model.Openvino import Openvino
from model.Openvino import build_argparser as openvino_args
from PoseEstimation import PoseEstimation
from utils.data_handler import DataHandler

def convert_to_openpose(data):
    return [data[0][0], data[0][1],
            (data[5][0]+data[6][0])/2, (data[5][1]+data[6][1])/2,
            data[6][0], data[6][1],
            data[8][0], data[8][1],
            data[10][0], data[10][1],
            data[5][0], data[5][1],
            data[7][0], data[7][1],
            data[9][0], data[9][1],
            (data[11][0]+data[12][0])/2, (data[11][1]+data[12][1])/2]

if __name__ == "__main__":
    args = openvino_args().parse_args()

    cap_width = 640
    cap_height = 480
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    model = Openvino(args, [cap_height, cap_width])
    pose_classifier = PoseEstimation()
    data_handler = DataHandler()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        kps_list = model.inference(frame)
        for kps in kps_list:
            kps = convert_to_openpose(kps)
            kps = data_handler.preprocess_data(kps)

            pose = pose_classifier.predict(kps)
            color = pose_classifier.pose_color(pose)
            print('pose : ', pose)
            out_frame = data_handler.draw_data(kps, color=color)
            cv2.putText(out_frame, pose_classifier.to_str(pose), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('test', out_frame)

        cv2.imshow('test2', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    sys.exit(0)