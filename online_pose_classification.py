"""
    online_pose_classification.py
    Author: Park Jaehun
    Purpose
        Online inference for human pose classification
"""
#!/usr/bin/env python3

import os
import sys
import logging
from time import perf_counter

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

def isNotValid(data, prob_threshold):
    check_list = [0, 5, 6, 7, 8, 9, 10, 11, 12]
    for idx in check_list:
        d = data[idx]
        if d[0] == 0.0 or d[1] == 0.0 or d[2] < prob_threshold:
            return True
    return False

def clear_buf(buf):
    for i in range(len(buf)):
        buf[i] = 0

def find_max(buf):
    max_pose = 0
    for i in range(len(buf)):
        if buf[max_pose] <= buf[i]:
            max_pose = i
    return max_pose

if __name__ == "__main__":
    args = openvino_args().parse_args()

    prob_threshold = args.prob_threshold

    cap_width = 640
    cap_height = 480
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    model = Openvino(args, [cap_height, cap_width])
    pose_classifier = PoseEstimation()
    data_handler = DataHandler()

    pose_buf = [0] * 8

    start_time = perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        kps_list = model.inference(frame)
        for kps in kps_list:
            if isNotValid(kps, prob_threshold):
                print("is not valid")
                continue
            kps = convert_to_openpose(kps)
            kps = data_handler.preprocess_data(kps)

            pose = pose_classifier.predict(kps)
            color = pose_classifier.pose_color(pose)

            pose_buf[pose] += 1

            out_frame = data_handler.draw_data(kps, color=color)
            cv2.putText(out_frame, pose_classifier.to_str(pose), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('test', out_frame)

        if perf_counter() - start_time > 3:
            max_pose = find_max(pose_buf)
            cv2.putText(frame, pose_classifier.to_str(max_pose), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2, cv2.LINE_AA)
            clear_buf(pose_buf)
            start_time = perf_counter()

        cv2.imshow('test2', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    sys.exit(0)