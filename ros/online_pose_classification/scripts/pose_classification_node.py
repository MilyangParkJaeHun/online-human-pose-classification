#!/usr/bin/env python3
"""
    pose_classification_node.py
    Author: Park Jaehun
    Purpose
        ROS node for online human pose classification
    Pub Topic Info
        data type : Int16
        name : /manta/vision/pose
"""
# pose list : /path/to/online_pose_classification/config

import os
import sys
import logging
import cv2
import numpy as np
from typing import List
from time import perf_counter

POSE_CLF_PATH=os.environ["POSE_CLF_PATH"]
sys.path.append(POSE_CLF_PATH)

from kps_extraction.data_handler import DataHandler
from pose_classification.pose_classifier import PoseClassifier
from online_pose_classification import build_argparser
from online_pose_classification import OnlinePoseClassfier
from openpose_ros_msgs.msg import OpenPoseHumanList
from openpose_ros_msgs.msg import OpenPoseHuman

import rospy
from std_msgs.msg import Int16

args = build_argparser().parse_args()
prob_threshold = args.prob_threshold
interval = args.interval

data_handler = DataHandler()
pose_classifier = PoseClassifier(model_path=args.classifier_model)
online_pose_clf = OnlinePoseClassfier(pose_classifier, data_handler, prob_threshold)

start_time = perf_counter()

frame = None
pub = rospy.Publisher('/manta/vision/pose', Int16, queue_size=10)

def callback(data):
    """
    Run main loop
    1. Get camera frame
    2. Extract human keypoints
    3. Classify human pose
    4. Find most detected human pose
    5. Publish most detected human pose
    """
    global show_on, interval, online_pose_clf, start_time, pub

    if len(data.human_list) == 0:
        return

    kps_list = list()
    for human in data.human_list:
        kps_prob = human.body_key_points_with_prob[:9]
        kps = list()
        for kp in kps_prob:
            kps.append(kp.x)
            kps.append(kp.y)
        kps_list.append(kps)

    online_pose_clf.inference(kps_list)

    if perf_counter() - start_time > interval:
        most_pose = online_pose_clf.get_most_pose()

        pose_msgs = Int16()
        pose_msgs.data = most_pose
        pub.publish(pose_msgs)

        start_time = perf_counter()

if __name__ == "__main__":
    rospy.init_node('online_pose_classification_node', anonymous=True)

    rospy.Subscriber('/openpose_ros/human_list', OpenPoseHumanList, callback)
    rospy.spin()