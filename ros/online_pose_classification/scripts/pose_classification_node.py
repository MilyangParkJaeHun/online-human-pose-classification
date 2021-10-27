"""
    pose_classification_node.py
    Author: Park Jaehun
    Purpose
        ROS node for online human pose classification
    Pub Topic Info
        data type : Int16
        name : /manta/vision/pose
"""
#!/usr/bin/env python3

# pose list : /path/to/online_pose_classification/config

import os
import sys
import logging
import cv2
from typing import List
from time import perf_counter

POSE_CLF_PATH=os.environ["POSE_CLF_PATH"]
sys.path.append(POSE_CLF_PATH)

from kps_extraction.Openvino import Openvino
from kps_extraction.Openvino import build_argparser as openvino_args
from kps_extraction.data_handler import DataHandler
from pose_classification.pose_classifier import PoseClassifier

import rospy
from std_msgs.msg import Int16

def convert_to_openpose(data: List[float]) -> List[float]:
    """
    Convert data format from openvino to openpose.

    openvino format : [Nose, LEye, REye, LEar, REar, LShoulder, RShoulder, LElbow, RElbow, LWrist, Rwrisst, LHip, RHip]
    openpose format : [Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist, MidHip]
    """
    return [data[0][0], data[0][1],
            (data[5][0]+data[6][0])/2, (data[5][1]+data[6][1])/2,
            data[6][0], data[6][1],
            data[8][0], data[8][1],
            data[10][0], data[10][1],
            data[5][0], data[5][1],
            data[7][0], data[7][1],
            data[9][0], data[9][1],
            (data[11][0]+data[12][0])/2, (data[11][1]+data[12][1])/2]

def isNotValid(data: List[float], prob_threshold: float) -> bool:
    """
    Check if kepoints has unvalid kepoint data.
    """
    check_list = [0, 5, 6, 7, 8, 9, 10, 11, 12]
    for idx in check_list:
        d = data[idx]
        if d[0] == 0.0 or d[1] == 0.0 or d[2] < prob_threshold:
            return True
    return False

def clear_buf(buf: List[int]) -> None:
    """
    Convert all buffer elements to zero.
    """
    for i in range(len(buf)):
        buf[i] = 0

def find_max(buf: List[int]) -> int:
    """
    Find the most detected poses.
    """
    max_pose = 0
    for i in range(len(buf)):
        if buf[max_pose] <= buf[i]:
            max_pose = i
    return max_pose

def main(pub: rospy.Publisher) -> None:
    """
    Run main loop
    1. Get camera frame
    2. Extract human keypoints
    3. Classify human pose.
    4. Publish most detected human pose.
    """
    args = openvino_args().parse_args()
    show_on = args.show_on
    input_stream = args.input
    prob_threshold = args.prob_threshold

    cap_width = 640
    cap_height = 480
    cap = cv2.VideoCapture(input_stream)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Create human keypoints extractor instance
    kps_extractor = Openvino(args, [cap_height, cap_width])

    # Create human keypoints data handler instance
    data_handler = DataHandler()

    # Create human pose classifier
    clf_weights = os.path.join(POSE_CLF_PATH, "pose_classification","weights", "test.pkl")
    pose_classifier = PoseClassifier(model_path=clf_weights)

    pose_msgs = Int16()
    pose_buf = [0] * 8

    start_time = perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        kps_list = kps_extractor.inference(frame)
        for kps in kps_list:
            if isNotValid(kps, prob_threshold):
                continue
            kps = convert_to_openpose(kps)
            kps = data_handler.preprocess_data(kps)

            pose = pose_classifier.predict(kps)
            color = pose_classifier.pose_color(pose)

            pose_buf[pose] += 1

        if perf_counter() - start_time > 3:
            max_pose = find_max(pose_buf)
            pose_msgs.data = max_pose
            pub.publish(pose_msgs)

            clear_buf(pose_buf)
            start_time = perf_counter()

            if show_on:
                cv2.putText(frame, pose_classifier.to_str(max_pose), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2, cv2.LINE_AA)

        if show_on:
            cv2.imshow('pose classification', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    rospy.init_node('oneline_pose_classification_node', anonymous=True)
    
    pub = rospy.Publisher('/manta/vision/pose', Int16, queue_size=10)

    try:
        main(pub)
    except rospy.ROSInterruptException:
        sys.exit(0)