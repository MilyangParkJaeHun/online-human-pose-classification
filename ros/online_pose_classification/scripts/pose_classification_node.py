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
from kps_extraction.data_handler import DataHandler
from pose_classification.pose_classifier import PoseClassifier
from online_pose_classification import build_argparser
from online_pose_classification import OnlinePoseClassfier

import rospy
from std_msgs.msg import Int16

def main(pub: rospy.Publisher) -> None:
    """
    Run main loop
    1. Get camera frame
    2. Extract human keypoints
    3. Classify human pose
    4. Find most detected human pose
    5. Publish most detected human pose
    """
    args = build_argparser().parse_args()
    show_on = args.show_on
    input_stream = args.input
    prob_threshold = args.prob_threshold
    interval = args.interval

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
    pose_classifier = PoseClassifier(model_path=args.classifier_model)

    # Create online human pose classifier
    online_pose_clf = OnlinePoseClassfier(kps_extractor, pose_classifier, data_handler, prob_threshold)

    pose_msgs = Int16()
    pose_buf = [0] * 8
    most_pose = 7

    start_time = perf_counter()
    while not rospy.is_shutdown() and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            rospy.loginfo("Could not read camera...")
            break
        
        online_pose_clf.inference(frame)

        if perf_counter() - start_time > interval:
            most_pose = online_pose_clf.get_most_pose()

            pose_msgs.data = most_pose
            pub.publish(pose_msgs)

            start_time = perf_counter()

        if show_on:
            cv2.putText(frame, pose_classifier.to_str(most_pose), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('pose classification', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
                
    cap.release()

if __name__ == "__main__":
    rospy.init_node('oneline_pose_classification_node', anonymous=True)
    
    pub = rospy.Publisher('/manta/vision/pose', Int16, queue_size=10)

    try:
        main(pub)
        sys.exit(0)
    except rospy.ROSInterruptException:
        sys.exit(1)