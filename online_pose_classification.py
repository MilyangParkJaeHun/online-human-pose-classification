"""
    online_pose_classification.py
    Author: Park Jaehun
    Purpose
        Class for online inference for human pose classification
"""
from typing import List
from pathlib import Path
from time import perf_counter
from argparse import ArgumentParser, SUPPRESS

import cv2
import numpy as np

from kps_extraction.data_handler import DataHandler
from pose_classification.pose_classifier import PoseClassifier

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-i', '--input', required=True, type=int,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('-it', '--interval', required=False, type=float, default=1.0, 
                      help='Optinal. Time interval to get classification results')

    common_model_args = parser.add_argument_group('Common model options')
    common_model_args.add_argument('-t', '--prob_threshold', default=0.1, type=float,
                                   help='Optional. Probability threshold for poses filtering.')
    common_model_args.add_argument('--tsize', default=None, type=int,
                                   help='Optional. Target input size. This demo implements image pre-processing '
                                        'pipeline that is common to human pose estimation approaches. Image is first '
                                        'resized to some target size and then the network is reshaped to fit the input '
                                        'image shape. By default target image size is determined based on the input '
                                        'shape from IR. Alternatively it can be manually set via this parameter. Note '
                                        'that for OpenPose-like nets image is resized to a predefined height, which is '
                                        'the target size in this case. For Associative Embedding-like nets target size '
                                        'is the length of a short first image side.')

    clf_infer_args = parser.add_argument_group('Pose classifier inference options')
    clf_infer_args.add_argument('-cm', '--classifier_model', help='Required. Path to an .pkl file with a trained model.',
                      required=True, type=Path)

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('-show_on', '--show_on', help="Optional. Show output.", action='store_true')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')
    return parser

class OnlinePoseClassfier():
    def __init__(self, pose_classifier: PoseClassifier, \
                    data_handler: DataHandler, \
                    prob_threshold: float):
        self.pose_classifier = pose_classifier
        self.data_handler = data_handler

        self.prob_threshold = prob_threshold
        self.pose_buf = [0] * 8

    def inference(self, kps_list: list):
        """
        After estimating human pose in the camera frame,
        results of estimating human pose is stored in pose buffer.
        """
        for kps in kps_list:
            if self.isNotValid(kps, self.prob_threshold):
                continue
            # kps = self.convert_to_openpose(kps)
            kps = self.data_handler.preprocess_data(kps)

            pose = self.pose_classifier.predict(kps)

            self.pose_buf[pose] += 1

    def get_most_pose(self) -> int:
        """
        Find most detected pose in pose buffer.
        """
        most_pose = self.find_most()
        self.clear_pose_buf()

        return most_pose

    def convert_to_openpose(self, data: List[float]) -> List[float]:
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

    def isNotValid(self, data: List[float], prob_threshold: float) -> bool:
        """
        Check if kepoints has unvalid kepoint data.
        """
        for d in data:
            if d == 0.0:
                return True
        return False

    def clear_pose_buf(self) -> None:
        """
        Convert all buffer elements to zero.
        """
        for i in range(len(self.pose_buf)):
            self.pose_buf[i] = 0

    def find_most(self) -> int:
        """
        Find the most detected pose except normal pose.
        normal pose index is always last index of pose_buf.
        """
        normal_pose = len(self.pose_buf) - 1
        most_pose = 0
        for i in range(len(self.pose_buf) - 1):
            if self.pose_buf[most_pose] <= self.pose_buf[i]:
                most_pose = i

        if self.pose_buf[most_pose] > 0:
            return most_pose
        return normal_pose