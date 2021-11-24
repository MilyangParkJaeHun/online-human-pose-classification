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

from kps_extraction.Openvino import Openvino
from kps_extraction.Openvino import build_argparser as openvino_args
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

    openvino_infer_args = parser.add_argument_group('Openvino inference options')
    openvino_infer_args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                            default=1, type=int)
    openvino_infer_args.add_argument('-nstreams', '--num_streams',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).',
                            default='', type=str)
    openvino_infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')
    openvino_infer_args.add_argument('-em', '--extractor_model', help='Required. Path to an .xml file with a trained model.',
                      required=True, type=Path)
    openvino_infer_args.add_argument('-at', '--architecture_type', help='Required. Specify model\' architecture type.',
                      type=str, required=True, choices=('ae', 'openpose'))
    openvino_infer_args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.')

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
    def __init__(self, kps_extractor: Openvino, \
                    pose_classifier: PoseClassifier, \
                    data_handler: DataHandler, \
                    prob_threshold: float):
        self.kps_extractor = kps_extractor
        self.pose_classifier = pose_classifier
        self.data_handler = data_handler

        self.prob_threshold = prob_threshold
        self.pose_buf = [0] * 8

    def inference(self, frame: np.ndarray):
        """
        After estimating human pose in the camera frame,
        results of estimating human pose is stored in pose buffer.
        """
        kps_list = self.kps_extractor.inference(frame)
        for kps in kps_list:
            if self.isNotValid(kps, self.prob_threshold):
                continue
            kps = self.convert_to_openpose(kps)
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
        check_list = [0, 5, 6, 7, 8, 9, 10, 11, 12]
        for idx in check_list:
            d = data[idx]
            if d[0] == 0.0 or d[1] == 0.0 or d[2] < prob_threshold:
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

if __name__ == "__main__":
    args = build_argparser().parse_args()
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
    pose_classifier = PoseClassifier(args.classifier_model)

    # Create online human pose classifier
    online_pose_clf = OnlinePoseClassfier(kps_extractor, pose_classifier, data_handler, prob_threshold)

    start_time = perf_counter()
    most_pose = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        online_pose_clf.inference(frame)

        if perf_counter() - start_time > interval:
            most_pose = online_pose_clf.get_most_pose()
            start_time = perf_counter()

        cv2.putText(frame, pose_classifier.to_str(most_pose), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('pose classification', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    sys.exit(0)