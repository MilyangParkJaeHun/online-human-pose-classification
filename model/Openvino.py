"""
    Openvino.py
    Author: Park Jaehun
    Purpose
        Class for real-time inference of pose estimation using Openvino
"""
#!/usr/bin/env python3

import os
import sys
import logging
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from openvino.inference_engine import IECore

OPENVINO_PATH = os.environ['OPENVINO_PATH']
sys.path.append(OPENVINO_PATH)
import models
import monitors
from images_capture import open_images_capture
from pipelines import AsyncPipeline
from performance_metrics import PerformanceMetrics

log = logging.getLogger()
log.setLevel(logging.INFO)

formatter = logging.Formatter('[ %(levelname)s ] %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.',
                      required=True, type=Path)
    args.add_argument('-at', '--architecture_type', help='Required. Specify model\' architecture type.',
                      type=str, required=True, choices=('ae', 'openpose'))
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.')

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

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                            default=1, type=int)
    infer_args.add_argument('-nstreams', '--num_streams',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).',
                            default='', type=str)
    infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('-no_show', '--no_show', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')
    return parser

default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

colors = (
        (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
        (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
        (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
        (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
        (0, 170, 255))

class Openvino():
    def __init__(self, args, frame_shape):
        metrics = PerformanceMetrics()

        log.info('Initializing Inference Engine...')
        ie = IECore()

        plugin_config = self.get_plugin_configs(args.device, args.num_streams, args.num_threads)
        
        log.info('Loading network...')
        model = self.get_model(ie, args, frame_shape[1] / frame_shape[0])
        self.hpe_pipeline = AsyncPipeline(ie, model, plugin_config, device=args.device, max_num_requests=args.num_infer_requests)

        self.next_frame_id = 1
        self.next_frame_id_to_show = 0

        self.presenter = monitors.Presenter(args.utilization_monitors, 55,
                                    (round(frame_shape[1] / 4), round(frame_shape[0] / 8)))
        self.is_first_inference = True
        self.prob_threshold = args.prob_threshold
        self.raw_output_message = args.raw_output_message

    def inference(self, in_frame):
        if self.is_first_inference:
            self.is_first_inference = False
            log.info('Starting inference...')
            start_time = perf_counter()
            self.hpe_pipeline.submit_data(in_frame, 0, {'frame': in_frame, 'start_time': start_time})

            print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
            return []

        if self.hpe_pipeline.callback_exceptions:
                raise self.hpe_pipeline.callback_exceptions[0]
        # Process all completed requests
        results = self.hpe_pipeline.get_result(self.next_frame_id_to_show)
        if results:
            (poses, scores), frame_meta = results
            frame = frame_meta['frame']
            self.presenter.drawGraphs(frame)
            frame = self.draw_poses(in_frame, poses, self.prob_threshold)

            if len(poses) and self.raw_output_message:
                self.print_raw_results(poses, scores)

            self.next_frame_id_to_show += 1
            
            return poses

        if self.hpe_pipeline.is_ready():
            # Get new image/frame
            start_time = perf_counter()
            frame = in_frame
            if frame is None:
                return []

            # Submit for inference
            self.hpe_pipeline.submit_data(frame, self.next_frame_id, {'frame': frame, 'start_time': start_time})
            self.next_frame_id += 1

        else:
            # Wait for empty request
            self.hpe_pipeline.await_any()
        
        return []

    def get_model(self, ie, args, aspect_ratio):
        if args.architecture_type == 'ae':
            Model = models.HpeAssociativeEmbedding
        elif args.architecture_type == 'openpose':
            Model = models.OpenPose
        else:
            raise RuntimeError('No model type or invalid model type (-at) provided: {}'.format(args.architecture_type))
        return Model(ie, args.model, target_size=args.tsize, aspect_ratio=aspect_ratio, prob_threshold=args.prob_threshold)


    def get_plugin_configs(self, device, num_streams, num_threads):
        config_user_specified = {}

        devices_nstreams = {}
        if num_streams:
            devices_nstreams = {device: num_streams for device in ['CPU', 'GPU'] if device in device} \
                if num_streams.isdigit() \
                else dict(device.split(':', 1) for device in num_streams.split(','))

        if 'CPU' in device:
            if num_threads is not None:
                config_user_specified['CPU_THREADS_NUM'] = str(num_threads)
            if 'CPU' in devices_nstreams:
                config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                    if int(devices_nstreams['CPU']) > 0 \
                    else 'CPU_THROUGHPUT_AUTO'

        if 'GPU' in device:
            if 'GPU' in devices_nstreams:
                config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                    if int(devices_nstreams['GPU']) > 0 \
                    else 'GPU_THROUGHPUT_AUTO'

        return config_user_specified



    def draw_poses(self, img, poses, point_score_threshold, skeleton=default_skeleton, draw_ellipses=False):
        if poses.size == 0:
            return img
        stick_width = 4

        img_limbs = np.copy(img)
        for pose in poses:
            points = pose[:, :2].astype(int).tolist()
            points_scores = pose[:, 2]
            # Draw joints.
            for i, (p, v) in enumerate(zip(points, points_scores)):
                if v > point_score_threshold:
                    cv2.circle(img, tuple(p), 1, colors[i], 2)
                    cv2.putText(img, str(i), tuple(p), cv2.FONT_HERSHEY_PLAIN, 1, colors[i], 2, cv2.LINE_AA)
            # Draw limbs.
            for i, j in skeleton:
                if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                    if draw_ellipses:
                        middle = (points[i] + points[j]) // 2
                        vec = points[i] - points[j]
                        length = np.sqrt((vec * vec).sum())
                        angle = int(np.arctan2(vec[1], vec[0]) * 180 / np.pi)
                        polygon = cv2.ellipse2Poly(tuple(middle), (int(length / 2), min(int(length / 50), stick_width)),
                                                angle, 0, 360, 1)
                        cv2.fillConvexPoly(img_limbs, polygon, colors[j])
                    else:
                        cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=stick_width)
        cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
        return img

    def print_raw_results(self, poses, scores):
        log.info('Poses:')
        for pose, pose_score in zip(poses, scores):
            pose_str = ' '.join('({:.2f}, {:.2f}, {:.2f})'.format(p[0], p[1], p[2]) for p in pose)
            log.info('{} | {:.2f}'.format(pose_str, pose_score))

if __name__ == '__main__':
    args = build_argparser().parse_args()

    cap_width = 640
    cap_height = 480
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    model = Openvino(args, [cap_height, cap_width])

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        poses = model.inference(frame)
        cv2.imshow('test', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    sys.exit(0)