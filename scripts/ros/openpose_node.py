"""
    openpose_node.py
    Author: Park Jaehun
    Purpose
        Detect human keypoints using OpenPose python api
        The execution environment can be found at https://github.com/MilyangParkJaeHun/docker_ws.git 
        under the path Dockerfile/OpenPose/OpenPose.Dockerfile.
"""
#!/usr/bin/env python
import rospy

import cv2
import numpy as np
import argparse
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from openpose import pyopenpose as op

image_width = 640
image_height = 480

BODY_PARTS = \
{
  "Nose": 0,      "Neck": 1,   "RShoulder": 2, "RElbow": 3,    "RWrist": 4, \
  "LShoulder": 5, "LElbow": 6, "LWrist": 7,    "MidHip": 8,    "RHip": 9, \
  "RKnee": 10,    "RAnkle": 11,"LHip": 12,     "LKnee": 13,    "LAnkle": 14, \
  "REye": 15,     "LEye": 16,  "REar": 17,     "LEar": 18,     "LBigToe": 19, \
  "LSmallToe": 20,"LHeel": 21, "RBigToe": 22,  "RSmallToe": 23,"RHeel": 24 \
}

POSE_PAIRS = \
[ 
  ["MidHip", "Neck"], \
  ["Neck", "RShoulder"], ["Neck", "LShoulder"],\
  ["RShoulder", "RElbow"], ["LShoulder", "LElbow"], \
  ["RElbow", "RWrist"], ["LElbow", "LWrist"] \
]

def checkHandOn(nectPoint, ShoulderPoint, ElbowPoint, WristPoint):
  shoulderY = ShoulderPoint[1]
  elbowY = ElbowPoint[1]
  wristY = WristPoint[1]

  if 0.0 == wristY:
    if 0.0 in [shoulderY, elbowY]:
      return False
    else:
      if elbowY > shoulderY:
        return True
      else:
        return False
    return False

  if wristY < elbowY and wristY < shoulderY:
    return True
  else:
    return False

def checkMotion(body_keypoint):
  nectPoint = body_keypoint[BODY_PARTS["Neck"]]
  RShoulderPoint = body_keypoint[BODY_PARTS["RShoulder"]]
  RElbowPoint = body_keypoint[BODY_PARTS["RElbow"]]
  RWristPoint = body_keypoint[BODY_PARTS["RWrist"]]
  LShoulderPoint = body_keypoint[BODY_PARTS["LShoulder"]]
  LElbowPoint = body_keypoint[BODY_PARTS["LElbow"]]
  LWristPoint = body_keypoint[BODY_PARTS["LWrist"]]

  if checkHandOn(nectPoint, RShoulderPoint, RElbowPoint, RWristPoint) \
      and checkHandOn(nectPoint, LShoulderPoint, LElbowPoint, LWristPoint):
    return True
  else:
    return False

def drawImage(frame, body_keypoints):
  for body_keypoint in body_keypoints:
    if len(body_keypoint) < 25:
      continue

    motion_flag = checkMotion(body_keypoint)
    if motion_flag == True:
      color = (0, 255, 0) # green
    else:
      color = (0, 0, 255) # red

    for pose_pair in POSE_PAIRS:
      partA = pose_pair[0]
      partB = pose_pair[1]
      partA_idx = BODY_PARTS[partA]
      partB_idx = BODY_PARTS[partB]
      partA_prob = body_keypoint[partA_idx][2]
      partB_prob = body_keypoint[partB_idx][2]

      partA_point = tuple([int(x) for x in body_keypoint[partA_idx][:2]])
      partB_point = tuple([int(x) for x in body_keypoint[partB_idx][:2]])

      if 0.0 in partA_point + partB_point:
        continue

      cv2.line(frame, partA_point, partB_point, color, 2)

  return frame   

if __name__ == "__main__":
  rospy.init_node("openpose_node", anonymous=False)

  params = dict()
  params["model_folder"] = "/home/park/openpose/models/"

  opWrapper = op.WrapperPython()
  opWrapper.configure(params)
  opWrapper.start()

  cap = cv2.VideoCapture(0)

  cap.set(3, image_width)
  cap.set(4, image_height)

  while not rospy.is_shutdown():
    ret, frame = cap.read()
    if not ret:
      break
    
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    body_keypoints = datum.poseKeypoints
    if not hasattr(body_keypoints, 'size'):
      pass
    elif len(body_keypoints) <= 0:
      pass
    else:
      frame = drawImage(frame, body_keypoints)
  
    cv2.imshow("OpenPose", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()
