"""
    data_collector.py
    Author: Park Jaehun
    Purpose
        collect raw data from OpenPose ros package.
        raw data format : [frame_id, pose_x, pose_y, pose_x, pose_y, ...]
"""
#!/usr/bin/env python
import roslib
import rospkg
import rospy

from openpose_ros_msgs.msg import OpenPoseHumanList
from openpose_ros_msgs.msg import OpenPoseHuman
from whale_ros_msgs.msg import BodypartDetection
from whale_ros_msgs.msg import PersonDetection
from sensor_msgs.msg import Image

from cv_bridge import CvBridge

import os
import sys
import cv2

type_name = 'heart'

bridge = CvBridge()

rospack = rospkg.RosPack()
pkg_path = rospack.get_path('openpose_src')
out_path = os.path.join(pkg_path, 'output')
out_img_path = os.path.join(pkg_path, 'output', type_name)
if not os.path.exists(out_img_path):
    os.makedirs(out_img_path)
out_fn = os.path.join(out_path, type_name+'.txt')
out_file = open(out_fn, 'w')

frame_id = 0
frame = None

def isValid(keypoints):
    for kp in keypoints:
        if kp.prob < 0.4 or kp.x == 0.0 or kp.y == 0.0:
            return False
    print('Valid!!!!!!!!!')
    return True

def callback(data):
    global out_file, frame_id, frame
    if len(data.human_list) == 0:
        return

    save_flag = False
    for human in data.human_list:
        kps = human.body_key_points_with_prob[:9]
        if isValid(kps):
            save_flag = True
            out_file.write("%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,\
%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,\
%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n"% 
                (frame_id,
                kps[0].x, kps[0].y, 
                kps[1].x, kps[1].y, 
                kps[2].x, kps[2].y, 
                kps[3].x, kps[3].y, 
                kps[4].x, kps[4].y, 
                kps[5].x, kps[5].y, 
                kps[6].x, kps[6].y, 
                kps[7].x, kps[7].y, 
                kps[8].x, kps[8].y))
            for i in range(9):
                cv2.line(frame, (int(kps[i].x), int(kps[i].y)), (int(kps[i].x), int(kps[i].y)), (255, 0, 0), 5)
    print(save_flag)
    if save_flag and hasattr(frame, 'shape'):
        print('save')
        cv2.imwrite(os.path.join(out_img_path, '%d.jpg'%(frame_id)), frame)
        frame_id += 1

def imageCallback(data):
    global frame
    frame = bridge.imgmsg_to_cv2(data, "bgr8")

if __name__ == '__main__':
    rospy.init_node('keypoints_collecter', anonymous=True)
    rospy.Subscriber('/openpose_ros/human_list', OpenPoseHumanList, callback)
    rospy.Subscriber('/rtsp2/image_raw', Image, imageCallback)

    rospy.spin()
    out_file.close()
