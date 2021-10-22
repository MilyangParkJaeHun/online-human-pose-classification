"""
    pose_classification_node.py
    Author: Park Jaehun
    Purpose
        ROS node for online human pose classification
"""
#!/usr/bin/env python3

# topic data type : Int16
# topic name : /manta/vision/pose
# pose list : /path/to/online_pose_classification/config


import os
import sys
import logging

import rospy

