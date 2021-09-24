import os
import sys
import math
import cv2
import numpy as np
from enum import Enum

Pose = {"leftDown"  : 0,
        "leftUp"    : 1,
        "rightDown" : 2,
        "rightUp"   : 3,
        "twoDown"   : 4,
        "twoUp"     : 5,
        "heart"     : 6}

data_dir_list = ['/home/park/DATA/pose_estimation/output', '/home/park/DATA/pose_estimation/output2']

class Pos():
    def __init__(self, x, y):
        self.x = x
        self.y = y

def rad_to_degree(rad):
    return rad / math.pi * 180

def isEven(num):
    return (num+1)%2 

def distance(a_x, a_y, b_x, b_y):
    return math.sqrt((a_x-b_x)**2 + (a_y-b_y)**2)

def parse_line(line):
    data = line[:-1].split(',')[1:]
    data = [float(d) for d in data]

    neck_pos = Pos(data[2], data[3])
    data = [data[i]-neck_pos.x if isEven(i) else data[i]-neck_pos.y for i in range(len(data))]
    
    # distance between neck(1) and midheap(8).
    waist_len = distance(data[2], data[3], data[16], data[17])
    waist_angle = -1*math.atan2(data[17], data[16]) + math.pi/2

    rotated_data = []
    for i in range(9):
        x = data[2*i]
        y = data[2*i + 1]

        r_x = x * math.cos(waist_angle) - y * math.sin(waist_angle)
        r_y = x * math.sin(waist_angle) + y * math.cos(waist_angle)

        rotated_data.append(r_x)
        rotated_data.append(r_y)

    data = [d / waist_len for d in rotated_data]
 
    return data

def shift_pos(x, y):
    size = 100
    x_center = 320
    y_center = 240

    return int(size*x + x_center), int(size*y + y_center)

def draw_data(data):
    frame = np.zeros((480, 640, 3))
    for i in range(9):
        point = shift_pos(data[2*i], data[2*i+1])
        cv2.line(frame, point, point, (255, 100, 100), 5)
    return frame

if __name__ == '__main__':
    out_dir = 'output'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_fn = os.path.join(out_dir, 'pose.txt')

    count = 0
    with open(out_fn, 'w') as out_file:
        print("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s"%
        ("pose",
        "Nose_x", "Nose_y",
        "RShoulder_x", "RShoulder_y", 
        "RElbow_x", "RElbow_y",
        "RWrist_x", "RWrist_y",
        "LShoulder_x", "LShoulder_y",
        "LElbow_x", "LElbow_y",
        "LWrist_x", "LWrist_y"
        ), file=out_file)

        for data_dir in data_dir_list:
            for pose in Pose.keys():
                print(pose)
                # if not os.path.exists(os.path.join(out_dir, pose)):
                    # os.makedirs(os.path.join(out_dir, pose))
                in_fn = os.path.join(data_dir, '%s.txt'%(pose))
                with open(in_fn, 'r') as in_file:
                    while True:
                        line = in_file.readline()
                        if not line:
                            break    
                        d = parse_line(line)

                        print("%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f"%
                        (Pose[pose],
                        d[0], d[1],
                        d[4], d[5],
                        d[6], d[7],
                        d[8], d[9],
                        d[10], d[11],
                        d[12], d[13],
                        d[14], d[15]), file=out_file)
                        # cv2.imwrite(os.path.join(out_dir, 'img','%06d.png'%(count)), draw_data(d))
                        count += 1