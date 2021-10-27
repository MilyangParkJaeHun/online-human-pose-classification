"""
    data_handler.py
    Author: Park Jaehun
    Purpose
        parse raw data collected from OpenPose ros package.
        raw data format : [frame_id, pose_x, pose_y, pose_x, pose_y, ...]
        parsed data format : [pose_id, pose_x, pose_y, pose_x, pose_y, ...]
"""
import os
import sys
import math
import cv2
import numpy as np

Pose = {"leftDown"  : 0,
        "leftUp"    : 1,
        "rightDown" : 2,
        "rightUp"   : 3,
        "twoDown"   : 4,
        "twoUp"     : 5,
        "heart"     : 6,
        "normal"    : 7}

data_dir_list = ['/home/park/DATA/pose_estimation/output', '/home/park/DATA/pose_estimation/output2', '/home/park/DATA/pose_estimation/output3']

def rad_to_degree(rad):
    return rad / math.pi * 180

def isEven(num):
    return (num+1)%2 

def distance(a_x, a_y, b_x, b_y):
    return max(1., math.sqrt((a_x-b_x)**2 + (a_y-b_y)**2))

class Pos():
    def __init__(self, x, y):
        self.x = x
        self.y = y

class DataHandler():
    def parse_line(self, line):
        data = line[:-1].split(',')
        data = [float(d) for d in data]
        data[0] = int(data[0])

        return data[0], data[1:]

    def preprocess_data(self, data):
        # remove image file name
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

        remove_kps = [17, 16, 3, 2]
        for remove_kp in remove_kps:
            del data[remove_kp]
        return data

    def shift_pos(self, x, y):
        size = 100
        x_center = 320
        y_center = 240

        return int(size*x + x_center), int(size*y + y_center)

    def draw_data(self, data, color=(255, 100, 100)):
        print(color)
        frame = np.zeros((480, 640, 3))
        cv2.line(frame, (320, 240), (320, 240), color, 5)
        cv2.line(frame, (320, 340), (320, 340), color, 5)
        for i in range(7):
            point = self.shift_pos(data[2*i], data[2*i+1])
            cv2.line(frame, point, point, color, 5)
        return frame

if __name__ == '__main__':
    out_dir = '../output'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_fn = os.path.join(out_dir, 'pose.txt')

    data_handler = DataHandler()
    count = 0
    if not os.path.exists(os.path.join(out_dir, 'img')):
        os.makedirs(os.path.join(out_dir, 'img'))
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
            print("Processing path : %s ..."%(data_dir))
            for pose in Pose.keys():
                print(pose)
                in_fn = os.path.join(data_dir, '%s.txt'%(pose))
                if not os.path.exists(in_fn):
                    continue
                if not os.path.exists(os.path.join(out_dir, pose)):
                    os.makedirs(os.path.join(out_dir, pose))
                with open(in_fn, 'r') as in_file:
                    # cut column name line
                    in_file.readline()
                    while True:
                        line = in_file.readline()
                        if not line:
                            break
                        img_id, d = data_handler.parse_line(line)
                        d = data_handler.preprocess_data(d)

                        print("%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f"%
                        (Pose[pose],
                        d[0], d[1],
                        d[2], d[3],
                        d[4], d[5],
                        d[6], d[7],
                        d[8], d[9],
                        d[10], d[11],
                        d[12], d[13]), file=out_file)
                        cv2.imwrite(os.path.join(out_dir, 'img','%06d.png'%(count)), data_handler.draw_data(d))
                        count += 1
            print("Done %s!!!"%(data_dir))