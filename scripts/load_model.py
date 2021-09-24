import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# load trained model
clf = joblib.load('test.pkl')

Pose = {0: "leftDown",
        1: "leftUp",
        2: "rightDown",
        3: "rightUp",
        4: "twoDown",
        5: "twoUp",
        6: "heart"}

points = ['Nose_x','Nose_y','RShoulder_x','RShoulder_y','RElbow_x','RElbow_y','RWrist_x','RWrist_y', \
        'LShoulder_x','LShoulder_y','LElbow_x','LElbow_y','LWrist_x','LWrist_y']

data = pd.read_csv("./output/pose.txt")

X = data[points]
y = data['pose']

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.1, random_state = 1)

predict1 = clf.predict(test_x)
print("accuracy : %.2f%%"%(accuracy_score(test_y,predict1)*100))

# # load trained model
# clf = joblib.load('test.pkl')

# Pose = {0: "leftDown",
#         1: "leftUp",
#         2: "rightDown",
#         3: "rightUp",
#         4: "twoDown",
#         5: "twoUp",
#         6: "heart"}

# points = ['Nose_x','Nose_y','RShoulder_x','RShoulder_y','RElbow_x','RElbow_y','RWrist_x','RWrist_y', \
#         'LShoulder_x','LShoulder_y','LElbow_x','LElbow_y','LWrist_x','LWrist_y']

# input_x = [[0,-0.0807,-0.3684,-0.3501,-0.0325,-0.4590,0.4717,-0.4961,0.8472,0.3323,0.0079,0.4783,0.4129,0.6792,0.3512]]
# answer = Pose[input_x[0][0]]

# x = [input_x[0][1:]]
# predict1 = int(clf.predict(x))
# print("GT : %s\t Predict : %s"%(answer, Pose[predict1]))