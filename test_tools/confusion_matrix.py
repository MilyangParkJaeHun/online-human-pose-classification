"""
    confusion_matrix.py
    Author: Park Jaehun
    Purpose
        get confusion matrix.
"""
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn
import joblib

import os
import sys

POSE_CLF_PATH=os.environ["POSE_CLF_PATH"]
sys.path.append(POSE_CLF_PATH)

# load trained model
clf = joblib.load(os.path.join(POSE_CLF_PATH, 'pose_classification', 'weights', 'test.pkl'))

Pose = ["leftDown","leftUp","rightDown","rightUp","twoDown","twoUp","heart", "normal"]

points = ['Nose_x','Nose_y','RShoulder_x','RShoulder_y','RElbow_x','RElbow_y','RWrist_x','RWrist_y', \
        'LShoulder_x','LShoulder_y','LElbow_x','LElbow_y','LWrist_x','LWrist_y']

data = pd.read_csv(os.path.join(POSE_CLF_PATH, 'data', 'pose.txt'))

X = data[points]
y = data['pose']

predict = clf.predict(X)

CM = confusion_matrix(y, predict)
seaborn.heatmap(CM)
plt.show()