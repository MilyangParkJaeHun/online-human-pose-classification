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

# load trained model
clf = joblib.load('test.pkl')

Pose = ["leftDown","leftUp","rightDown","rightUp","twoDown","twoUp","heart"]

points = ['Nose_x','Nose_y','RShoulder_x','RShoulder_y','RElbow_x','RElbow_y','RWrist_x','RWrist_y', \
        'LShoulder_x','LShoulder_y','LElbow_x','LElbow_y','LWrist_x','LWrist_y']

data = pd.read_csv("./output/pose.txt")

X = data[points]
y = data['pose']

predict = clf.predict(X)

CM = confusion_matrix(y, predict)
seaborn.heatmap(CM)
plt.show()