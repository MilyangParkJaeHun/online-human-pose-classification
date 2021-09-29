"""
    load_model.py
    Author: Park Jaehun
    Purpose
        load trained random forest model.
"""
import pandas as pd
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
        6: "heart",
        7: "normal"}

points = ['Nose_x','Nose_y','RShoulder_x','RShoulder_y','RElbow_x','RElbow_y','RWrist_x','RWrist_y', \
        'LShoulder_x','LShoulder_y','LElbow_x','LElbow_y','LWrist_x','LWrist_y']

data = pd.read_csv("./output/pose.txt")

X = data[points]
y = data['pose']

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.1, random_state = 1)

predict1 = clf.predict(test_x)
print("accuracy : %.2f%%"%(accuracy_score(test_y,predict1)*100))