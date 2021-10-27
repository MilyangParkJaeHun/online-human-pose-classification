"""
    train_random_forest.py
    Author: Park Jaehun
    Purpose
        train random forest model to classify human pose
        using human pose keypoints
    Reference
        https://todayisbetterthanyesterday.tistory.com/51
"""
import os
import yaml
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

POSE_CLF_PATH = os.environ['POSE_CLF_PATH']
yaml_fn_ = os.path.join(POSE_CLF_PATH, 'config', 'pose.yml')

Pose = dict()
with open(yaml_fn_) as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    for d in data:
        Pose[d['index']] = d['name']

points = ['Nose_x','Nose_y','RShoulder_x','RShoulder_y','RElbow_x','RElbow_y','RWrist_x','RWrist_y', \
        'LShoulder_x','LShoulder_y','LElbow_x','LElbow_y','LWrist_x','LWrist_y']

os.getcwd()

data = pd.read_csv(os.path.join(POSE_CLF_PATH, 'data', 'pose.txt'))

nData = data.shape[0]
nVar = data.shape[1]
print('nData: %d' % nData, 'nVar: %d' % nVar )

X = data[points]
y = data['pose']

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

clf = RandomForestClassifier(n_estimators=20, max_depth=10,random_state=0)

# train
clf.fit(train_x,train_y)

# inference
predict1 = clf.predict(test_x)
print("accuracy : %.2f%%"%(accuracy_score(test_y,predict1)*100))

# save trained model
weights_name = 'pose_clf_20211027.pkl'
model_path = os.path.join(POSE_CLF_PATH, 'pose_classification','weights', weights_name)
joblib.dump(clf, model_path)
