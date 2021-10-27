"""
    pose_classifier.py
    Author: Park Jaehun
    Purpose
        Class for classifying human pose.
        Estimating human pose with a classification machine learning model
        using human pose keypoints data.
"""
import os
import random
import joblib
from argparse import ArgumentParser

def build_argparser():
    parser = ArgumentParser(description='Pose classification')
    parser.add_argument('-m', '--model', help='Path to an .pkl file with a trained model.',
                      default='weights/test.pkl', type=str)

    args = parser.parse_args()
    return args

POSE_CLF_PATH = os.environ['POSE_CLF_PATH']
default_model_path_ = os.path.join(POSE_CLF_PATH, 'pose_classification','weights','test.pkl')

class PoseClassifier():
    def __init__(self, model_path=default_model_path_):
        self.clf = joblib.load(model_path)
        self.Pose = {0: "leftDown",
                     1: "leftUp",
                     2: "rightDown",
                     3: "rightUp",
                     4: "twoDown",
                     5: "twoUp",
                     6: "heart",
                     7: "normal"}
        self.points = ['Nose_x','Nose_y','RShoulder_x','RShoulder_y','RElbow_x','RElbow_y','RWrist_x','RWrist_y', \
                        'LShoulder_x','LShoulder_y','LElbow_x','LElbow_y','LWrist_x','LWrist_y']
        self.color = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(len(self.Pose))]
    
    def predict(self, x):
        return self.clf.predict([x])[0]

    def to_str(self, index):
        return self.Pose[index]
    
    def pose_color(self, pose_idx):
        return self.color[pose_idx]

if __name__ == "__main__":
    args = build_argparser()
    pose_classifier = PoseClassifier(args.model)

    input_x = [0,-0.0807,-0.3684,-0.3501,-0.0325,-0.4590,0.4717,-0.4961,0.8472,0.3323,0.0079,0.4783,0.4129,0.6792,0.3512]
    answer = input_x[0]
    x = input_x[1:]
    print("GT : %s\t Predict : %s"%(answer, pose_classifier.predict(x)))
