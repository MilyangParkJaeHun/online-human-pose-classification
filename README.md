# online-human-pose-classification
Online inference for human pose classification

## Set Environment Variable
```
$ export POSE_CLF_PATH=/path/to/online-human-pose-classification
```

## Demo Environments
- Ubuntu 18.04
- ROS melodic
- OpenCV 3.4.2
- scikit-learn 0.24.2

## Create Symbolic Link for ROS
```
$ cd ~/catkin_ws/src
$ ln -s /path/to/online-human-pose-classification/ros/online_pose_classification .
$ cd online-human-pose-classification
$ ln -s /path/to/online-human-pose-classification/config .
```

## Build
```
$ catkin_make -C ~/catkin_ws
```

## Run Demo
```
$ roslaunch openpose_ros openpose_ros.launch
$ rosrun online_pose_classification pose_classification_node.py \
       -i 2 \
       -cm /home/whale/online-human-pose-classifition/pose_classification/weights/pose_clf_20211027.pkl \
       -t 0.1 \
       -r \
       --show_on
```
