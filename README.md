# online-human-pose-classification
Online inference for human pose classification

## Set Environment Variable
```
$ export OPENVINO_PATH=/opt/intel/openvino/deployment_tools/inference_engine/demos/common/python
$ export POSE_CLF_PATH=/path/to/online-human-pose-classification
```

## Demo Environments
- Ubuntu 20.04
- ROS noetic
- OpenCV 4.5.2
- OpenVINO 2021.3.394
- scikit-learn 0.24.1

## Create Symbolic Link for ROS
```
$ cd ~/catkin_ws/src
$ ln -s /path/to/online-human-pose-classification/ros/online_pose_classification .
$ cd online-human-pose-classification
$ ln -s /path/to/online-human-pose-classification/config .
```

## Build
```
$ catkin_make -C ~/catkin_ws/src
```

## Run Demo
```
$ rosrun online_pose_classification pose_classification_node.py \
	-d CPU \
	-i 2 \
	-em /opt/intel/openvino_2021.3.394/deployment_tools/open_model_zoo/tools/downloader/intel/human-pose-estimation-0005/FP16/human-pose-estimation-0005.xml \
	-cm /home/openvino/online-human-pose-classification/pose_classification/weights/pose_clf_20211027.pkl \
	-at ae \
	-t 0.1 \
	-r \
	--show_on
```

## Reference
- https://docs.openvino.ai/latest/omz_demos_human_pose_estimation_demo_python.html
