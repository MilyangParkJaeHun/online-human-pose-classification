cmake_minimum_required(VERSION 3.0.2)
project(online_pose_classification)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package(CATKIN_DEPENDS std_msgs)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/pose_classification_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)