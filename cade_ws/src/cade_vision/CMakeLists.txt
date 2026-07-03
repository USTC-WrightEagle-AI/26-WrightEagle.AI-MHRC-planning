cmake_minimum_required(VERSION 3.0.2)
project(cade_vision)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  tf2_ros
)

catkin_python_setup()
catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
  scripts/open_vision_node.py
  scripts/open_vision_frame_source.py
  scripts/open_vision_person_worker.py
  scripts/open_vision_pose_gesture_worker.py
  scripts/open_vision_cloth_worker.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

file(MAKE_DIRECTORY
  ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}/models
)
file(COPY src/cade_vision/models/
  DESTINATION ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}/models
  FILES_MATCHING
    PATTERN "*.pt"
    PATTERN "*.engine"
)

file(MAKE_DIRECTORY
  ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}/kits/cloth/models
)
file(COPY src/cade_vision/kits/cloth/models/
  DESTINATION ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_PYTHON_DESTINATION}/kits/cloth/models
  FILES_MATCHING
    PATTERN "*.xml"
    PATTERN "*.npz"
    PATTERN "*.json"
)

install(DIRECTORY src/cade_vision/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
