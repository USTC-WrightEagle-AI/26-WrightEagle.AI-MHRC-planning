cmake_minimum_required(VERSION 3.0.2)
project(cade_brain)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

# Install Python scripts
catkin_install_python(PROGRAMS
  scripts/brain_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install Python packages
install(DIRECTORY src/cade_brain/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)
