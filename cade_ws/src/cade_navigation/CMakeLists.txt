cmake_minimum_required(VERSION 3.0.2)
project(cade_navigation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  nav_msgs
  tf
  tf2_ros
  actionlib
  actionlib_msgs
  move_base_msgs
)

catkin_python_setup()
catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

catkin_install_python(PROGRAMS
  scripts/navigation_node.py
  scripts/get_current_map_pose.py
  scripts/set_nav_goal.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY src/cade_navigation/
  DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
