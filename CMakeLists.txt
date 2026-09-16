###########################################################
#                                                         #
# The Verifiable & Control-Theoretic Robotics (VECTR) Lab #
# University of California, Los Angeles                   #
#                                                         #
# Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   #
# Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         #
#                                                         #
###########################################################

cmake_minimum_required(VERSION 3.12.4)
project(direct_lidar_inertial_odometry)

# --- Build system / ROSIDL (C++-only) ---
find_package(ament_cmake REQUIRED)
find_package(rosidl_cmake REQUIRED)
find_package(rosidl_typesupport_cpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# --- ROS deps ---
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)
# CZ and Claude: G1 body filter node
find_package(tf2 REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(urdf REQUIRED)
# end of CZ and Claude

# --- 3rd party ---
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP) # optional; guard usage below

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# Not all machines have <cpuid.h> available
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# --- Generate interfaces (C++ + Python) ---
rosidl_generate_interfaces(${PROJECT_NAME}
  srv/SavePCD.srv
)
rosidl_get_typesupport_target(
  cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp
)

# --- Includes ---
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include)
include_directories(SYSTEM)

# --- Libraries you actually build ---
add_library(nanoflann STATIC
  src/nano_gicp/nanoflann.cc
)
target_link_libraries(nanoflann ${PCL_LIBRARIES})
ament_target_dependencies(nanoflann)

add_library(nano_gicp STATIC
  src/nano_gicp/lsq_registration.cc
  src/nano_gicp/nano_gicp.cc
)

# Link OpenMP only if found
if(OpenMP_CXX_FOUND)
  target_link_libraries(nano_gicp ${PCL_LIBRARIES} OpenMP::OpenMP_CXX nanoflann)
else()
  target_link_libraries(nano_gicp ${PCL_LIBRARIES} nanoflann)
endif()
ament_target_dependencies(nano_gicp)

# --- Executables ---
add_executable(dlio_odom_node
  src/dlio/odom_node.cc
  src/dlio/odom.cc
)
if(OpenMP_CXX_FOUND)
  target_link_libraries(dlio_odom_node ${PCL_LIBRARIES} Threads::Threads OpenMP::OpenMP_CXX nano_gicp)
else()
  target_link_libraries(dlio_odom_node ${PCL_LIBRARIES} Threads::Threads nano_gicp)
endif()
ament_target_dependencies(dlio_odom_node
  tf2_ros pcl_ros nav_msgs pcl_conversions rclcpp sensor_msgs geometry_msgs
)

add_executable(dlio_map_node
  src/dlio/map_node.cc
  src/dlio/map.cc
)
target_link_libraries(dlio_map_node ${PCL_LIBRARIES} ${cpp_typesupport_target})
ament_target_dependencies(dlio_map_node
  tf2_ros pcl_ros pcl_conversions rclcpp
)

# CZ and Claude: G1 self-filter (replaces the ROS 1 sensor_filters + robot_body_filter chain)
add_executable(g1_body_filter_node
  src/body_filter/g1_body_filter_node.cc
)
ament_target_dependencies(g1_body_filter_node
  rclcpp sensor_msgs geometry_msgs visualization_msgs tf2 tf2_ros tf2_eigen urdf
)
# end of CZ and Claude

# --- Export & install headers ---
ament_export_include_directories(include)
install(DIRECTORY include/
        DESTINATION include)

# --- Install and export the actual library targets ---
install(TARGETS nanoflann nano_gicp
        EXPORT export_${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

ament_export_targets(export_${PROJECT_NAME})
ament_export_libraries(nano_gicp nanoflann)

# Export dependencies for downstream consumers
ament_export_dependencies(
  rclcpp std_msgs sensor_msgs geometry_msgs nav_msgs
  pcl_ros pcl_conversions
  rosidl_runtime_cpp rosidl_typesupport_cpp rosidl_default_runtime
)

# --- Install binaries ---
install(TARGETS dlio_odom_node dlio_map_node g1_body_filter_node
        DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY cfg launch
        DESTINATION share/${PROJECT_NAME})

ament_package()
