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
# Plain target_link_libraries instead of ament_target_dependencies, which
# ROS 2 Lyrical removed. pcl_conversions exports targets only from Lyrical on;
# on Jazzy its include-dir variable carries it (the other is then empty).
target_include_directories(dlio_odom_node PRIVATE ${pcl_conversions_INCLUDE_DIRS})
target_link_libraries(dlio_odom_node
  tf2_ros::tf2_ros ${pcl_ros_TARGETS} ${nav_msgs_TARGETS} ${pcl_conversions_TARGETS}
  rclcpp::rclcpp ${sensor_msgs_TARGETS} ${geometry_msgs_TARGETS}
)

add_executable(dlio_map_node
  src/dlio/map_node.cc
  src/dlio/map.cc
)
target_link_libraries(dlio_map_node ${PCL_LIBRARIES} ${cpp_typesupport_target})
target_include_directories(dlio_map_node PRIVATE ${pcl_conversions_INCLUDE_DIRS})
target_link_libraries(dlio_map_node
  tf2_ros::tf2_ros ${pcl_ros_TARGETS} ${pcl_conversions_TARGETS} rclcpp::rclcpp
)

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
install(TARGETS dlio_odom_node dlio_map_node
        DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY cfg launch
        DESTINATION share/${PROJECT_NAME})

ament_package()
