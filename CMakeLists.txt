###########################################################
#                                                         #
# Copyright (c)                                           #
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

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(ENABLE_CLANG_TIDY "Run clang-tidy during the default build" OFF)

if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  find_program(RUN_CLANG_TIDY_EXE NAMES run-clang-tidy run-clang-tidy.py)

  if(NOT CLANG_TIDY_EXE)
    message(STATUS "clang-tidy was not found. Skipping clang-tidy integration.")
    set(ENABLE_CLANG_TIDY OFF)
  elseif(NOT RUN_CLANG_TIDY_EXE)
    message(STATUS "run-clang-tidy was not found. Skipping clang-tidy integration.")
    set(ENABLE_CLANG_TIDY OFF)
  endif()
endif()

find_package(rosidl_default_generators REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_srvs REQUIRED)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED )
find_package(Threads REQUIRED)

add_compile_options(-std=c++17 -O3 -mtune=native -march=native -fopenmp -DNDEBUG) #-funsafe-math-optimizations -funsafe-loop-optimizations
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# Not all machines have <cpuid.h> available
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

rosidl_generate_interfaces(
    ${PROJECT_NAME}
        srv/SavePCD.srv
    DEPENDENCIES
)
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(include)
include_directories(SYSTEM)

# NanoFLANN
add_library(nanoflann STATIC
  src/nano_gicp/nanoflann.cc
)
target_link_libraries(nanoflann ${PCL_LIBRARIES})
ament_target_dependencies(nanoflann)

# NanoGICP
add_library(nano_gicp STATIC
  src/nano_gicp/lsq_registration.cc
  src/nano_gicp/nano_gicp.cc
)
target_link_libraries(nano_gicp ${PCL_LIBRARIES} ${OpenMP_LIBS} nanoflann)
ament_target_dependencies(nano_gicp)

# Odometry Node
add_executable(dlio_odom_node src/dlio/odom_node.cc src/dlio/odom.cc)
target_link_libraries(dlio_odom_node ${PCL_LIBRARIES} ${OpenMP_LIBS} Threads::Threads nano_gicp)
ament_target_dependencies(dlio_odom_node "tf2_ros" "pcl_ros" "nav_msgs" "pcl_conversions" "rclcpp" "std_msgs" "sensor_msgs" "geometry_msgs" "visualization_msgs" "std_srvs")

# Mapping Node
add_executable (dlio_map_node src/dlio/map_node.cc src/dlio/map.cc)
target_link_libraries(dlio_map_node ${PCL_LIBRARIES}  ${cpp_typesupport_target})
ament_target_dependencies(dlio_map_node "tf2_ros" "pcl_ros" "pcl_conversions" "rclcpp" "nav_msgs" "std_srvs")

if(ENABLE_CLANG_TIDY)
  file(GLOB_RECURSE DLIO_CLANG_TIDY_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
  )

  add_custom_target(clang_tidy ALL
    COMMAND "${RUN_CLANG_TIDY_EXE}"
      -quiet
      -clang-tidy-binary "${CLANG_TIDY_EXE}"
      -p "${CMAKE_BINARY_DIR}"
      -extra-arg-before=--driver-mode=g++
      ${DLIO_CLANG_TIDY_SOURCES}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running standalone clang-tidy analysis"
    USES_TERMINAL
    VERBATIM
  )

  if(TARGET ${PROJECT_NAME}__cpp)
    add_dependencies(clang_tidy ${PROJECT_NAME}__cpp)
  endif()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME} nano_gicp nanoflann)
ament_export_dependencies(rclcpp std_msgs sensor_msgs geometry_msgs pcl_ros)
ament_package()

# Binaries
install( TARGETS dlio_odom_node dlio_map_node
         DESTINATION lib/${PROJECT_NAME} 
)
install( DIRECTORY cfg launch
         DESTINATION share/${PROJECT_NAME} 
)

add_compile_options(-std=c++17 -O3 -mtune=native -march=native -funroll-loops -fopenmp -fopt-info-vec-optimized)

link_directories(${PCL_LIBRARY_DIRS})

include(FindOpenMP)
if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
  message("ERROR: OpenMP could not be found.")
endif(OPENMP_FOUND)
file(WRITE ${CMAKE_BINARY_DIR}/test_cpuid.cpp "#include <cpuid.h>")
try_compile(HAS_CPUID ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
file(REMOVE ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
if(HAS_CPUID)
  add_compile_definitions(HAS_CPUID)
endif()

target_compile_options(dlio_odom_node PRIVATE ${OpenMP_FLAGS})

target_compile_options(dlio_map_node PRIVATE ${OpenMP_FLAGS})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_imu_timestamp_tracker
    test/test_imu_timestamp_tracker.cpp
  )
  target_include_directories(test_imu_timestamp_tracker PRIVATE include)
endif()
