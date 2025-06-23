################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 3.8)
project(IBEC3)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
################################################################################
# Find ament packages and libraries for ament and system dependencies
################################################################################
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(Pangolin REQUIRED)
find_package(yaml-cpp REQUIRED)
################################################################################
# Build
################################################################################
# ------- RCLCPP (plain style only) -------


# ------- C++ -------
add_executable(ibec3
  src/main.cpp
  src/EventData/H5Reader.cpp
  src/EventData/EventArray.cpp
)
target_link_libraries(ibec3 PRIVATE
  ${OpenCV_LIBS}
  Eigen3::Eigen
  Ceres::ceres
  ${HDF5_LIBRARIES}
  ${Pangolin_LIBRARIES}
  yaml-cpp
)
target_include_directories(ibec3 PRIVATE
  include
  ${HDF5_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
)
################################################################################
# Install
################################################################################
# lib/<package>: executable binaries
install(
  TARGETS
    ibec3
  DESTINATION lib/${PROJECT_NAME}
)
# share/<package>: launch files and other resources
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)
