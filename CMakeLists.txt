cmake_minimum_required(VERSION 3.10)  # Or the version you need

# Set the project name and version
project(KalmanLab VERSION 1.0)

# Set the C++ standard to C++17 (or later)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Specify the path to Eigen
set(EIGEN3_INCLUDE_DIR "/home/szajnecki/Projects/eigen-3.4.0")

# Add Eigen to include directories
include_directories(${EIGEN3_INCLUDE_DIR})

# Add the include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Alternatively, using target_include_directories (more modern and preferred)
add_executable(KalmanLab 
    src/KalmanFilter.cpp
    src/KalmanLab.cpp)

# Add the include directory for a specific target
target_include_directories(KalmanLab PRIVATE ${PROJECT_SOURCE_DIR}/include)
