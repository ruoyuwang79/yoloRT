# cmake version limitation
cmake_minimum_required(VERSION 2.8)

# set the mode to release
set(CMAKE_BUILD_TYPE Release)

# project name
project(yoloRT)

# set flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors")

# set source and include directory
set(SRC_DIR src)
include_directories(${SRC_DIR}/include)

# add wrapper
set(TRT_WRAPPER_DIR ${SRC_DIR}/wrapper)
add_subdirectory(${TRT_WRAPPER_DIR})

# add auxiliary functions
set(AUX_DIR ${SRC_DIR}/aux)
add_subdirectory(${AUX_DIR})

# add opencv
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES_DIRS})

# build runYolov3
aux_source_directory(${SRC_DIR} OBJ)
add_executable(yoloRT ${OBJ})
target_link_libraries(yoloRT TrtNet Aux ${OpenCV_LIBS})

# specify install position
install(TARGETS yoloRT DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/build)
