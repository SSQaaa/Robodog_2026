cmake_minimum_required(VERSION 3.10)

project(yolov5)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)  # 改为 Release 提升性能

# CUDA 10.2
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

# ========== 修改 1: CUDA 路径 ==========
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# ========== 修改 2: TensorRT 7.1 路径 ==========
# 系统默认安装路径（不是 /home/nvidia/...）
include_directories(/usr/include/x86_64-linux-gnu)
link_directories(/usr/lib/x86_64-linux-gnu)

# 如果上面不行，尝试这个（根据你的实际安装路径）：
# include_directories(/usr/local/TensorRT-7.1.0.16/include)
# link_directories(/usr/local/TensorRT-7.1.0.16/lib)

include_directories(${PROJECT_SOURCE_DIR}/src/)
include_directories(${PROJECT_SOURCE_DIR}/plugin/)
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE PLUGIN_SRCS ${PROJECT_SOURCE_DIR}/plugin/*.cu)

add_library(myplugins SHARED ${PLUGIN_SRCS})
target_link_libraries(myplugins nvinfer nvinfer_plugin cudart)  # 添加 nvinfer_plugin

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(yolov5_det yolov5_det.cpp ${SRCS})
target_link_libraries(yolov5_det nvinfer nvinfer_plugin cudart myplugins ${OpenCV_LIBS})

add_executable(yolov5_cls yolov5_cls.cpp ${SRCS})
target_link_libraries(yolov5_cls nvinfer nvinfer_plugin cudart myplugins ${OpenCV_LIBS})

add_executable(yolov5_seg yolov5_seg.cpp ${SRCS})
target_link_libraries(yolov5_seg nvinfer nvinfer_plugin cudart myplugins ${OpenCV_LIBS})