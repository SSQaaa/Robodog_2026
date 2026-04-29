cmake_minimum_required(VERSION 3.10)
project(orbbec_native)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

execute_process(
    COMMAND python3 -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

separate_arguments(PYBIND11_INCLUDES)

add_library(orbbec_native MODULE bindings.cpp)

target_include_directories(orbbec_native PRIVATE
    /home/ysc/Desktop/OrbbecSDK/include
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_options(orbbec_native PRIVATE
    ${PYBIND11_INCLUDES}
)

target_link_libraries(orbbec_native PRIVATE
    /home/ysc/Desktop/OrbbecSDK/lib/arm64/libOrbbecSDK.so
    ${OpenCV_LIBS}
    pthread
)

set_target_properties(orbbec_native PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)