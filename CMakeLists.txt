cmake_minimum_required(VERSION 3.10)

PROJECT(video_stitch)

option(USE_GPU "use cuda" OFF)

IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    #OpenCV
ELSEIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
ELSEIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(THIRD_PARTY_PATH F:/3-thirdparty)
ELSE ()
    message(FATAL "Unsupported platforms")
ENDIF ()

include(cmake/opencv.cmake)

add_definitions(-DSTITCHER_DEBUG -D_CRT_SECURE_NO_WARNINGS)

include_directories(offline util)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(test)

