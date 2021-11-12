﻿IF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
ELSEIF ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(OpenCV_INCLUDE_DIRS ${THIRD_PARTY_PATH}/opencv/include)
    set(Opencv_LIB_DIRS ${THIRD_PARTY_PATH}/opencv/lib)
    if (CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
        set(OpenCV_LIBS opencv_world453.lib)
    else ()
        set(OpenCV_LIBS opencv_world453d.lib)
    endif ()
ELSE ()
    message(FATAL "Unsupported platforms")
ENDIF ()

link_directories(${Opencv_LIB_DIRS})