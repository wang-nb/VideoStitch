set(MODULE_NAME get_lut)
aux_source_directory(../offline STITCHER_SRCS)
add_executable(${MODULE_NAME} get_lut_main.cpp ${STITCHER_SRCS})
target_link_libraries(${MODULE_NAME} ${OpenCV_LIBS})