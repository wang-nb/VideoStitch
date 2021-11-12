set(MODULE_NAME get_lut)
aux_source_directory(../offline STITCHER_SRCS)
add_executable(${MODULE_NAME} getLUTMain.cpp ${STITCHER_SRCS})
target_link_libraries(${MODULE_NAME} ${OpenCV_LIBS})

aux_source_directory(../online VIDEO_STITCHER_SRCS)
include_directories(../online)
add_executable(video_stitch videoStitchingMain.cpp ${VIDEO_STITCHER_SRCS})
target_link_libraries(video_stitch ${OpenCV_LIBS})