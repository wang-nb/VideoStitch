﻿//
// Created by Administrator on 2021-11-15.
//
#include "opencv2/opencv.hpp"
#include "GetRemapLUT.h"

static void printUsage()
{
    std::cout << "视频拼接.\n\n"
                 "VideoStitch [flags]\n"
                 "flags:\n"
                 "    image image ...		标定模式下输入的图片\n"
                 "    -plane			尝试平面投影，仅在拼接视角小于140°时可用\n"
                 "    -trim			尝试裁剪未填充区域，仅在平面投影时可用\n"
                 "    --trim  x1 y1 x2 y2		按照x1 y1 x2 y2构成的矩形裁剪最终结果\n"
                 "    --cp camera_param_path	使用camera_param_path的摄像机参数\n";
}
static std::vector<std::string> video_names;
static bool is_trim = false, is_trim_rect = false;
static std::string warp_type = "cylindrical";
static cv::Rect trim_rect;
static int parseCmdArgs(int argc, char *argv[])
{
    if (argc == 1) {
        printUsage();
        return -1;
    }

    video_names.clear();
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "/?") {
            printUsage();
            return -1;
        } else if (std::string(argv[i]) == "-trim")
            is_trim = true;
        else if (std::string(argv[i]) == "--trim") {
            is_trim = is_trim_rect = true;
            int x1                 = atoi(argv[i + 1]);
            int y1                 = atoi(argv[i + 2]);
            int x2                 = atoi(argv[i + 3]);
            int y2                 = atoi(argv[i + 4]);
            trim_rect              = cv::Rect(x1, y1, x2 - x1, y2 - y1);
            i += 4;
        } else if (std::string(argv[i]) == "-plane")
            warp_type = "plane";
        else
            video_names.push_back(argv[i]);
    }
    return 0;
}

static int VideoStitch(int argc, char *argv[])
{
    for (int i = 0; i < argc; i++)
        printf("%s\n", argv[i]);
    int retval = parseCmdArgs(argc, argv);
    if (retval)
        return retval;

    //	输入视频流
    std::vector<cv::Mat> imgs;
    size_t video_num = video_names.size();
    imgs.resize(video_num);
    for (size_t i = 0; i < video_num; i++) {
        imgs[i] = cv::imread(video_names[i]);
        if (imgs[i].empty()) {
            std::cout << "Fail to open " << video_names[i] << std::endl;
            return -1;
        }
    }
    std::cout << "Video capture success" << std::endl;

    StitcherRemap video_stitcher;
    //	拼接参数
    video_stitcher.setTrim(is_trim);
    if (is_trim_rect)
        video_stitcher.setTrim(trim_rect);
    video_stitcher.setWarpType(warp_type);

    //	拼接
    std::string save_path = "data/";
    video_stitcher.stitch(imgs, save_path);
    return 0;
}

int main(int argc, char *argv[])
{
    VideoStitch(argc, argv);
    return 0;
}
