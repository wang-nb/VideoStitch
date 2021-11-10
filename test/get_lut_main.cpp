#include <iostream>

#include <time.h>

#include "opencv2/highgui/highgui.hpp"

#include "../offline/video_stitcher.h"

using namespace cv;
using namespace std;


static void printUsage()
{
    cout << "视频拼接.\n\n"
            "VideoStitch [flags]\n"
            "flags:\n"
            "    video1 video2 ...		视频模式，输入视频路径（视频模式和摄像机模式只能开启一种）\n"
            "    --range start end		拼接范围，从start到end帧，end=-1表示拼接到结尾\n"
            "    -plane			尝试平面投影，仅在拼接视角小于140°时可用\n"
            "    -trim			尝试裁剪未填充区域，仅在平面投影时可用\n"
            "    --trim  x1 y1 x2 y2		按照x1 y1 x2 y2构成的矩形裁剪最终结果\n"
            "    --cp camera_param_path	使用camera_param_path的摄像机参数\n";
}
static vector<string> video_names;
static bool is_trim = false, is_trim_rect = false;
static int range_start = 0, range_end = -1;
static string warp_type = "cylindrical";
static Rect trim_rect;
static int parseCmdArgs(int argc, char *argv[])
{
    if (argc == 1) {
        printUsage();
        return -1;
    }

    video_names.clear();
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?") {
            printUsage();
            return -1;
        } else if (string(argv[i]) == "--range") {
            range_start = atoi(argv[i + 1]);
            range_end   = atoi(argv[i + 2]);
            i += 2;
        } else if (string(argv[i]) == "-trim")
            is_trim = true;
        else if (string(argv[i]) == "--trim") {
            is_trim = is_trim_rect = true;
            int x1                 = atoi(argv[i + 1]);
            int y1                 = atoi(argv[i + 2]);
            int x2                 = atoi(argv[i + 3]);
            int y2                 = atoi(argv[i + 4]);
            trim_rect              = Rect(x1, y1, x2 - x1, y2 - y1);
            i += 4;
        } else if (string(argv[i]) == "-plane")
            warp_type = "plane";
        else
            video_names.push_back(argv[i]);
    }
    return 0;
}

//两个样例：
//	VideoStitch data/6-2/my_cam_0.avi data/6-2/my_cam_1.avi data/6-2/my_cam_2.avi data/6-2/my_cam_3.avi data/6-2/my_cam_4.avi data/6-2/my_cam_5.avi -v -gpu
//	VideoStitch --camera 5 1280 720 -v -gpu --debug data/tmp/ --cp data/tmp/camera_param_5.dat
static int VideoStitch(int argc, char *argv[])
{
    for (int i = 0; i < argc; i++)
        printf("%s\n", argv[i]);
    int retval = parseCmdArgs(argc, argv);
    if (retval)
        return retval;

    for (int i = 0; i < video_names.size(); i++)
        cout << video_names[i] << endl;

    //	输入视频流
    vector<VideoCapture> captures;
    int video_num = video_names.size();
    captures.resize(video_num);
    for (int i = 0; i < video_num; i++) {
        captures[i].open(video_names[i]);
        if (!captures[i].isOpened()) {
            cout << "Fail to open " << video_names[i] << endl;
            for (int j = 0; j < i; j++) captures[j].release();
            return -1;
        }
    }
    cout << "Video capture success" << endl;

    MyVideoStitcher video_stitcher;

    //	显示/保存
    video_stitcher.setRange(range_start, range_end);

    //	拼接参数
    video_stitcher.setTrim(is_trim);
    if (is_trim_rect)
        video_stitcher.setTrim(trim_rect);
    video_stitcher.setWarpType(warp_type);

    //	拼接
    std::string save_path = "data/";
    video_stitcher.stitch(captures, save_path);

    //	释放资源
    for (int i = 0; i < captures.size(); i++)
        captures[i].release();

    cout << "Released all" << endl;

    return 0;
}

int main(int argc, char *argv[])
{
    VideoStitch(argc, argv);
    system("pause");
    return 0;
}
