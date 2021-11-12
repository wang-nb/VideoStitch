#include <iostream>

#include <time.h>

#include "opencv2/highgui/highgui.hpp"

#include "../offline/get_remap.h"

using namespace cv;
using namespace std;


static void printUsage()
{
    cout << "��Ƶƴ��.\n\n"
            "VideoStitch [flags]\n"
            "flags:\n"
            "    image image ...		�궨ģʽ�������ͼƬ\n"
            "    -plane			����ƽ��ͶӰ������ƴ���ӽ�С��140��ʱ����\n"
            "    -trim			���Բü�δ������򣬽���ƽ��ͶӰʱ����\n"
            "    --trim  x1 y1 x2 y2		����x1 y1 x2 y2���ɵľ��βü����ս��\n"
            "    --cp camera_param_path	ʹ��camera_param_path�����������\n";
}
static vector<string> video_names;
static bool is_trim = false, is_trim_rect = false;
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

static int VideoStitch(int argc, char *argv[])
{
    for (int i = 0; i < argc; i++)
        printf("%s\n", argv[i]);
    int retval = parseCmdArgs(argc, argv);
    if (retval)
        return retval;

    //	������Ƶ��
    vector<cv::Mat> imgs;
    int video_num = video_names.size();
    imgs.resize(video_num);
    for (int i = 0; i < video_num; i++) {
        imgs[i] = cv::imread(video_names[i]);
        if (imgs[i].empty()) {
            cout << "Fail to open " << video_names[i] << endl;
            return -1;
        }
    }
    cout << "Video capture success" << endl;

    StitcherRemap video_stitcher;
    //	ƴ�Ӳ���
    video_stitcher.setTrim(is_trim);
    if (is_trim_rect)
        video_stitcher.setTrim(trim_rect);
    video_stitcher.setWarpType(warp_type);

    //	ƴ��
    std::string save_path = "data/";
    video_stitcher.stitch(imgs, save_path);
    return 0;
}

int main(int argc, char *argv[])
{
    VideoStitch(argc, argv);
    return 0;
}
