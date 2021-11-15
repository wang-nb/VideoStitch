//
// Created by bing on 2021/11/10.
//
#include "logging.hpp"
#include "opencv2/opencv.hpp"
#include "video_stitcher.h"

int stitchVideo(const std::vector<std::string> &video_names,
                const std::string &save_video, MyVideoStitcher &videoStitcher)
{
    int flg = 0;
    std::vector<cv::VideoCapture> caps;
    for (int i = 0; i < video_names.size(); i++) {
        cv::VideoCapture video(video_names[i]);
        if (!video.isOpened()) {
            flg = -1;
            LOG(INFO) << "failed to open video :" << video_names[i];
            break;
        }
        caps.push_back(video);
    }
    if (0 == flg) {
        int video_nums = (int) video_names.size();
        std::vector<cv::Mat> frames(video_nums);
        cv::Mat pano;
        cv::Size dst_size;
        videoStitcher.getDstSize(dst_size);
        cv::VideoWriter mp4;
        mp4.open(save_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                 25, dst_size);
        while (true) {
            int flag = 0;
            for (int i = 0; i < video_nums; i++) {
                caps[i] >> frames[i];
                if (frames[i].empty()) {
                    flag = -1;
                }
            }
            if (0 == flag) {
                videoStitcher.stitchImage(frames, pano);
                cv::imshow("pano", pano);
                mp4 << pano;
                int key = cv::waitKey(1);
                if (27 == key) {
                    break;
                }
            } else {
                break;
            }
        }
        mp4.release();
    }
    return flg;
}

int main(int argc, char *argv[])
{
    std::string config_path   = "data/";
    std::string video_path    = "data/video1";
    std::string video_pattern = "mp4";
    std::string save_mp4_path = "data/result.mp4";
    for (int i = 1; i < argc; i++) {
        if (1 == i) {
            config_path = static_cast<std::string>(argv[1]);
        } else if (2 == i) {
            video_path = static_cast<std::string>(argv[2]);
        } else if (3 == i) {
            video_pattern = static_cast<std::string>(argv[3]);
        } else if (4 == i) {
            save_mp4_path = static_cast<std::string>(argv[4]) + ".mp4";
        }
    }
    std::vector<std::string> video_names;
    cv::glob(video_path + "/*" + video_pattern, video_names);
    if (video_names.size() < 2) {
        LOG(INFO) << "can not find enough videos about " << video_pattern;
    } else {
        MyVideoStitcher videoStitcher;
        videoStitcher.init(config_path);
        stitchVideo(video_names, save_mp4_path, videoStitcher);
    }
    return 0;
}