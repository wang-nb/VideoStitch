//
// Created by bing on 2022/1/22.
//
#include "opencv2/opencv.hpp"
//日志
#include "logging.hpp"
#include <string>

void stitchImg(const std::vector<cv::Mat>& imgs, cv::Mat& pano)
{
    //设置拼接图像 warp 模式，有PANORAMA与SCANS两种模式
    //panorama: 图像会投影到球面或者柱面进行拼接
    //scans: 默认没有光照补偿与柱面等投影，直接经过仿射变换进行拼接
    cv::Stitcher::Mode mode = cv::Stitcher::SCANS;
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(mode);
    cv::Stitcher::Status status = stitcher->stitch(imgs, pano);
    if(cv::Stitcher::OK != status){
        LOG(INFO) << "failed to stitch images, err code: " << (int)status;
    }
}

int main(int argc, char* argv[])
{
    std::string pic_path = "data/img/*";
    std::string pic_pattern = ".jpg";

    if(2 == argc){
        pic_path = std::string(argv[1]);
    }else if(3 == argc){
        pic_path = std::string(argv[1]);
        pic_pattern = std::string(argv[2]);
    }else{
        LOG(INFO) << "default value";
    }
    std::vector<cv::String> img_names;
    std::vector<cv::Mat> imgs;
    pic_pattern = pic_path + pic_pattern;
    cv::glob(pic_pattern, img_names);
    if(img_names.empty()){
        LOG(INFO) << "no images";
        return -1;
    }
    for(size_t i = 0; i < img_names.size(); ++i){
        cv::Mat img = cv::imread(img_names[i]);
        imgs.push_back(img.clone());
    }
    cv::Mat pano;
    stitchImg(imgs, pano);
    if(!pano.empty()){
        cv::imshow("pano", pano);
        cv::imwrite("pano.jpg", pano);
        cv::waitKey(0);
    }
    return 0;
}
