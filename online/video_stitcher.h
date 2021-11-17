#pragma once
/**********************************************************
Filename: video_stitcher.h
Author:   wangnb 
Version:  v0.1
Description: 
Function List: 
            1.
History:
       wangnb  2021-11-15 19:47   0.1  
       
************************************************************/

#include "logging.hpp"
#include "opencv2/opencv.hpp"

class MyVideoStitcher
{
public:
    int init(const std::string &config_path);
    int stitchImage(std::vector<cv::Mat> &src, cv::Mat &pano);
    int getDstSize(cv::Size &dst_size);

private:
    int StitchFrameCPU(std::vector<cv::Mat> &src, cv::Mat &dst);

    /* 参数 */
    std::vector<int> src_indices_;
    std::vector<cv::Point> corners_;
    std::vector<cv::Size> sizes_;
    std::vector<cv::Mat> xmaps_;
    std::vector<cv::Mat> ymaps_;
    std::vector<cv::Mat_<float>> total_weight_maps_;
    cv::Rect dst_roi_;
    /* 缓存 */
    std::vector<cv::Mat> final_warped_images_;
    int video_num_;
    int parallel_num_;
};
