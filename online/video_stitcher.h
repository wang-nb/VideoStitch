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

using namespace std;
using namespace cv;

class MyVideoStitcher
{
public:
    int init(const std::string &config_path);
    int stitchImage(vector<Mat> &src, Mat &pano);
    int getDstSize(cv::Size &dst_size);

private:
    int StitchFrameCPU(vector<Mat> &src, Mat &dst);

    /* 参数 */
    vector<int> src_indices_;
    vector<Point> corners_;
    vector<Size> sizes_;
    Rect dst_roi_;
    vector<Mat> xmaps_;
    vector<Mat> ymaps_;
    vector<Mat_<float>> total_weight_maps_;

    /* 缓存 */
    vector<Mat> final_warped_images_;
    int video_num_;
    int parallel_num_;
};
