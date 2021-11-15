﻿#pragma once
/**********************************************************
Filename: blend.h
Author:   wangnb 
Version:  v0.1
Description: 
Function List: 
            1.
History:
       wangnb  2021-11-15 19:43   0.1  
       
************************************************************/
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/util.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;


class MyFeatherBlender
{
public:
    void setImageNum(int image_num)
    {
        weight_maps_.resize(image_num);
    }

    float sharpness() const { return m_sharpness_; }

    void setSharpness(float val) { m_sharpness_ = val; }

    void createWeightMaps(Rect dst_roi, vector<Point> corners, vector<Mat> &masks, vector<Mat> &weight_maps);

    void prepare(Rect dst_roi, vector<Point> corners, vector<Mat> &masks);

    void clear()
    {
        dst_.setTo(Scalar::all(0));
    }

    void feed(const Mat &img, const Mat &mask, Point tl, int img_idx);

    void blend(Mat &dst, Mat &dst_mask);

private:
    int m_image_num = 0;
    float m_sharpness_ = 1.0f;
    vector<Mat> weight_maps_;
    Mat dst_weight_map_;

protected:
    Mat dst_, dst_mask_;
    Rect dst_roi_;
};