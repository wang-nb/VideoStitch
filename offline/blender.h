#pragma once
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

class MyFeatherBlender
{
public:
    void setImageNum(int image_num)
    {
        weight_maps_.resize(image_num);
    }

    float sharpness() const { return m_sharpness_; }

    void setSharpness(float val) { m_sharpness_ = val; }

    void createWeightMaps(cv::Rect dst_roi, std::vector<cv::Point> corners,
                          std::vector<cv::Mat> &masks, std::vector<cv::Mat> &weight_maps);

    void prepare(cv::Rect dst_roi, std::vector<cv::Point> corners,
                 std::vector<cv::Mat> &masks);

    void clear()
    {
        dst_.setTo(cv::Scalar::all(0));
    }

    void feed(const cv::Mat &img, const cv::Mat &mask, cv::Point tl, int img_idx);

    void blend(cv::Mat &dst, cv::Mat &dst_mask);

private:
    int m_image_num = 0;
    float m_sharpness_ = 1.0f;
    std::vector<cv::Mat> weight_maps_;
    cv::Mat dst_weight_map_;

protected:
    cv::Mat dst_, dst_mask_;
    cv::Rect dst_roi_;
};