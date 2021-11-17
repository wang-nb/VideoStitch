#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/util.hpp"

/*
 * 默认使用BlocksGainCompensator
 */
class MyExposureCompensator
{
public:
    MyExposureCompensator(int bl_width = 32, int bl_height = 32)
        : bl_width_(bl_width), bl_height_(bl_height) {}

    void createWeightMaps(const std::vector<cv::Point> &corners, const std::vector<cv::Mat> &images,
                          const std::vector<cv::Mat> &masks, std::vector<cv::Mat_<float>> &ec_maps);

    void createWeightMaps(const std::vector<cv::Point> &corners, const std::vector<cv::Mat> &images,
                          const std::vector<std::pair<cv::Mat, uchar>> &masks, std::vector<cv::Mat_<float>> &ec_maps);

    void feed(const std::vector<cv::Point> &corners, const std::vector<cv::Mat> &images,
              std::vector<cv::Mat> &masks);

    void gainMapResize(std::vector<cv::Size> sizes_, std::vector<cv::Mat_<float>> &ec_maps);

    void apply(int index, cv::Mat &image);

private:
    int bl_width_, bl_height_;
    std::vector<cv::Mat_<float>> ec_maps_;
};