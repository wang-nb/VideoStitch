#ifndef __MY_VIDEO_STITCHER_H__
#define __MY_VIDEO_STITCHER_H__

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

#endif