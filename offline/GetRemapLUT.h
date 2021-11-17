#pragma once
/**********************************************************
Filename: GetRemapLut.h
Author:   wangnb 
Version:  v0.1
Description: 
Function List: 
            1.
History:
       wangnb  2021-11-15 19:40   0.1  
       
************************************************************/


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/stitching/detail/autocalib.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/camera.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/util.hpp>
#include <opencv2/stitching/detail/warpers.hpp>
#include <opencv2/stitching/warpers.hpp>

#include "blender.h"
#include "logging.hpp"

#define BUFFER_SIZE 1
#define STITCH_SUCCESS 0
#define STITCH_CONFIG_ERROR -1
#define STITCH_NOISE -2

class StitcherRemap
{
public:
    StitcherRemap();
    ~StitcherRemap();
    void setTrim(bool is_trim)
    {
        if (is_trim)
            trim_type_ = StitcherRemap::TRIM_AUTO;
        else
            trim_type_ = StitcherRemap::TRIM_NO;
    };
    void setTrim(cv::Rect trim_rect)
    {
        trim_rect_ = trim_rect;
        trim_type_ = StitcherRemap::TRIM_RECTANGLE;
    };
    void setWarpType(std::string warp_type) { warp_type_ = warp_type; };

    int stitch(std::vector<cv::Mat> &imgs, std::string &save_path);

    void saveCameraParam(const std::string &filename);
    int loadCameraParam(const std::string &filename);

private:
    int saveRemap(const std::string &save_path);
    int Prepare(std::vector<cv::Mat> &src);
    int PrepareClassical(std::vector<cv::Mat> &src);
    int StitchFrame(std::vector<cv::Mat> &src, cv::Mat &dst);
    int StitchFrameCPU(std::vector<cv::Mat> &src, cv::Mat &dst);

    /*
     * 计算一些放缩的尺度，在特征检测和计算接缝的时候，为了提高程序效率，可以对源图像进行一些放缩
     */
    void SetScales(std::vector<cv::Mat> &src);

    int FindFeatures(std::vector<cv::Mat> &src, std::vector<cv::detail::ImageFeatures> &features);

    /*
     * 特征匹配，然后去除噪声图片。本代码实现时，一旦出现噪声图片，就终止算法
     * 返回值：
     *		0	——	正常
     *		-2	——	存在噪声图片
     */
    int MatchImages(std::vector<cv::detail::ImageFeatures> &features,
                    std::vector<cv::detail::MatchesInfo> &pairwise_matches);

    /*
     * 摄像机标定
     */
    int CalibrateCameras(std::vector<cv::detail::ImageFeatures> &features,
                         std::vector<cv::detail::MatchesInfo> &pairwise_matches,
                         std::vector<cv::detail::CameraParams> &cameras);

    /*
     *	计算水平视角
     */
    double GetViewAngle(std::vector<cv::Mat> &src, std::vector<cv::detail::CameraParams> &cameras);


    /*
     * 为接缝的计算做Warp
     */
    int WarpForSeam(std::vector<cv::Mat> &src, std::vector<cv::detail::CameraParams> &cameras,
                    std::vector<cv::Mat> &masks_warped, std::vector<cv::Mat> &images_warped);

    /*
     * 计算接缝
     */
    int FindSeam(std::vector<cv::Mat> &images_warped, std::vector<cv::Mat> &masks_warped);

    /*
     *	把摄像机参数和masks还原到正常大小
     */
    int Rescale(std::vector<cv::Mat> &src, std::vector<cv::detail::CameraParams> &cameras,
                std::vector<cv::Mat> &seam_masks);

    int RegistEvaluation(std::vector<cv::detail::ImageFeatures> &features,
                         std::vector<cv::detail::MatchesInfo> &pairwise_matches,
                         std::vector<cv::detail::CameraParams> &cameras);

    /*
     *	解决360°拼接问题。对于横跨360°接缝的图片，找到最宽的inpaint区域[x1, x2]
     */
    int FindWidestInpaintRange(cv::Mat mask, int &x1, int &x2);

    /*
     * 裁剪掉inpaint区域
     */
    int TrimRect(cv::Rect rect);
    int TrimInpaint(std::vector<cv::Mat> &src);
    bool IsRowCrossInpaint(uchar *row, int width);

    /* 裁剪类型 */
    enum { TRIM_NO,
           TRIM_AUTO,
           TRIM_RECTANGLE };

    /* 参数 */
    int trim_type_;
    cv::Rect trim_rect_;

    float work_megapix_;
    float seam_megapix_;
    float conf_thresh_;
    std::string features_type_;
    std::string ba_cost_func_;
    std::string ba_refine_mask_;
    bool is_do_wave_correct_;
    cv::detail::WaveCorrectKind wave_correct_;
    std::string warp_type_;
    float match_conf_;
    std::string seam_find_type_;

    cv::Ptr<cv::WarperCreator> warper_creator_;
    float work_scale_, seam_scale_;
    float median_focal_len_;

    /* 第一帧计算出的参数，不用重复计算 */
    cv::Rect dst_roi_;
    std::vector<cv::detail::CameraParams> cameras_;
    std::vector<int> src_indices_;
    std::vector<cv::Point> corners_;
    std::vector<cv::Size> sizes_;
    std::vector<cv::Mat> final_warped_masks_;//warp的mask
    std::vector<cv::Mat> xmaps_;
    std::vector<cv::Mat> ymaps_;
    std::vector<cv::Mat> blend_weight_maps_;
    std::vector<cv::Mat_<float>> total_weight_maps_;
    std::vector<cv::Mat> final_blend_masks_;//blend_mask = seam_mask & warp_mask
    float view_angle_;
    float blend_strength_;
    MyFeatherBlender blender_;

    /* 缓存 */
    std::vector<cv::Mat> final_warped_images_;
    int parallel_num_;
    bool is_prepared_;
};
