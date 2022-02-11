#include "opencv2/opencv.hpp"
#include "ORBextractor.h"
#include "logging.hpp"
#include <string>
#include <vector>
#include <fstream>

int MatchImages(std::vector<cv::detail::ImageFeatures> &features,
                std::vector<cv::detail::MatchesInfo> &pairwise_matches)
{
    int total_num_images = static_cast<int>(features.size());
    float match_conf_ = 0.5f;
    float conf_thresh_ = 1.0f;
    cv::detail::BestOf2NearestMatcher matcher(false, match_conf_);
    matcher(features, pairwise_matches);
    matcher.collectGarbage();

    std::vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh_);

    int num_images = (int)(indices.size());
    if (num_images != total_num_images) {
        fprintf(stderr, " videos are invaild");
        return -1;
    }

    return 0;
}

int featureMatches(const std::vector<cv::Mat>& imgs,
                   std::vector<cv::detail::ImageFeatures>& features,
                   std::vector<cv::detail::MatchesInfo> &pairwise_matches)
{
    std::vector<int> vlapp{30};
    ORB_SLAM3::ORBextractor orb_extractor = ORB_SLAM3::ORBextractor(
            500, 1.5, 8, 20, 10);
    features.clear();
    features.resize(imgs.size());
    pairwise_matches.clear();
    pairwise_matches.resize(imgs.size());
    for (int i = 0; i < imgs.size(); i++) {
        features[i].img_idx = i;
        features[i].img_size = imgs[i].size();
        cv::Mat gray_img;
        cv::cvtColor(imgs[i], gray_img, cv::COLOR_BGR2GRAY);
        orb_extractor(gray_img, cv::Mat(), features[i].keypoints,
                      features[i].descriptors, vlapp);
    }
    MatchImages(features, pairwise_matches);
    cv::Mat pairwiseImg;
    drawMatches(imgs[0], features[0].keypoints, imgs[1], features[1].keypoints,
                pairwise_matches[1].matches, pairwiseImg, cv::Scalar::all(-1),
                cv::Scalar::all(-1), std::vector<char>(),
                cv::DrawMatchesFlags::DEFAULT);
    return 0;
}

//	保存摄像机参数，文件格式如下：
//	第一行是中间焦距median_focal_len_
//	之后每一行是一个相机--
//		数据依次是focal、aspect、ppx、ppy、R、t
void saveCameraParam(const std::string& filename, float median_focal,
                     std::vector<cv::detail::CameraParams>& cameras)
{
    std::ofstream cp_file(filename.c_str());
    cp_file << median_focal << std::endl;
    for (auto cp : cameras) {
        cp_file << cp.focal << " " << cp.aspect << " " << cp.ppx << " " << cp.ppy;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                cp_file << " " << cp.R.at<float>(r, c);
        for (int r = 0; r < 3; r++)
            cp_file << " " << cp.t.at<double>(r, 0);
        cp_file << std::endl;
    }
    cp_file.close();
}
/*
 * 摄像机标定
 */
int calibrateCameras(std::vector<cv::detail::ImageFeatures> &features,
                     std::vector<cv::detail::MatchesInfo> &pairwise_matches,
                     std::vector<cv::detail::CameraParams> &cameras,
                     float& median_focal)
{
    cv::detail::HomographyBasedEstimator estimator;
    cv::Ptr<cv::detail::BundleAdjusterBase> adjuster;
    cv::Mat_<uchar> refine_mask;
    std::vector<double> focals;
    cameras.clear();
    cameras.resize(features.size());
    LOG(INFO) << "estimate camera instrins";
    estimator(features, pairwise_matches, cameras);

    for (size_t i = 0; i < (int) cameras.size(); ++i) {
        cv::Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;

        std::cout << "src camera R " << cameras[i].R << std::endl;
    }
    std::string ba_cost_func_ = "ray";
    if (ba_cost_func_ == "reproj")
        adjuster = new cv::detail::BundleAdjusterReproj();
    else if (ba_cost_func_ == "ray")
        adjuster = new cv::detail::BundleAdjusterRay();
    adjuster->setConfThresh(1.0f);
    refine_mask = cv::Mat::zeros(3, 3, CV_8U);
    std::string ba_refine_mask_ = "xxxxx";
    if (ba_refine_mask_[0] == 'x') refine_mask(0, 0) = 1;
    if (ba_refine_mask_[1] == 'x') refine_mask(0, 1) = 1;
    if (ba_refine_mask_[2] == 'x') refine_mask(0, 2) = 1;
    if (ba_refine_mask_[3] == 'x') refine_mask(1, 1) = 1;
    if (ba_refine_mask_[4] == 'x') refine_mask(1, 2) = 1;
    LOG(INFO) << "refine camera instrins";
    adjuster->setRefinementMask(refine_mask);
    (*adjuster)(features, pairwise_matches, cameras);

    for (size_t i = 0; i < (int) cameras.size(); ++i) {
        std::cout << "adjuster camera R " << cameras[i].R << std::endl;
    }

    // Find median focal length
    for (size_t i = 0; i < cameras.size(); ++i) {
        focals.push_back(cameras[i].focal);
    }

    sort(focals.begin(), focals.end());
    float median_focal_len_ = 1.0f;
    if (focals.size() % 2 == 1)
        median_focal_len_ = static_cast<float>(focals[focals.size() / 2]);
    else
        median_focal_len_ = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
    median_focal = median_focal_len_;
    bool is_do_wave_correct_ = true;
    LOG(INFO) << "do_wave_correct";
    if (is_do_wave_correct_) {
        std::vector<cv::Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R);
        cv::detail::WaveCorrectKind wave_correct_ = cv::detail::WAVE_CORRECT_HORIZ;
        waveCorrect(rmats, wave_correct_);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }
    return 0;
}

int main()
{
    std::string root_path              = "data/video1/";
    std::string left_pic               = "left.jpg";
    std::string right_pic              = "right.jpg";
    std::vector<std::string> img_names = {root_path + left_pic, root_path + right_pic};
    std::vector<cv::Mat> imgs;
    for (auto name : img_names) {
        cv::Mat img = cv::imread(name);
        if (img.empty()) {
            LOG(FATAL) << "failed to load image : " << name;
        }
        imgs.push_back(img);
    }
    std::vector<cv::detail::ImageFeatures> features;
    std::vector<cv::detail::MatchesInfo> pairwise_matches;
    std::vector<cv::detail::CameraParams> cameras;

    featureMatches(imgs, features, pairwise_matches);
    float median_focal;
    calibrateCameras(features, pairwise_matches, cameras, median_focal);
    saveCameraParam(root_path + "/camera_param.txt",
                    median_focal, cameras);
    return 0;
}