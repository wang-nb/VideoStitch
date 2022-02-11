//
// Created by bing on 2021/12/31.
//
#include "ORBextractor.h"
#include "logging.hpp"
#include "opencv2/opencv.hpp"

int main()
{
    std::string root_path    = "data/img/";
    std::string pic_pattern  = "medium11.jpg";
    std::string feature_type = "orb";

    std::string img_names = root_path + pic_pattern;

    cv::Ptr<cv::FeatureDetector> finder;
    if (feature_type == "orb") {
        finder = cv::ORB::create();
    } else if (feature_type == "sift") {
        finder = cv::SIFT::create();
    } else {
        LOG(INFO) << "Unknown 2D features type: '" << feature_type << "'.\n";
        LOG(INFO) << "use default ORB\n";
        finder = cv::ORB::create();
    }
    //opencv orb检测函数
    cv::Mat img = cv::imread(img_names);
    if (img.empty()) {
        LOG(INFO) << "failed to load image : " << img_names;
        return -1;
    }
    std::vector<cv::detail::ImageFeatures> features(2);
    NEW_TIME_VALUE
    START_GETTIME
    for (int i = 0; i < 10; i++) {
        finder->detectAndCompute(img, cv::Mat(), features[0].keypoints,
                                 features[0].descriptors, false);
    }
    END_GETTIME(10, "opencv extract orb , time cost ")
    features[0].img_idx = 0;
    std::vector<cv::Mat> feature_img(2);
    cv::drawKeypoints(img, features[0].keypoints, feature_img[0]);
    LOG(INFO) << "cv orb feature num: " << features[0].keypoints.size();

    ORB_SLAM3::ORBextractor orb_extractor = ORB_SLAM3::ORBextractor(
            500, 2, 4, 20, 10);
    std::vector<int> vlapp{30};
    cv::Mat gray_img;
    cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    START_GETTIME
    for (int i = 0; i < 10; i++) {
        orb_extractor(gray_img, cv::Mat(), features[1].keypoints,
                      features[1].descriptors, vlapp);
    }
    END_GETTIME(10, "slam extract orb , time cost ")
    cv::drawKeypoints(img, features[1].keypoints, feature_img[1]);
    LOG(INFO) << "slam orb feature num: " << features[1].keypoints.size();

    cv::imshow("cv_orb", feature_img[0]);
    cv::imshow("slam_orb", feature_img[1]);
    cv::waitKey(0);
    return 0;
}