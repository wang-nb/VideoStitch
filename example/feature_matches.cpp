#include "ORBextractor.h"
#include "logging.hpp"
#include "opencv2/opencv.hpp"

/*
 * 特征匹配，然后去除噪声图片。本代码实现时，一旦出现噪声图片，就终止算法
 * 返回值：
 *      0     正常
 *      -2    存在噪声图片
 */
int MatchImages(std::vector<cv::detail::ImageFeatures> &features,
                               std::vector<cv::detail::MatchesInfo> &pairwise_matches)
{
    int total_num_images = static_cast<int>(features.size());
    float match_conf_ = 0.5f;
    float conf_thresh_ = 1.0f;
    cv::detail::BestOf2NearestMatcher matcher(false, match_conf_);
    matcher(features, pairwise_matches);
    matcher.collectGarbage();

    // 去除噪声图像
    std::vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh_);
    // 一旦出现噪声图片，就终止算法

    int num_images = (int)(indices.size());
    if (num_images != total_num_images) {
        fprintf(stderr, " videos are invaild");
        return -1;
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
    //opencv orb检测函数
    std::vector<cv::detail::ImageFeatures> features(2);
    std::vector<cv::Mat> feature_img(2);
    std::vector<int> vlapp{30};
    ORB_SLAM3::ORBextractor orb_extractor = ORB_SLAM3::ORBextractor(
            500, 1.5, 8, 20, 10);
    for (int i = 0; i < imgs.size(); i++) {
        features[i].img_idx = i;
        cv::Mat gray_img;
        cv::cvtColor(imgs[i], gray_img, cv::COLOR_BGR2GRAY);
        orb_extractor(gray_img, cv::Mat(), features[i].keypoints,
                      features[i].descriptors, vlapp);
    }
    std::vector<cv::detail::MatchesInfo> pairwise_matches;
    MatchImages(features, pairwise_matches);
    cv::Mat pairwiseImg;
    drawMatches(imgs[0], features[0].keypoints, imgs[1], features[1].keypoints,
                pairwise_matches[1].matches, pairwiseImg, cv::Scalar::all(-1),
                cv::Scalar::all(-1), std::vector<char>(),
                cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("matcher", pairwiseImg);
    cv::waitKey(0);
    return 0;
}