//
// Created by bing on 2021/11/17.
//
#include "opencv2/opencv.hpp"
#include "logging.hpp"

int main()
{
    std::string root_path = "data/video1/";
    std::string pic_pattern = "/*.png";
    std::string feature_type = "sift";//"orb"

    std::vector<std::string> img_names;
    cv::glob(root_path + pic_pattern, img_names);

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
    std::vector<cv::detail::ImageFeatures> features(img_names.size());
    for(size_t i = 0; i < img_names.size(); i++){
        cv::Mat img = cv::imread(img_names[i]);
        if(img.empty()){
            LOG(INFO) << "failed to load image : " << img_names[i];
            return -1;
        }
        finder->detectAndCompute(img, cv::Mat(), features[i].keypoints,
                                 features[i].descriptors, false);
        features[i].img_idx = i;
        cv::Mat feature_img;
        cv::drawKeypoints(img, features[i].keypoints, feature_img);
        cv::imwrite(root_path + feature_type + "_" + std::to_string(i) + ".jpg", feature_img);
        cv::imshow(feature_type, feature_img);
        cv::waitKey(0);
    }

    return 0;
}