#include "opencv2/opencv.hpp"
#include <vector>

#define MAX_OPTIMAL_POINT_NUM 10
#define DISPLAY_DEBUG 1

//计算原始图像点位在经过矩阵变换后在目标图像上对应位置
cv::Point2f getTransformPoint(const cv::Point2f originalPoint, const cv::Mat &transformMaxtri)
{
    cv::Mat originelP, targetP;
    originelP = (cv::Mat_<double>(3, 1) << originalPoint.x, originalPoint.y, 1.0);
    targetP   = transformMaxtri * originelP;
    float x   = targetP.at<double>(0, 0) / targetP.at<double>(2, 0);
    float y   = targetP.at<double>(1, 0) / targetP.at<double>(2, 0);
    return cv::Point2f(x, y);
}

void featureExtract(const std::vector<cv::Mat> &imgs,
                    std::vector<std::vector<cv::KeyPoint>> &keyPoints,
                    std::vector<cv::Mat> &imageDescs)
{
    keyPoints.clear();
    imageDescs.clear();
    //提取特征点
    int minHessian       = 800;
    cv::Ptr<cv::ORB> orbDetector = cv::ORB::create(minHessian);
    for (int i = 0; i < imgs.size(); ++i) {
        std::vector<cv::KeyPoint> keyPoint;
        //灰度图转换
        cv::Mat image;
        cvtColor(imgs[i], image, cv::COLOR_BGR2GRAY);
        orbDetector->detect(image, keyPoint);
        keyPoints.push_back(keyPoint);
        cv::Mat imageDesc1;
        orbDetector->compute(image, keyPoint, imageDesc1);
        /*需要将imageDesc转成浮点型，不然会出错Unsupported format or combination of formats in buildIndex using FLANN algorithm*/
        imageDesc1.convertTo(imageDesc1, CV_32F);
        imageDescs.push_back(imageDesc1.clone());
    }
}

void featureMatching(const std::vector<cv::Mat> &imgs,
                     const std::vector<std::vector<cv::KeyPoint>> &keyPoints,
                     const std::vector<cv::Mat> &imageDescs,
                     std::vector<std::vector<cv::Point2f>> &optimalMatchePoint)
{
    optimalMatchePoint.clear();
    //获得匹配特征点，并提取最优配对,此处假设是顺序输入，测试使用假设是两张图
    cv::FlannBasedMatcher matcher;
    std::vector<cv::DMatch> matchePoints;
    matcher.match(imageDescs[0], imageDescs[1], matchePoints, cv::Mat());

    cv::Mat matcherImg;
    //保存匹配点
    drawMatches(imgs[0], keyPoints[0], imgs[1],
                keyPoints[1], matchePoints, matcherImg);
    cv::imshow("data/matcher.jpg", matcherImg);

    sort(matchePoints.begin(), matchePoints.end());//特征点排序
    //获取排在前N个的最优匹配特征点
    std::vector<cv::Point2f> imagePoints1, imagePoints2;
    for (int i = 0; i < MAX_OPTIMAL_POINT_NUM; i++) {
        imagePoints1.push_back(keyPoints[0][matchePoints[i].queryIdx].pt);
        imagePoints2.push_back(keyPoints[1][matchePoints[i].trainIdx].pt);
    }
    optimalMatchePoint.push_back(std::vector<cv::Point2f>{
            imagePoints1[0], imagePoints1[3], imagePoints1[6]});
    optimalMatchePoint.push_back(std::vector<cv::Point2f>{
            imagePoints2[0], imagePoints2[3], imagePoints2[6]});
    drawMatches(imgs[0], keyPoints[0], imgs[1], keyPoints[1],
                std::vector<cv::DMatch>(matchePoints.begin(), matchePoints.begin() + MAX_OPTIMAL_POINT_NUM), matcherImg);

    cv::imshow("data/matcher_sort.jpg", matcherImg);
}

void blendImg(cv::Mat &scale, cv::Mat &pano)
{
    for (int irow = 0; irow < pano.rows; ++irow) {
        uchar *img_row_ptr   = pano.ptr<uchar>(irow);
        float *scale_row_ptr = scale.ptr<float>(irow);
        for (int icol = 0; icol < pano.cols; ++icol) {
            for (int ich = 0; ich < 3; ++ich) {
                img_row_ptr[icol * 3 + ich] *= scale_row_ptr[icol];
            }
        }
    }
}

void getAffineMat(std::vector<std::vector<cv::Point2f>>& optimalMatchePoint,
                  int left_cols, std::vector<cv::Mat>& Hs)
{
    std::vector<cv::Point2f> newMatchingPt;
    for (int i = 0; i < optimalMatchePoint[1].size(); i++) {
        cv::Point2f pt = optimalMatchePoint[1][i];
        pt.x += left_cols;
        newMatchingPt.push_back(pt);
    }
    cv::Mat homo1 = getAffineTransform(optimalMatchePoint[0], newMatchingPt);
    cv::Mat homo2 = getAffineTransform(optimalMatchePoint[1], newMatchingPt);

    Hs.push_back(homo1);
    Hs.push_back(homo2);
}


void getPano1(std::vector<cv::Mat> &imgs, const std::vector<cv::Mat> &H, cv::Mat &pano)
{
    //以右边图像为参考，将left的图像经过仿射变换变到与右边图像重合
    //默认的全景图画布尺寸为：width=left.width + right.width, height = std::max(left.height, right.height)
    int pano_width  = imgs[0].cols + imgs[1].cols;
    int pano_height = std::max(imgs[0].rows, imgs[1].rows);
    pano            = cv::Mat::zeros(cv::Size(pano_width, pano_height), CV_8UC3);
    cv::Mat img_trans0, img_trans1;
    img_trans0 = cv::Mat::zeros(pano.size(), CV_8UC3);
    img_trans1 = cv::Mat::zeros(pano.size(), CV_8UC3);
    cv::warpAffine(imgs[0], img_trans0, H[0], pano.size(), cv::INTER_NEAREST);
    cv::warpAffine(imgs[1], img_trans1, H[1], pano.size(), cv::INTER_NEAREST);

    //取均值进行融合
    cv::Mat overlap_roi;
    cv::Mat gray_img1, gray_img2;
    cv::cvtColor(img_trans0, gray_img1, cv::COLOR_BGR2GRAY);
    cv::threshold(gray_img1, gray_img1, 1, 1, CV_8UC1);
    gray_img1 *= 255;
    cv::cvtColor(img_trans1, gray_img2, cv::COLOR_BGR2GRAY);
    cv::threshold(gray_img2, gray_img2, 1, 1, CV_8UC1);
    gray_img2 *= 255;
    cv::bitwise_and(gray_img1, gray_img2, overlap_roi);

    cv::Mat scale = cv::Mat::ones(pano.size(), CV_32FC1);
    cv::imshow("mask", overlap_roi);
    scale.setTo(0.5f, overlap_roi);
    blendImg(scale, img_trans0);
    blendImg(scale, img_trans1);

    pano = img_trans0 + img_trans1;
    cv::imshow("pano", pano);
    cv::waitKey(0);
}


void getPano2(std::vector<cv::Mat> &imgs, const std::vector<cv::Mat> &H, cv::Point2f &optimalPt, cv::Mat &pano)
{
    //以右边图像为参考，将left的图像经过仿射变换变到与右边图像重合,取最强响应特征点作为两幅图像融合的中心
    //默认的全景图画布尺寸为：width=left.width + right.width, height = std::max(left.height, right.height)
    int pano_width  = imgs[0].cols + imgs[1].cols;
    int pano_height = std::max(imgs[0].rows, imgs[1].rows);
    pano            = cv::Mat::zeros(cv::Size(pano_width, pano_height), CV_8UC3);
    cv::Mat img_trans0, img_trans1;
    img_trans0 = cv::Mat::zeros(pano.size(), CV_8UC3);
    img_trans1 = cv::Mat::zeros(pano.size(), CV_8UC3);
    cv::rectangle(imgs[0], cv::Rect(0, 0, imgs[0].cols, imgs[0].rows), cv::Scalar(0,0,255), 2);
    cv::warpAffine(imgs[0], img_trans0, H[0], pano.size());
    cv::warpAffine(imgs[1], img_trans1, H[1], pano.size());

    cv::Mat trans_pt = (cv::Mat_<double>(3, 1) << optimalPt.x, optimalPt.y, 1.0f);
    trans_pt = H[0]*trans_pt;

    cv::Rect left_roi  = cv::Rect(0, 0, trans_pt.at<double>(0, 0), pano_height);
    cv::Rect right_roi = cv::Rect(trans_pt.at<double>(0, 0), 0,
            pano_width - trans_pt.at<double>(0, 0) + 1, pano_height);

    cv::rectangle(img_trans1, right_roi, cv::Scalar(255,0,255), 2);
    img_trans0(left_roi).copyTo(pano(left_roi));
    img_trans1(right_roi).copyTo(pano(right_roi));
    cv::imshow("pano", pano);
    cv::imwrite("pano.jpg", pano);
    cv::waitKey(0);
}

int main(int argc, char *argv[])
{
    cv::Mat image01 = cv::imread("data/img/medium11.jpg");
    cv::resize(image01, image01, cv::Size(image01.cols, image01.rows + 1));
    cv::Mat image02 = cv::imread("data/img/medium12.jpg");
    cv::resize(image02, image02, cv::Size(image02.cols, image02.rows + 1));
    std::vector<cv::Mat> imgs = {image01, image02};
    std::vector<std::vector<cv::KeyPoint>> keyPoints;
    std::vector<std::vector<cv::Point2f>> optimalMatchePoint;
    std::vector<cv::Mat> imageDescs;
    featureExtract(imgs, keyPoints, imageDescs);
    featureMatching(imgs, keyPoints, imageDescs, optimalMatchePoint);

    std::vector<cv::Mat> Hs;
    getAffineMat(optimalMatchePoint, imgs[0].cols, Hs);
    cv::Mat pano;
    getPano2(imgs, Hs, optimalMatchePoint[0][0], pano);
    return 0;
}
