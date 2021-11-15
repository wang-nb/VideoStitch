//
// Created by Administrator on 2021-11-15.
//

#include <fstream>
#include <string>

#include "video_stitcher.h"

int MyVideoStitcher::StitchFrameCPU(vector<Mat> &src, Mat &dst)
{
    int flag = 0;
    if (src.size() != video_num_) {
        flag = -1;
    } else {
        final_warped_images_.resize(video_num_);
        for (int j = 0; j < video_num_; j++)
            final_warped_images_[j].create(sizes_[j], src[src_indices_[j]].type());

        int num_images = (int)src_indices_.size();

        int dst_width  = dst_roi_.width;
        int dst_height = dst_roi_.height;
        if (dst.empty())
            dst.create(dst_roi_.size(), CV_8UC3);
        uchar *dst_ptr_00 = dst.ptr<uchar>(0);
        memset(dst_ptr_00, 0, dst_width * dst_height * 3);

        for (int img_idx = 0; img_idx < num_images; ++img_idx) {
            // Warp the current image
            remap(src[src_indices_[img_idx]], final_warped_images_[img_idx],
                  xmaps_[img_idx], ymaps_[img_idx], INTER_LINEAR);
            int dx       = corners_[img_idx].x - dst_roi_.x;
            int dy       = corners_[img_idx].y - dst_roi_.y;
            int img_rows = sizes_[img_idx].height;
            int img_cols = sizes_[img_idx].width;

            parallel_num_         = 4;
            int rows_per_parallel = img_rows / parallel_num_;
#pragma omp parallel for
            for (int parallel_idx = 0; parallel_idx < parallel_num_; parallel_idx++) {
                int row_start = parallel_idx * rows_per_parallel;
                int row_end   = row_start + rows_per_parallel;
                if (parallel_idx == parallel_num_ - 1)
                    row_end = img_rows;

                uchar *dst_ptr;
                uchar *warped_img_ptr   = final_warped_images_[img_idx].ptr<uchar>(row_start);
                float *total_weight_ptr = total_weight_maps_[img_idx].ptr<float>(row_start);
                for (int y = row_start; y < row_end; y++) {
                    dst_ptr = dst_ptr_00 + ((dy + y) * dst_width + dx) * 3;
                    for (int x = 0; x < img_cols; x++) {
                        /* 曝光补偿和融合加权平均 */
                        (*dst_ptr) += (uchar)(cvRound((*warped_img_ptr) * (*total_weight_ptr)));
                        warped_img_ptr++;
                        dst_ptr++;

                        (*dst_ptr) += (uchar)(cvRound((*warped_img_ptr) * (*total_weight_ptr)));
                        warped_img_ptr++;
                        dst_ptr++;

                        (*dst_ptr) += (uchar)(cvRound((*warped_img_ptr) * (*total_weight_ptr)));
                        warped_img_ptr++;
                        dst_ptr++;

                        total_weight_ptr++;
                    }
                }
            }
        }
    }

    return flag;
}

int MyVideoStitcher::stitchImage(vector<Mat> &src, Mat &pano)
{
    StitchFrameCPU(src, pano);
    return 0;
}

int MyVideoStitcher::getDstSize(cv::Size &dst_size)
{
    dst_size = dst_roi_.size();
    return 0;
}

int MyVideoStitcher::init(const std::string &config_path)
{
    std::string bin_path = config_path + "/config.bin";
    std::shared_ptr<FILE> fid(fopen(bin_path.c_str(), "r"), fclose);
    fscanf(fid.get(), "%d", &video_num_);
    src_indices_.resize(video_num_);
    for (int i = 0; i < video_num_; i++) {
        int indices;
        fscanf(fid.get(), "%d ", &indices);
        src_indices_[i] = indices;
    }
    fscanf(fid.get(), "%d %d %d %d", &dst_roi_.x, &dst_roi_.y,
           &dst_roi_.width, &dst_roi_.height);
    corners_.resize(video_num_);
    for (int i = 0; i < video_num_; i++) {
        int pt_x, pt_y;
        fscanf(fid.get(), "%d %d ", &pt_x, &pt_y);
        corners_[i] = cv::Point(pt_x, pt_y);
    }
    sizes_.resize(video_num_);
    for (int i = 0; i < video_num_; i++) {
        int w, h;
        fscanf(fid.get(), "%d %d ", &w, &h);
        sizes_[i].width  = w;
        sizes_[i].height = h;
    }
    cv::FileStorage fs(config_path + "/mapx.xml", cv::FileStorage::READ);
    xmaps_.resize(video_num_);
    ymaps_.resize(video_num_);
    total_weight_maps_.resize(video_num_);
    for (int i = 0; i < video_num_; i++) {
        fs["xmap" + std::to_string(i)] >> xmaps_[i];
    }
    fs.release();
    fs.open(config_path + "/mapy.xml", cv::FileStorage::READ);
    for (int i = 0; i < video_num_; i++) {
        fs["ymap" + std::to_string(i)] >> ymaps_[i];
    }
    fs.release();
    fs.open(config_path + "/blend_weight.xml", cv::FileStorage::READ);
    for (int i = 0; i < video_num_; i++) {
        fs["blend_weight" + std::to_string(i)] >> total_weight_maps_[i];
    }
    fs.release();
    return 0;
}
