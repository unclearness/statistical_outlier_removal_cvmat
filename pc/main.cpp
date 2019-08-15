/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include "statistical_outlier_removal.h"

#include "opencv2/imgcodecs.hpp"

cv::Mat1b Depth2Gray(const cv::Mat1f& depth, float dmin = 0.5,
                     float dmax = 5.0) {
  cv::Mat1b gray(depth.size());

  float dfactor = 1.0f / (dmax - dmin);

  depth.forEach([&](const float& org_d, const int* pos) -> void {
    // int index = pos[0] * depth.cols + pos[1];

    float d = std::min(std::max(org_d, dmin), dmax);

    gray.at<unsigned char>(pos[0], pos[1]) =
        static_cast<unsigned char>((d - dmin) * dfactor * 255);
  });
  return gray;
}

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;

  // RGB-D SLAM Dataset and Benchmark
  // https://vision.in.tum.de/data/datasets/rgbd-dataset/download
  // rgbd_dataset_freiburg3_long_office_household
  std::string org_depth_path = "../data/1341847980.723020.png";

  cv::Mat1w org_depth =
      cv::imread(org_depth_path, cv::ImreadModes::IMREAD_UNCHANGED);

  cv::Mat1f depth(org_depth.size());
  // divide by 5000 to recover meter scale
  // https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
  org_depth.convertTo(depth, CV_32FC1, 1.0 / 5000.0);

  std::string org_vis_depth_path = "../data/org.png";
  cv::imwrite(org_vis_depth_path, Depth2Gray(depth));

  std::string vis_depth_path = "../data/outlier_removed.png";
  std::string mask_path = "../data/outlier_mask.png";
  cv::Mat1f processed_depth = cv::Mat1f::zeros(org_depth.size());
  double mean, stddev;
  cv::Mat1b outlier_mask;

  // intrinsics of Freiburg 3 RGB
  StatisticalOutlierRemoval(depth, 535.4f, 539.2f, 320.1f, 247.6f,
                            &outlier_mask, &mean, &stddev);

  cv::Mat1b inlier_mask(outlier_mask.size());
  cv::bitwise_not(outlier_mask, inlier_mask);
  depth.copyTo(processed_depth, inlier_mask);

  cv::imwrite(mask_path, outlier_mask);
  cv::imwrite(vis_depth_path, Depth2Gray(processed_depth));

  return 0;
}
