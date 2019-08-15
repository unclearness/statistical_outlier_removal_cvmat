/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#pragma once

#include <limits>
#include <vector>

#include "opencv2/core.hpp"

template <typename T>
bool StatisticalOutlierRemoval(const cv::Mat_<T>& depth, float fx, float fy,
                               float cx, float cy, cv::Mat1b* outlier_mask,
                               double* mean, double* stddev,
                               int nn_kernel_size = 5,
                               int valid_neighbor_num_th = -1,
                               double std_mul = 1.0) {
  if (nn_kernel_size < 0 || std_mul < 0) {
    return false;
  }

  cv::Mat3f point_cloud = cv::Mat3f::zeros(depth.size());
  constexpr float e = std::numeric_limits<float>::epsilon();
  bool is_perspective = fx > 0 && fy > 0 && cx > 0 && cy > 0;

  std::function<void(cv::Vec3f*, int, int, float, float, float, float, float)>
      convert = [&](cv::Vec3f* p, int x, int y, float z, float fx, float fy,
                    float cx, float cy) {
        (void)x;
        (void)y;
        (void)fx;
        (void)fy;
        (void)cx;
        (void)cy;
        // keep x and y zero to evalate depth only
        // (*p)[0] = x;
        // (*p)[1] = y;
        (*p)[2] = z;
      };

  if (is_perspective) {
    convert = [&](cv::Vec3f* p, int x, int y, float z, float fx, float fy,
                  float cx, float cy) {
      (*p)[0] = (x - cx) * z / fx;
      (*p)[1] = (y - cy) * z / fy;
      (*p)[2] = z;
    };
  }

  cv::Vec3f* pc_data = reinterpret_cast<cv::Vec3f*>(point_cloud.data);

  depth.forEach([&](const T& p, const int* pos) -> void {
    int index = pos[0] * depth.cols + pos[1];

    if (p < e) {
      return;
    }

    convert(&pc_data[index], pos[1], pos[0], static_cast<float>(p), fx, fy, cx,
            cy);
  });

  return StatisticalOutlierRemoval(point_cloud, outlier_mask, mean, stddev,
                                   nn_kernel_size, valid_neighbor_num_th,
                                   std_mul);
}

// http://docs.pointclouds.org/trunk/statistical__outlier__removal_8hpp_source.html
// modify for organized point cloud
inline bool StatisticalOutlierRemoval(const cv::Mat3f& point_cloud,
                                      cv::Mat1b* outlier_mask, double* mean,
                                      double* stddev, int nn_kernel_size = 5,
                                      int valid_neighbor_num_th = -1,
                                      double std_mul = 1.0) {
  if (nn_kernel_size < 0 || std_mul < 0) {
    return false;
  }

  *outlier_mask = cv::Mat1b::zeros(point_cloud.size());
  // The arrays to be used
  std::vector<float> distances(point_cloud.cols * point_cloud.rows, -1.0);
  const int hk = nn_kernel_size / 2;
  if (valid_neighbor_num_th < 0) {
    // 25% of inside kernel
    valid_neighbor_num_th = nn_kernel_size * nn_kernel_size / 4;
  }

  constexpr float e = std::numeric_limits<float>::epsilon();

  // First pass: Compute the mean distances for all points with respect to their
  // k nearest neighbors
  int valid_distances = 0;
  for (int j = hk; j < point_cloud.rows - hk; j++) {
    const cv::Vec3f* pc_row = point_cloud.ptr<cv::Vec3f>(j);
    unsigned char* outlier_mask_row = outlier_mask->ptr<unsigned char>(j);
    for (int i = hk; i < point_cloud.cols - hk; i++) {
      if (pc_row[i][2] < e) {
        // no outlier mask on invalid depth pixels
        continue;
      }

      // Calculate the mean distance to its neighbors
      double dist_sum = 0.0;
      int valid_distances_in = 0;
      for (int jj = j - hk; jj < j + hk; jj++) {
        const cv::Vec3f* pc_row_in = point_cloud.ptr<cv::Vec3f>(jj);
        // unsigned char* outlier_mask_row_in =
        //    outlier_mask->ptr<unsigned char>(jj);
        for (int ii = i - hk; ii < i + hk; ii++) {
          if (ii == 0 && jj == 0) {
            continue;
          }
          if (pc_row_in[ii][2] < e) {
            continue;
          }

          dist_sum += cv::norm(pc_row[i] - pc_row_in[ii]);
          valid_distances_in++;
        }
      }

      // outlier condition 1: #valid pixels is smaller than threshold
      if (valid_distances_in < valid_neighbor_num_th) {
        outlier_mask_row[i] = 255;
        continue;
      }

      distances[j * point_cloud.cols + i] =
          static_cast<float>(dist_sum / valid_distances_in);
      valid_distances++;
    }
  }

  // Estimate the mean and the standard deviation of the distance vector
  double sum = 0, sq_sum = 0;
  for (const float& distance : distances) {
    sum += distance;
    sq_sum += distance * distance;
  }
  *mean = sum / static_cast<double>(valid_distances);
  double variance =
      (sq_sum - sum * sum / static_cast<double>(valid_distances)) /
      (static_cast<double>(valid_distances) - 1);
  *stddev = sqrt(variance);

  double distance_threshold = (*mean) + std_mul * (*stddev);

  // Second pass: Classify the points on the computed distance threshold
  for (int j = hk; j < point_cloud.rows - hk; j++) {
    const cv::Vec3f* pc_row = point_cloud.ptr<cv::Vec3f>(j);
    unsigned char* outlier_mask_row = outlier_mask->ptr<unsigned char>(j);
    for (int i = hk; i < point_cloud.cols - hk; i++) {
      if (pc_row[i][2] < e || outlier_mask_row[i] == 255) {
        continue;
      }

      // outlier condition 2: average distance is higher than threshold
      if (distance_threshold < distances[j * point_cloud.cols + i]) {
        outlier_mask_row[i] = 255;
        continue;
      }
    }
  }

  return true;
}
