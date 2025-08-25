#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

#include "rclcpp/rclcpp.hpp" // For propper logging
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using namespace pcl;
constexpr double PI = std::acos(-1.0);

/**
 * @brief Radial struct for points
 */
typedef struct radial {
  double angle;
  double radius;
  double z;
} radial_t;

/**
 * @brief Converts (x,y,z) to (radius,ang,z), where ang is in radians
 * @param pt: The (x,y,z) point to convert
 * @return the converted point
 */
inline radial_t point2radial(PointXYZ &pt) {
  radial_t rd;
  rd.angle = std::atan2(pt.y, pt.x);
  rd.radius = std::sqrt(pt.x * pt.x + pt.y * pt.y);
  rd.z = pt.z;
  return rd;
}

/**
 * @brief Converts (radius,ang,z) to (x,y,z), where ang is in radians
 * @param rd: The (radius,ang,z) point to convert
 * @return the converted point
 */
inline PointXYZ radial2point(radial_t &rd) {
  PointXYZ pt;
  pt.x = rd.radius * std::cos(rd.angle);
  pt.y = rd.radius * std::sin(rd.angle);
  pt.z = rd.z;
  return pt;
}

/**
 * @brief Gets the minimum point in a bin
 * @param bin: The bin to search
 * @return the point with the lowest z
 */
inline radial_t min_height(const std::vector<radial_t> &bin) {
  if (bin.empty()) {
    return {-100, -100, -100};
  }
  radial_t min = bin[0];
  for (const auto& rd : bin) {
    if (rd.z < min.z) {
      min = rd;
    }
  }
  return min;
}

/**
 * @brief Function implementing the GraceAndConrad algorithm
 * @param cloud: The input vector of rectangular points to parse
 * @param alpha: The (angle) size of each segment (radians)
 * @param num_bins: The number of (concentric) bins per segment
 * @param lower_height_threshold: Keep all points this distance above the best fit line
 * @return A point cloud of ground-filtered points
 */
PointCloud<PointXYZ> GraceAndConrad(PointCloud<PointXYZ> &cloud, double alpha, 
                                           int num_bins, double lower_height_threshold) {

  const double fov_angle_min = -1. * PI;  // -0.5 * PI;
  const double fov_angle_max = PI;  // 0.5 * PI;
  const double radius_max = 20; // outer bound of bins
  double upper_height_threshold = 0.2;

  // multiplication better than division
  const double inv_alpha = 1.0 / alpha;
  const double inv_bin   = static_cast<double>(num_bins) / radius_max;
  
  // num segs = ceiling[ total fov (rad) / alpha (rad/seg) ]
  int num_segs = static_cast<int>(std::ceil((fov_angle_max - fov_angle_min) * inv_alpha));

  std::vector<std::vector<std::vector<radial_t>>> segments(num_segs, std::vector<std::vector<radial_t>>(num_bins));
  PointCloud<PointXYZ> output;

//   idea to preallocate memory in hopes of increasing speed. Untested. Stanley Yin
//   const int csize = static_cast<int>(cloud.points.size());
//   output.points.reserve(csize);
//   const double bins_total = static_cast<double>(num_segs) * num_bins;
//   const int est_per_bin = std::max(1, static_cast<int>(std::ceil(csize / bins_total)));
//   for (int s = 0; s < num_segs; ++s) {
//     for (int b = 0; b < num_bins; ++b) {
//       segments[s][b].reserve(est_per_bin);
//     }
//   } 

  // Parse all points from XYZ to radial,Z and separate into bins
  for (size_t i = 0; i < cloud.points.size(); i++) {
    PointXYZ pt = cloud.points[i];
    const radial_t rd = point2radial(pt);

    if (pt.y > 1.5 || pt.y < -1.75) continue; // MOVIA filter objects outside of lane bc breezeway is very noisy
    // if (pt.x > 2. || pt.x < -1.75) continue; // HESAI filter objects outside of lane bc breezeway is very noisy

    if (rd.radius < radius_max && rd.angle >= fov_angle_min && rd.angle <= fov_angle_max) {
      int seg_index = static_cast<int>(std::floor((rd.angle - fov_angle_min) * inv_alpha));
      seg_index = std::clamp(seg_index, 0, num_segs - 1);

      int bin_index = static_cast<int>(std::floor(rd.radius * inv_bin));
      bin_index = std::clamp(bin_index, 0, num_bins - 1);

    //   segments[seg_index][bin_index].emplace_back(rd); // used with the preallocated memory. Stanley Yin
      segments[seg_index][bin_index].push_back(rd);
    }
  }

  // Grace and Conrad Algorithm
  for (int seg = 0; seg < num_segs; seg++) {

    // Extract minimum points in each bin
    std::vector<double> minis_rad = {};
    std::vector<double> minis_z = {};
    for (int bin = 0; bin < num_bins; bin++) {
      if (!segments[seg][bin].empty()) {
        const radial_t mini = min_height(segments[seg][bin]);
        minis_rad.push_back(mini.radius);
        minis_z.push_back(mini.z);
      }
      // Removed {-100, -100, -100} resulting from empty bin as that would skew the GNC plane. AL
    }

    // Performing linear regression
    double sum_rad = 0.0;
    double sum_rad2 = 0.0;
    double sum_z = 0.0;
    double sum_radz = 0.0;
    int n = static_cast<int>(minis_rad.size());
    for (int i = 0; i < n; i++) {
      const double rad = minis_rad[i];
      const double z = minis_z[i];
      sum_rad += rad;
      sum_rad2 += rad * rad;
      sum_z += z;
      sum_radz += rad * z;
    }
    
    // Calculating slope and intercept
    double slope = 0.0;
    double intercept = sum_z;
    if (n > 1) {
      slope = (n * sum_radz - sum_rad * sum_z) / (n * sum_rad2 - sum_rad * sum_rad);
      intercept = (sum_z - slope * sum_rad) / n;
    }


    // Untested change to slope intercept calculations. Stanley Yin
    // double intercept = (n > 0 ? sum_z / n : 0.0);
    // if (n > 3) {
    //   const double denom = n * sum_rad2 - sum_rad * sum_rad;
    //   if (std::abs(denom) > 1e-9) {
    //     slope = (n * sum_radz - sum_rad * sum_z) / denom;
    //     intercept = (sum_z - slope * sum_rad) / n;
    //   } else {
    //     slope = 0.0;
    //     intercept = (n > 0 ? sum_z / n : 0.0);
    //   }
    // }

    // Convert all correct points to XYZ and push to output vector
    for (int bin = 0; bin < num_bins; bin++) {
      for (radial_t pt_r : segments[seg][bin]) {
        const double low_cutoff = slope * pt_r.radius + intercept + lower_height_threshold;
        const double high_cutoff = slope * pt_r.radius + intercept + upper_height_threshold;
        if (pt_r.z > low_cutoff && pt_r.z < high_cutoff) {
        //   output.points.emplace_back(radial2point(pt_r)); // used with the preallocated memory. Stanley Yin
          output.points.push_back(radial2point(pt_r));
        }
      }
    }
  }

  return output;
}