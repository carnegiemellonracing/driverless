#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <chrono>
#include <bits/stdc++.h>
#include <thread>
#include <atomic>

#include "rclcpp/rclcpp.hpp" // For propper logging
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#define _USE_MATH_DEFINES
#include <cmath>
#include <unordered_map>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using namespace pcl;

typedef struct radial {
  double angle;
  double radius;
  double z;
} radial_t;

/**
 * Converts (x,y,z) to (radius,ang,z), where ang is in radians
 * @param pt: The (x,y,z) point to convert
 * @return the converted point
 */
inline radial_t point2radial(PointXYZ pt) {
  radial_t rd;
  rd.angle = std::atan2(pt.y, pt.x);
  rd.radius = std::sqrt(pt.x * pt.x + pt.y * pt.y);
  rd.z = pt.z;
  return rd;
}

/**
 * Converts (radius,ang,z) to (x,y,z), where ang is in radians
 * @param rd: The (radius,ang,z) point to convert
 * @return the converted point
 */
inline PointXYZ radial2point(radial_t rd) {
  PointXYZ pt;
  pt.x = rd.radius * cos(rd.angle);
  pt.y = rd.radius * sin(rd.angle);
  pt.z = rd.z;
  return pt;
}

/**
 * Gets the minimum point in a bin
 * @param bin: The bin to search
 * @return the point with the lowest z
 */
inline radial_t min_height(vector<radial_t> bin) {
  int size = bin.size();
  if (size == 0) {
    return {-100, -100, -100};
  }

  radial_t mini = bin[0];
  for (int i = 0; i < size; i++) {
    radial_t rd = bin[i];
    if (rd.z < mini.z) {
      mini = rd;
    }
  }
  return mini;
}

/**
 * Function implementing the GraceAndConrad algorithm
 * @param cloud: The input vector of rectangular points to parse
 * @param alpha: The size of each segment (radians)
 * @param num_bins: The number of bins per segment
 * @param height_threshold: Keep all points this distance above the best fit line
 * @return A point cloud of ground-filtered points
 */
inline PointCloud<PointXYZ> GraceAndConrad(PointCloud<PointXYZ> cloud, double alpha, 
                                    int num_bins, double height_threshold) {

  double upper_height_threshold = 0.2;

  const double angle_min = -0.5 * M_PI;
  const double angle_max = 0.5 * M_PI;
  const double radius_max = 15;
  int num_segs = static_cast<int>((angle_max - angle_min) / alpha);
  vector<vector<vector<radial_t>>> segments(num_segs, vector<vector<radial_t>>(num_bins));
  //&& rd.angle > -4 * (M_PI/9) && rd.angle < 4 * (M_PI/9)
  PointCloud<PointXYZ> output;

  // Parse all points from XYZ to radial,Z and separate into bins
  int csize = cloud.points.size();
  for (int i = 0; i < csize; i++) {
    PointXYZ pt = cloud.points[i];
    radial_t rd = point2radial(pt);

    if (rd.radius < radius_max) {
      int seg_index = static_cast<int>(rd.angle / alpha) + num_segs / 2 - (rd.angle < 0);
      int bin_index = static_cast<int>(rd.radius / (radius_max / num_bins));
      if (seg_index < 0)
      seg_index = 0;
      if (seg_index >= num_segs)
      seg_index = num_segs - 1;
      segments[seg_index][bin_index].push_back(rd);   // This line is doubling the execution time of sector 1
    }
  }

  // Grace and Conrad Algorithm
  for (int seg = 0; seg < num_segs; seg++) {

    // Extract minimum points in each bin
    vector<double> minis_rad = {};
    vector<double> minis_z = {};
    for (int bin = 0; bin < num_bins; bin++) {
      radial_t mini = min_height(segments[seg][bin]);
      if (mini.radius != -100) {
      minis_rad.push_back(mini.radius);
      minis_z.push_back(mini.z);
      }
    }

    // Performing linear regression
    double sum_rad = 0;
    double sum_rad2 = 0;
    double sum_z = 0;
    double sum_radz = 0;
    int n = minis_rad.size();
    for (int i = 0; i < n; i++) {
      double rad = minis_rad[i];
      double z = minis_z[i];
      sum_rad += rad;
      sum_rad2 += rad * rad;
      sum_z += z;
      sum_radz += rad * z;
    }
    
    // Calculating slope and intercept
    double slope = 0;
    double intercept = sum_z;
    if (n > 1) {
      slope = (n * sum_radz - sum_rad * sum_z) / (n * sum_rad2 - sum_rad * sum_rad);
      intercept = (sum_z - slope * sum_rad) / n;
    }

    // Convert all correct points to XYZ and push to output vector
    for (int bin = 0; bin < num_bins; bin++) {
      for (int j = segments[seg][bin].size() - 1; j >= 0; j--) {
        radial_t pt = segments[seg][bin][j];
        double low_cutoff = slope * pt.radius + intercept + height_threshold;
        double high_cutoff = slope * pt.radius + intercept + upper_height_threshold;
        if (pt.z > low_cutoff & pt.z < high_cutoff) {
          output.points.push_back(radial2point(pt));
        }
      }
    }

  }

  return output;
}

// Calculates Euclidean distance between two PointXYZ.
inline double euclideanDistance(const PointXYZ &a, const PointXYZ &b) {
  double dx = a.x - b.x;
  double dy = a.y - b.y;
  double dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Returns indices of points in cloud that are within epsilon of point.
inline vector<int> regionQuery(
  PointCloud<PointXYZ> &cloud,
  const PointXYZ &point,
  double epsilon) {
  vector<int> neighbors;
  neighbors.reserve(cloud.points.size());
  for (size_t i = 0; i < cloud.points.size(); ++i) {
    if (euclideanDistance(point, cloud.points[i]) <= epsilon) {
      neighbors.push_back(static_cast<int>(i));
    }
  }
  return neighbors;
}

// Expands the cluster by checking neighbors and assigning them as needed.
inline void expandCluster(
  PointCloud<PointXYZ> &cloud,
  vector<bool> &visited,
  vector<int> &cluster,
  int point_idx,
  vector<int> &neighbors,
  int cluster_id,
  double epsilon,
  int min_points) {
  cluster[point_idx] = cluster_id;

  size_t i = 0;
  while (i < neighbors.size()) {
    int neighbor_idx = neighbors[i];
    if (!visited[neighbor_idx]) {
      visited[neighbor_idx] = true;
      auto new_neighbors = regionQuery(cloud, cloud.points[neighbor_idx], epsilon);
      if (new_neighbors.size() >= static_cast<size_t>(min_points)) {
        neighbors.insert(neighbors.end(), new_neighbors.begin(), new_neighbors.end());
      }
    }
    if (cluster[neighbor_idx] == -1) {
      cluster[neighbor_idx] = cluster_id;
    }
    ++i;
  }
}

// Computes centroids for the clusters.
inline PointCloud<PointXYZ> computeCentroids(
  PointCloud<PointXYZ> &cloud,
  const unordered_map<int, vector<int>> &clusters) {
  PointCloud<PointXYZ> centroids;
  centroids.points.reserve(clusters.size());

  for (const auto &kv : clusters) {
    const auto &indices = kv.second;
    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
    for (int idx : indices) {
      sum_x += cloud.points[idx].x;
      sum_y += cloud.points[idx].y;
      sum_z += cloud.points[idx].z;
    }
    double size = static_cast<double>(indices.size());
    PointXYZ centroid;
    centroid.x = sum_x / size;
    centroid.y = sum_y / size;
    centroid.z = sum_z / size;
    centroids.points.push_back(centroid);
  }
  return centroids;
}

// DBSCAN that works on a PointCloud<PointXYZ>
inline PointCloud<PointXYZ> DBSCAN(PointCloud<PointXYZ> &cloud, double epsilon, int min_points) {

  vector<bool> visited(cloud.points.size(), false);
  vector<int> cluster(cloud.points.size(), -1);

  int cluster_id = 0;

  for (size_t i = 0; i < cloud.points.size(); ++i) {
    if (visited[i]) {
      continue;
    }
    visited[i] = true;
    auto neighbors = regionQuery(cloud, cloud.points[i], epsilon);

    if (neighbors.size() < static_cast<size_t>(min_points)) {
      cluster[i] = 0;  // Mark as noise
    } else {
      ++cluster_id;
      expandCluster(cloud, visited, cluster, static_cast<int>(i), neighbors, cluster_id, epsilon, min_points);
    }
  }

  // Collect cluster indices.
  unordered_map<int, vector<int>> clusters;
  for (size_t i = 0; i < cluster.size(); ++i) {
    if (cluster[i] > 0) {
      clusters[cluster[i]].push_back(static_cast<int>(i));
    }
  }

  return computeCentroids(cloud, clusters);
}

//Use for secondary filtering to get rid of extraneous clusters outside of cones
inline PointCloud<PointXYZ> DBSCAN2(PointCloud<PointXYZ> &cloud, double epsilon, int min_points) {
  // visited[i] indicates whether the point has been visited.
  // cluster[i] = -1 for unclassified, 0 for noise, >0 for cluster ID.
  vector<bool> visited(cloud.points.size(), false);
  vector<int> cluster(cloud.points.size(), -1);

  int cluster_id = 0;

  for (size_t i = 0; i < cloud.points.size(); ++i) {
    if (visited[i]) {
      continue;
    }
    visited[i] = true;
    auto neighbors = regionQuery(cloud, cloud.points[i], epsilon);

    if (neighbors.size() < static_cast<size_t>(min_points)) {
      cluster[i] = 0;  // Mark as noise
    } else {
      ++cluster_id;
      expandCluster(cloud, visited, cluster, static_cast<int>(i), neighbors, cluster_id, epsilon, min_points);
    }
  }

  return cloud;
}

inline interfaces::msg::ConeArray color_cones_without_camera(const PointCloud<PointXYZ>& cloud) {
    interfaces::msg::ConeArray message = interfaces::msg::ConeArray();
    message.blue_cones = std::vector<geometry_msgs::msg::Point> {};
    message.yellow_cones = std::vector<geometry_msgs::msg::Point> {};
    message.orange_cones = std::vector<geometry_msgs::msg::Point> {};

    std::vector<bool> processed_cones(cloud.points.size(), false);
    size_t processed_count = 0;
    const size_t total_cones = cloud.points.size();

    double min_distance_left = std::numeric_limits<double>::max();
    double min_distance_right = std::numeric_limits<double>::max();
    int initial_left_idx = -1;
    int initial_right_idx = -1;

    for (size_t i = 0; i < cloud.points.size(); i++) {
        const auto& point = cloud.points[i];
        double distance = std::sqrt(std::pow(point.x, 2) + std::pow(point.y, 2));

        if (point.x < 0 && distance < min_distance_left) {
            min_distance_left = distance;
            initial_left_idx = i;
        }
        else if (point.x > 0 && distance < min_distance_right) {
            min_distance_right = distance;
            initial_right_idx = i;
        }
    }

    if (initial_left_idx != -1) {
        geometry_msgs::msg::Point p;
        p.x = cloud.points[initial_left_idx].x;
        p.y = cloud.points[initial_left_idx].y;
        p.z = cloud.points[initial_left_idx].z;
        message.blue_cones.push_back(p);
        processed_cones[initial_left_idx] = true;
        processed_count++;
    }
    if (initial_right_idx != -1) {
        geometry_msgs::msg::Point p;
        p.x = cloud.points[initial_right_idx].x;
        p.y = cloud.points[initial_right_idx].y;
        p.z = cloud.points[initial_right_idx].z;
        message.yellow_cones.push_back(p);
        processed_cones[initial_right_idx] = true;
        processed_count++;
    }

    PointXYZ current_left_cone = cloud.points[initial_left_idx];
    PointXYZ current_right_cone = cloud.points[initial_right_idx];

    while (processed_count < total_cones) {
        min_distance_left = std::numeric_limits<double>::max();
        min_distance_right = std::numeric_limits<double>::max();
        int next_left_idx = -1;
        int next_right_idx = -1;

        for (size_t i = 0; i < cloud.points.size(); i++) {
            if (!processed_cones[i]) {
                const auto& point = cloud.points[i];
                
                double distance_to_left = std::sqrt(
                    pow(point.x - current_left_cone.x, 2) + 
                    pow(point.y - current_left_cone.y, 2)
                );
                if (distance_to_left < min_distance_left) {
                    min_distance_left = distance_to_left;
                    next_left_idx = i;
                }

                double distance_to_right = std::sqrt(
                    pow(point.x - current_right_cone.x, 2) + 
                    pow(point.y - current_right_cone.y, 2)
                );
                if (distance_to_right < min_distance_right) {
                    min_distance_right = distance_to_right;
                    next_right_idx = i;
                }
            }
        }

        if (next_left_idx == -1 && next_right_idx == -1) break;

        if (next_left_idx != -1) {
            current_left_cone = cloud.points[next_left_idx];
            geometry_msgs::msg::Point p;
            p.x = current_left_cone.x;
            p.y = current_left_cone.y;
            p.z = current_left_cone.z;
            message.blue_cones.push_back(p);
            processed_cones[next_left_idx] = true;
            processed_count++;
        }

        if (next_right_idx != -1) {
            current_right_cone = cloud.points[next_right_idx];
            geometry_msgs::msg::Point p;
            p.x = current_right_cone.x;
            p.y = current_right_cone.y;
            p.z = current_right_cone.z;
            message.yellow_cones.push_back(p);
            processed_cones[next_right_idx] = true;
            processed_count++;
        }
    }

    return message;
}


inline interfaces::msg::ConeArray run_pipeline_dark(PointCloud<PointXYZ> &cloud, double alpha, 
                                          int num_bins, double height_threshold, 
                                          double epsilon, int min_points, 
                                          double epsilon2, int min_points2,
                                          const rclcpp::Logger &logger) {

    // Start overall timer
    auto start_pipeline = std::chrono::high_resolution_clock::now();
    
    // Print the entry sizetime: 36.776 of the cloud
    printf("Entry Size: %zu\n", cloud.size());                               

    // Time GraceAndConrad step
    auto start_GNC = std::chrono::high_resolution_clock::now();
    PointCloud<PointXYZ> GNC_cloud = GraceAndConrad(cloud, alpha, num_bins, height_threshold);
    auto end_GNC = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_GNC = end_GNC - start_GNC;
    RCLCPP_INFO(logger, "GraceAndConrad time: %fms", duration_GNC.count());

    // Time DBSCAN step
    auto start_DBSCAN = std::chrono::high_resolution_clock::now();
    PointCloud<PointXYZ> clustered_cloud = DBSCAN(GNC_cloud, epsilon, min_points);
    auto end_DBSCAN = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_DBSCAN = end_DBSCAN - start_DBSCAN;
    RCLCPP_INFO(logger, "DBSCAN time: %fms", duration_DBSCAN.count());

    // Time DBSCAN2 step
    auto start_DBSCAN2 = std::chrono::high_resolution_clock::now();
    PointCloud<PointXYZ> filtered_cloud = DBSCAN2(clustered_cloud, epsilon2, min_points2);
    auto end_DBSCAN2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_DBSCAN2 = end_DBSCAN2 - start_DBSCAN2;
    RCLCPP_INFO(logger, "DBSCAN2 time: %fms", duration_DBSCAN2.count());

    // Time the overall pipeline
    auto end_pipeline = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_pipeline = end_pipeline - start_pipeline;
    RCLCPP_INFO(logger, "Total pipeline time: %fms", duration_pipeline.count());

    for (int i = 0; i < filtered_cloud.size(); i++) {
      double original_x = filtered_cloud.points[i].x;
      double original_y = filtered_cloud.points[i].y;
      double original_z = filtered_cloud.points[i].z;
      
      filtered_cloud.points[i].x = -original_y;
      filtered_cloud.points[i].y = original_x;
      filtered_cloud.points[i].z = original_z;
    }

    interfaces::msg::ConeArray message = color_cones_without_camera(filtered_cloud);

    return message;

  }

  inline PointCloud<PointXYZ> run_pipeline(PointCloud<PointXYZ> &cloud, double alpha,
                                           int num_bins, double height_threshold,
                                           double epsilon, int min_points,
                                           double epsilon2, int min_points2,
                                           const rclcpp::Logger &logger)
  {

    // Start overall timer
    auto start_pipeline = std::chrono::high_resolution_clock::now();

    // Time GraceAndConrad step
    auto start_GNC = std::chrono::high_resolution_clock::now();
    PointCloud<PointXYZ> GNC_cloud = GraceAndConrad(cloud, alpha, num_bins, height_threshold);
    auto end_GNC = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_GNC = end_GNC - start_GNC;
    RCLCPP_INFO(logger, "GraceAndConrad time: %fms", duration_GNC.count());

    // Time DBSCAN step
    auto start_DBSCAN = std::chrono::high_resolution_clock::now();
    PointCloud<PointXYZ> clustered_cloud = DBSCAN(GNC_cloud, epsilon, min_points);
    auto end_DBSCAN = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_DBSCAN = end_DBSCAN - start_DBSCAN;
    RCLCPP_INFO(logger, "DBSCAN time: %fms", duration_DBSCAN.count());

    // Time DBSCAN2 step
    auto start_DBSCAN2 = std::chrono::high_resolution_clock::now();
    PointCloud<PointXYZ> filtered_cloud = DBSCAN2(clustered_cloud, epsilon2, min_points2);
    auto end_DBSCAN2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_DBSCAN2 = end_DBSCAN2 - start_DBSCAN2;
    RCLCPP_INFO(logger, "DBSCAN2 time: %fms", duration_DBSCAN2.count());

    // Time the overall pipeline
    auto end_pipeline = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_pipeline = end_pipeline - start_pipeline;
    RCLCPP_INFO(logger, "Total pipeline time: %fms", duration_pipeline.count());

    return filtered_cloud;
  }