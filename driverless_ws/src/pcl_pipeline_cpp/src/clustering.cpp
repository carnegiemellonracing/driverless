#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <unordered_map>

#include "rclcpp/rclcpp.hpp" // For propper logging
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using namespace pcl;

// Calculates Euclidean distance between two PointXYZ.
inline double euclideanDistance(const PointXYZ &a, const PointXYZ &b) {
  double dx = a.x - b.x;
  double dy = a.y - b.y;
  double dz = a.z - b.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Returns indices of points in cloud that are within epsilon of point.
inline std::vector<int> regionQuery(
  PointCloud<PointXYZ> &cloud,
  const PointXYZ &point,
  double epsilon) {
  std::vector<int> neighbors;
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
  std::vector<bool> &visited,
  std::vector<int> &cluster,
  int point_idx,
  std::vector<int> &neighbors,
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
  const std::unordered_map<int, std::vector<int>> &clusters) {

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
PointCloud<PointXYZ> DBSCAN(PointCloud<PointXYZ> &cloud, double epsilon, int min_points) {

  std::vector<bool> visited(cloud.points.size(), false);
  std::vector<int> cluster(cloud.points.size(), -1);

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
  std::unordered_map<int, std::vector<int>> clusters;
  for (size_t i = 0; i < cluster.size(); ++i) {
    if (cluster[i] > 0) {
      clusters[cluster[i]].push_back(static_cast<int>(i));
    }
  }

  return computeCentroids(cloud, clusters);
}

// Use for secondary filtering to get rid of extraneous clusters outside of cones
PointCloud<PointXYZ> DBSCAN2(PointCloud<PointXYZ> &cloud, double epsilon, int min_points) {
  // visited[i] indicates whether the point has been visited.
  // cluster[i] = -1 for unclassified, 0 for noise, >0 for cluster ID.
  std::vector<bool> visited(cloud.points.size(), false);
  std::vector<int> cluster(cloud.points.size(), -1);

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