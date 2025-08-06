#pragma once

#include "sensor_msgs/msg/point_cloud2.hpp"

// Typedef for ease of use
using PointCloud2 = sensor_msgs::msg::PointCloud2;

/**
 * Function implementing the GraceAndConrad algorithm
 * @param cloud: The input vector of rectangular points to parse
 * @param alpha: The size of each segment (radians)
 * @param num_bins: The number of bins per segment
 * @param height_threshold: Keep all points this distance above the best fit line
 * @return A point cloud of ground-filtered points
 */

PointCloud2 filter_ground(
    PointCloud2 cloud,
    double alpha, 
    int num_bins,
    double height_threshold
);
