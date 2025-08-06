#pragma once

#include "sensor_msgs/msg/point_cloud2.hpp"

// Typedef for ease of use
using PointCloud2 = sensor_msgs::msg::PointCloud2;


/**
 * Function implementing DBSCAN on a PointCloud2
 * @param cloud reference to a PointCloud2 obj
 * @param epsilon double side of neighborhood for dbscan
 * @param min_points int number of points to constitute a cluster
 * 
 * @return a discrete set of cone centroids representing a cluster
 */
PointCloud2 cluster_points(
    PointCloud2 &cloud,
    double epsilon,
    int min_points
);

/**
 * Function removing extraneous clusters via an implementation of DBSCAN on a PointCloud2. 
 * Slight variation in that it checks that exactly one point is there per 3 meters. (this ensures that )
 * @param cloud reference to a PointCloud2 obj
 * @param epsilon double side of neighborhood for dbscan. set to 3.0 by default as cones are 3m apart
 * @param num_points int number of points to constitute a cluster
 * 
 * @return a discrete set of cone centroids (minus extraneous clusters) representing cones
 */
PointCloud2 remove_extraneous_clusters(
    PointCloud2 &cloud,
    double epsilon=3.0,
    int num_points=1
);