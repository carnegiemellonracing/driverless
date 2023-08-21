#include <memory>
#include <stdio.h>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include "eufs_msgs/msg/cone_array_with_covariance.hpp"
#include "eufs_msgs/msg/cone_with_covariance.hpp"
#include "geometry_msgs/msg/point.hpp"


#include "pcl_conversions.h"

#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <Eigen/Dense>

void create_cones(std::vector<pcl::PointXYZ> &centers, eufs_msgs::msg::ConeArrayWithCovariance::SharedPtr &msg) {
	std::vector<eufs_msgs::msg::ConeWithCovariance> blue_cones;
	std::vector<eufs_msgs::msg::ConeWithCovariance> yellow_cones;

	for (const auto& p: centers) {
		geometry_msgs::msg::Point point;
		point.x = p.x;
		point.y = p.y;
		point.z = p.z;

		eufs_msgs::msg::ConeWithCovariance cone;
		cone.point = point;
		cone.covariance = { 1.f, 0.f, 0.f, 1.f };
				
		if (p.x < 0) {
			blue_cones.push_back(cone);
		} else {
			yellow_cones.push_back(cone);
		}

	}

	msg->blue_cones = blue_cones;
	msg->yellow_cones = yellow_cones;
}


class LidarSubscriber : public rclcpp::Node {
	public:
		LidarSubscriber(): Node("lidar_subscriber") {
			printf("Initialized LiDAR Node\n");
			subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/velodyne_points", 10, std::bind(&LidarSubscriber::callback, this, std::placeholders::_1));
			ground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/ground_points", 10);
			non_ground_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/non_ground_points", 10);
			center_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/center_points", 10);
			cone_publisher_ = this->create_publisher<eufs_msgs::msg::ConeArrayWithCovariance>("/lidar_cones", 10);
		}
	private:
		
		

		void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) const {
			pcl::PointCloud<pcl::PointXYZ>::Ptr pcl(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::fromROSMsg(*msg, *pcl);

			pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
			pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

			// create segmentation object
			pcl::SACSegmentation<pcl::PointXYZ> seg;
			seg.setOptimizeCoefficients(true);

			// required settings
			seg.setModelType(pcl::SACMODEL_PLANE);
			seg.setMethodType(pcl::SAC_RANSAC);
			seg.setDistanceThreshold(0.01);

			seg.setInputCloud(pcl);
			seg.segment(*inliers, *coefficients);

			RCLCPP_INFO(this->get_logger(), "%d points of %d", inliers->indices.size(), pcl->size());

			// given indices that split point cloud, separate the points
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_not_plane (new pcl::PointCloud<pcl::PointXYZ>);

			pcl::ExtractIndices<pcl::PointXYZ> filter;
			filter.setInputCloud(pcl);
			filter.setIndices(inliers);
			filter.filter(*cloud_plane);
			filter.setNegative(true);
			filter.filter(*cloud_not_plane);

			// extract cluster from the point cloud with the ground filtered out
			pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
			tree->setInputCloud(cloud_not_plane);

			std::vector<pcl::PointIndices> cluster_indices;
			pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
			ec.setClusterTolerance(0.1); // 10 cm
			ec.setMinClusterSize(2);
			ec.setMaxClusterSize(999999999);
			ec.setSearchMethod(tree);
			ec.setInputCloud(cloud_not_plane);
			ec.extract(cluster_indices);

			// extract clusters and cluster centers
			int j = 0;
			RCLCPP_INFO(this->get_logger(), "found %d clusters", cluster_indices.size());
			std::vector<pcl::PointXYZ> centers;
			for (const auto& cluster : cluster_indices) {
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
				
				float x = 0.0f, y = 0.0f, z = 0.0f;
				int size = 0;
				pcl::PointXYZ p;
				for (const auto& idx : cluster.indices) {
					// create point cloud of cluster
					p = (*cloud_not_plane)[idx];
					cloud_cluster->push_back((*cloud_not_plane)[idx]); 

					// update cluster center aggregate
					x += p.x;
					y += p.y;
					z += p.z;
					size += 1;
				}

				// compute the cluster center and add to list of centers
				x /= size;
				y /= size;
				z /= size;
				centers.push_back(pcl::PointXYZ(x, y, z));
				RCLCPP_INFO(this->get_logger(), "\tx=%.3f, y=%.3f, d=%.3f", x, y, z);


				cloud_cluster->width = cloud_cluster->size();
				cloud_cluster->height = 1;
				cloud_cluster->is_dense = true;

				RCLCPP_INFO(this->get_logger(), "\tcluster: %d with %d points", j, cloud_cluster->width);
				j++;
			}
			RCLCPP_INFO(this->get_logger(), "# things: %d", centers.size());

			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_centers (new pcl::PointCloud<pcl::PointXYZ>);
			for (const auto& p : centers) {
				cloud_centers->push_back(p);
			}
			cloud_centers->width = cloud_centers->size();
			cloud_centers->height = 1;
			cloud_centers->is_dense = true;

			// get color information for the centers
			

			// create output message type
			eufs_msgs::msg::ConeArrayWithCovariance::SharedPtr message_cones (new eufs_msgs::msg::ConeArrayWithCovariance());
			create_cones(centers, message_cones);

			// publish point cloud information
			sensor_msgs::msg::PointCloud2::SharedPtr message_ground (new sensor_msgs::msg::PointCloud2());
			sensor_msgs::msg::PointCloud2::SharedPtr message_non_ground (new sensor_msgs::msg::PointCloud2());
			sensor_msgs::msg::PointCloud2::SharedPtr message_center (new sensor_msgs::msg::PointCloud2());

			pcl::toROSMsg<pcl::PointXYZ>(*cloud_plane, *message_ground);
			pcl::toROSMsg<pcl::PointXYZ>(*cloud_not_plane, *message_non_ground);
			pcl::toROSMsg<pcl::PointXYZ>(*cloud_centers, *message_center);

			ground_publisher_->publish(*message_ground);
			non_ground_publisher_->publish(*message_non_ground);
			center_publisher_->publish(*message_center);
			cone_publisher_->publish(*message_cones);


						
		}

		rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
		rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_publisher_;
		rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr non_ground_publisher_;
		rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr center_publisher_;
		rclcpp::Publisher<eufs_msgs::msg::ConeArrayWithCovariance>::SharedPtr cone_publisher_;
		pcl::PointCloud<pcl::PointXYZ> cloud;
};

int main(int argc, char *argv[]) {
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<LidarSubscriber>());
	rclcpp::shutdown();
}
