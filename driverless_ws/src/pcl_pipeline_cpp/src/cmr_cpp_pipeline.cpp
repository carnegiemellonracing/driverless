#include <cstdio>

// ROS2 Imports
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "pcl_conversions/pcl_conversions.h"


using namespace pcl;
constexpr bool verbose = true;

// redefine in main file bc I don't have header files written...
pcl::PointCloud<pcl::PointXYZ> GraceAndConrad(
    pcl::PointCloud<pcl::PointXYZ>& cloud,
    double alpha,
    int num_bins,
    double lower_height_threshold);

pcl::PointCloud<pcl::PointXYZ> DBSCAN(
    pcl::PointCloud<pcl::PointXYZ>& cloud,
    double epsilon,
    int min_points);

pcl::PointCloud<pcl::PointXYZ> DBSCAN2(
    pcl::PointCloud<pcl::PointXYZ>& cloud,
    double epsilon,
    int min_points);

class CMRCppPipelineNode : public rclcpp::Node {

  public:
    CMRCppPipelineNode() : Node("cmr_cpp_pipeline_node") {

      gnc_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "lidar_pipeline/ground_filtered_points", 1
      );

      dbs_1_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "lidar_pipeline/dbs_1", 1
      );

      cones_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "lidar_pipeline/cones", 1
      );

      movia_lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "microvision/pointcloud", 1,
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {points_callback(msg);}
      );

      hesai_lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "lidar_points", 1,
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {points_callback(msg);}
      );
      
    }

  private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr gnc_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dbs_1_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cones_pub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr movia_lidar_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr hesai_lidar_sub_;

    double alpha_ = .1;
    int num_bins_ = 3;
    double height_threshold_ = .12;

    double epsilon_ = .2;
    int min_points_ = 3;

    double epsilon_2_ = 3;
    int min_points_2_ = 3;

    void points_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
      // convert to PCL type
      PointCloud<PointXYZ> pcl_cloud;
      fromROSMsg(*msg, pcl_cloud); // takes in sensor_msgs::msg::PointCloud2 not SharedPtr

      PointCloud<PointXYZ> ground_filtered_cloud = GraceAndConrad(pcl_cloud,
        alpha_,
        num_bins_,
        height_threshold_
      );

      if (verbose) {
        sensor_msgs::msg::PointCloud2 gnc_output;
        toROSMsg(ground_filtered_cloud, gnc_output);
        gnc_pub_->publish(gnc_output);
      }

      PointCloud<PointXYZ> clustered_cloud = DBSCAN(ground_filtered_cloud,
        epsilon_,
        min_points_
      );

      if (verbose) {
        sensor_msgs::msg::PointCloud2 dbs_1_output;
        toROSMsg(ground_filtered_cloud, dbs_1_output);
        dbs_1_pub_->publish(dbs_1_output);
      }

      PointCloud<PointXYZ> filtered_cloud = DBSCAN2(clustered_cloud,
        epsilon_2_,
        min_points_2_
      );

      sensor_msgs::msg::PointCloud2 cones;
      toROSMsg(ground_filtered_cloud, cones);
      cones_pub_->publish(cones);
    }


};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CMRCppPipelineNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
