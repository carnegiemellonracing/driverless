#include <memory>

#include "rclcpp/rclcpp.hpp"

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "eufs_msgs/msg/cone_array_with_covariance.hpp"
#include "eufs_msgs/msg/car_state.hpp"

#include <boost/shared_ptr.hpp>

#define CONE_DATA_TOPIC "/cones"
#define VEHICLE_DATA_TOPIC "/ground_truth/state"

using std::placeholders::_1;

class SLAMValidation : public rclcpp::Node
{
  public:
    SLAMValidation(): Node("slam_validation"){
      cone_sub = this->create_subscription<eufs_msgs::msg::ConeArrayWithCovariance>(
      "/cones", 10, std::bind(&SLAMValidation::cone_callback, this, _1));
      vehicle_state_sub = this->create_subscription<eufs_msgs::msg::CarState>(
      "/ground_truth/state", 10, std::bind(&SLAMValidation::vehicle_state_callback, this, _1));
    }

  private:
    void cone_callback(const eufs_msgs::msg::ConeArrayWithCovariance::SharedPtr cone_data) const{
      RCLCPP_INFO(this->get_logger(), "cones\n");
    }
    void vehicle_state_callback(const eufs_msgs::msg::CarState::SharedPtr vehicle_state_data) const{
      RCLCPP_INFO(this->get_logger(), "vehicle state\n");
    }
    rclcpp::Subscription<eufs_msgs::msg::ConeArrayWithCovariance>::SharedPtr cone_sub;
    rclcpp::Subscription<eufs_msgs::msg::CarState>::SharedPtr vehicle_state_sub;
};

// class SLAMValidation : public rclcpp::Node {
//   public:
//     SLAMValidation(): Node("EKF_SLAM"){
//       RCLCPP_INFO(this->get_logger(), "ur mom was here\n");
//       cone_sub = this->create_subscription<eufs_msgs::msg::ConeArrayWithCovariance>("/cones", 10, std::bind(&SLAMValidation::cone_callback, this, _1));
//       vehicle_state_sub = this->create_subscription<eufs_msgs::msg::ConeArrayWithCovariance>("/ground_truth/state", 10, std::bind(&SLAMValidation::vehicle_state_callback, this, _1));
//       // this->create_subscription<eufs_msgs::msg::ConeArrayWithCovariance>("/cones", 10, std::bind(&SLAMValidation::topic_callback, this, _1));
//       // cone_sub = message_filters::Subscriber<eufs_msgs::msg::ConeArrayWithCovariance>(this, "/cones");
//       // vehicle_state_sub = message_filters::Subscriber<eufs_msgs::msg::CarState>(this, "/ground_truth/state");
//       // typedef message_filters::sync_policies::ApproximateTime<eufs_msgs::msg::ConeArrayWithCovariance, 
//       //                                                       eufs_msgs::msg::CarState> approx_sync_policy;
//       // message_filters::Synchronizer<approx_sync_policy> approx_sync(approx_sync_policy(10), cone_sub, vehicle_state_sub);
//       // approx_sync.setMaxIntervalDuration(rclcpp::Duration(1, 0));
//       // approx_sync.registerCallback(std::bind(&SLAMValidation::run_slam, this, std::placeholders::_1, std::placeholders::_2));
//       // sync.reset(new Sync(MySyncPolicy(10), cone_sub, vehicle_state_sub));
//       // sync->registerCallback(boost::bind(&SLAMValidation::run_slam, this, _1, _2));
//     }

//     void cone_callback(const eufs_msgs::msg::ConeArrayWithCovariance::SharedPtr cone_data){
//       RCLCPP_INFO(this->get_logger(), "cones\n");
//     };

//     void vehicle_state_callback(const eufs_msgs::msg::CarState::SharedPtr vehicle_state_data){
//       RCLCPP_INFO(this->get_logger(), "vehicle state\n");
//     };
//     void run_slam(const eufs_msgs::msg::ConeArrayWithCovariance::ConstSharedPtr cone_data, 
//                   const eufs_msgs::msg::CarState::ConstSharedPtr vehicle_state_data) const {
//       RCLCPP_INFO(this->get_logger(), 
//                   "Blue: %i, Yellow: %i, Orange: %i\n", 
//                   sizeof(cone_data->blue_cones)/sizeof(cone_data->blue_cones[0]),
//                   sizeof(cone_data->yellow_cones)/sizeof(cone_data->yellow_cones[0]),
//                   sizeof(cone_data->orange_cones)/sizeof(cone_data->orange_cones[0]));
//       RCLCPP_INFO(this->get_logger(), 
//                   "X: %f, Y: %f, Z: %f\n", 
//                   vehicle_state_data->pose.pose.position.x,
//                   vehicle_state_data->pose.pose.position.y,
//                   vehicle_state_data->pose.pose.position.z);
//       RCLCPP_INFO(this->get_logger(),
//                   "-------------------------------\n\n");
//     }
  
//   private:
//     // void topic_callback(const eufs_msgs::msg::ConeArrayWithCovariance msg) const
//     // {
//     //   RCLCPP_INFO(this->get_logger(), 
//     //               "Blue: %i, Yellow: %i, Orange: %i\n", 
//     //               sizeof(cone_data->blue_cones)/sizeof(cone_data->blue_cones[0]),
//     //               sizeof(cone_data->yellow_cones)/sizeof(cone_data->yellow_cones[0]),
//     //               sizeof(cone_data->orange_cones)/sizeof(cone_data->orange_cones[0]));
//     // }
//     // message_filters::Subscriber<eufs_msgs::msg::ConeArrayWithCovariance> cone_sub;
//     // message_filters::Subscriber<eufs_msgs::msg::CarState> vehicle_state_sub;
//     rclcpp::Subscription<eufs_msgs::msg::ConeArrayWithCovariance> cone_sub;
//     rclcpp::Subscription<eufs_msgs::msg::CarState> vehicle_state_sub;

    

// };

int main(int argc, char * argv[]){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SLAMValidation>());
  return 0;
}