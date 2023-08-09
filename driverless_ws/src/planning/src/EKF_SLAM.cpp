#include <memory>

#include "rclcpp/rclcpp.hpp"

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "eufs_msgs/msg/cone_array_with_covariance.hpp"
#include "eufs_msgs/msg/car_state.hpp"

#include <boost/shared_ptr.hpp>
#include <vector>
#include <cmath>
#include <chrono>

#define CONE_DATA_TOPIC "/cones"
#define VEHICLE_DATA_TOPIC "/ground_truth/state"

using namespace std;
using std::placeholders::_1;

struct Cone{
  double x;
  double y;

};

struct VehiclePosition{
  double x;
  double y;
  double yaw;
  double dx;
  double dy;
  double dyaw;
};
class SLAMValidation : public rclcpp::Node
{
  public:
    SLAMValidation(): Node("slam_validation"){
      cone_sub = this->create_subscription<eufs_msgs::msg::ConeArrayWithCovariance>(
      "/cones", 10, std::bind(&SLAMValidation::cone_callback, this, _1));
      vehicle_state_sub = this->create_subscription<eufs_msgs::msg::CarState>(
      "/ground_truth/state", 10, std::bind(&SLAMValidation::vehicle_state_callback, this, _1));
      timer = this->create_wall_timer(100ms, std::bind(&SLAMValidation::timer_callback, this));
    }
  private:
    void cone_callback(const eufs_msgs::msg::ConeArrayWithCovariance::SharedPtr cone_data){
      // RCLCPP_INFO(this->get_logger(), "B: %i| Y: %i| O: %i", cone_data->blue_cones.size(), cone_data->yellow_cones.size(), cone_data->orange_cones.size());
      for(int i = 0; i < cone_data->blue_cones.size(); i++){
        Cone to_add;
        to_add.x = cone_data->blue_cones[i].point.x;
        to_add.y = cone_data->blue_cones[i].point.y;
        blue_cones.push_back(to_add);
      }
      for(int i = 0; i < cone_data->yellow_cones.size(); i++){
        Cone to_add;
        to_add.x = cone_data->yellow_cones[i].point.x;
        to_add.y = cone_data->yellow_cones[i].point.y;
        yellow_cones.push_back(to_add);
      }
      for(int i = 0; i < cone_data->orange_cones.size(); i++){
        Cone to_add = {cone_data->orange_cones[i].point.x, cone_data->orange_cones[i].point.y};
        orange_cones.push_back(to_add);
      }
    }
    void vehicle_state_callback(const eufs_msgs::msg::CarState::SharedPtr vehicle_state_data){
      // RCLCPP_INFO(this->get_logger(), "vehicle state\n");
      double q1 = vehicle_state_data->pose.pose.orientation.x;
      double q2 = vehicle_state_data->pose.pose.orientation.y;
      double q3 = vehicle_state_data->pose.pose.orientation.z;
      double q0 = vehicle_state_data->pose.pose.orientation.w;
      double yaw = atan2(2*(q0*q3+q1*q2), pow(q0, 2)+pow(q1, 2)-pow(q2, 2)-pow(q3, 2));

      vpos_is_jacob.x = vehicle_state_data->pose.pose.position.x;
      vpos_is_jacob.y = vehicle_state_data->pose.pose.position.y;
      vpos_is_jacob.yaw = yaw;
      vpos_is_jacob.dx = vehicle_state_data->twist.twist.linear.x;
      vpos_is_jacob.dy = vehicle_state_data->twist.twist.linear.y;
      vpos_is_jacob.dyaw = vehicle_state_data->twist.twist.angular.z;
    }
    void timer_callback(){
      run_slam(std::chrono::system_clock::now());
    }

    void run_slam(auto curr_time){
      dt = std::chrono::duration_cast<std::chrono::microseconds>(curr_time - prev_time).count() / 1000000.0;
      prev_time = curr_time;
      RCLCPP_INFO(this->get_logger(), "curr_time: %d | prev_time: %d | dt: %d\n", curr_time.count(), prev_time.count(), dt);

    }
    rclcpp::Subscription<eufs_msgs::msg::ConeArrayWithCovariance>::SharedPtr cone_sub;
    rclcpp::Subscription<eufs_msgs::msg::CarState>::SharedPtr vehicle_state_sub;
    vector<Cone> blue_cones;
    vector<Cone> yellow_cones;
    vector<Cone> orange_cones;

    VehiclePosition vpos_is_jacob;
    rclcpp::TimerBase::SharedPtr timer;

    auto prev_time;
    double dt;
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
//                   vehicle_state_data->pose.x,
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