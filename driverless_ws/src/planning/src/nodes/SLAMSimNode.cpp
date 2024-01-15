#include <memory>

#include "rclcpp/rclcpp.hpp"

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "eufs_msgs/msg/cone_array_with_covariance.hpp"
#include "eufs_msgs/msg/car_state.hpp"
#include "interfaces/msg/slam_output.hpp"

// #include "new_slam.cpp"
// Adding new_slam_correct.cpp
#include "../planning_codebase/ekf_slam/ekf_slam.cpp"

#include <boost/shared_ptr.hpp>
#include <vector>
#include <cmath>
#include <chrono>
// #include <eigen3/Eigen/Dense>

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



class SLAMSimNode : public rclcpp::Node
{
  public:
    SLAMSimNode(): Node("sim_validation_node"){

      // Sim Cone Subscriber
      cone_sub = this->create_subscription<eufs_msgs::msg::ConeArrayWithCovariance>(
      "/cones", 10, std::bind(&SLAMSimNode::cone_callback, this, _1));

      // Sim Vehicle State Subscriber
      vehicle_state_sub = this->create_subscription<eufs_msgs::msg::CarState>(
      "/ground_truth/state", 10, std::bind(&SLAMSimNode::vehicle_state_callback, this, _1));

      slam_output_pub = this->create_publisher<interfaces::msg::SlamOutput>("/slam_output",10);
      
      // Timer to execute slam callback
      timer = this->create_wall_timer(100ms, std::bind(&SLAMSimNode::run_slam, this));
    }
  private:
    // Callback to store most recent cones output by sim
    void cone_callback(const eufs_msgs::msg::ConeArrayWithCovariance::SharedPtr cone_data){
      RCLCPP_INFO(this->get_logger(), "CONECALLBACK: B: %i| Y: %i| O: %i", cone_data->blue_cones.size(), cone_data->yellow_cones.size(), cone_data->orange_cones.size());

      // Clear cones from previous frame
      blue_cones.clear();
      yellow_cones.clear();
      orange_cones.clear();

      // Store all blue cones from current sim frame
      for(int i = 0; i < cone_data->blue_cones.size(); i++){
        Cone to_add;
        to_add.x = cone_data->blue_cones[i].point.x;
        to_add.y = cone_data->blue_cones[i].point.y;
        blue_cones.push_back(to_add);
      }

      // Store all yellow cones from current sim frame
      for(int i = 0; i < cone_data->yellow_cones.size(); i++){
        Cone to_add;
        to_add.x = cone_data->yellow_cones[i].point.x;
        to_add.y = cone_data->yellow_cones[i].point.y;
        yellow_cones.push_back(to_add);
      }

      // Store all orange cones from current sim frame
      for(int i = 0; i < cone_data->orange_cones.size(); i++){
        Cone to_add;
        to_add.x = cone_data->orange_cones[i].point.x;
        to_add.y = cone_data->orange_cones[i].point.y;
        orange_cones.push_back(to_add);
      }
    }
    
    // Callback to store most recent vehicle state output by sim
    void vehicle_state_callback(const eufs_msgs::msg::CarState::SharedPtr vehicle_state_data){
      double q1 = vehicle_state_data->pose.pose.orientation.x;
      double q2 = vehicle_state_data->pose.pose.orientation.y;
      double q3 = vehicle_state_data->pose.pose.orientation.z;
      double q0 = vehicle_state_data->pose.pose.orientation.w;
      double yaw = atan2(2*(q0*q3+q1*q2), pow(q0, 2)+pow(q1, 2)-pow(q2, 2)-pow(q3, 2));

      vehicle_pos.x = vehicle_state_data->pose.pose.position.x;
      vehicle_pos.y = vehicle_state_data->pose.pose.position.y;
      vehicle_pos.yaw = yaw;
      vehicle_pos.dx = vehicle_state_data->twist.twist.linear.x;
      vehicle_pos.dy = vehicle_state_data->twist.twist.linear.y;
      vehicle_pos.dyaw = vehicle_state_data->twist.twist.angular.z;
    }

    void run_slam(){
      // Generate input velocity matrix, u
      Eigen::MatrixXd u(2, 1);
      u << hypot(vehicle_pos.dx, vehicle_pos.dy), // linear velocity
           vehicle_pos.dyaw;                      // angular velocity

      // Create single matrix called z to store all blue, yellow, orange cones
      int n = blue_cones.size() + yellow_cones.size() + orange_cones.size();
      Eigen::MatrixXd z(n, 2);
      int idx = 0;

      // Iterate through all blue cones and store in z
      for(int i = 0; i < blue_cones.size(); i++){
        Cone c = blue_cones[i];
        double dist = hypot(c.x, c.y);
        double angle = atan2(c.y, c.x);
        z(idx, 0) = dist;
        z(idx, 1) = angle;
        idx++;
      }

      // Iterate through all yellow cones and store in z
      for(int i = 0; i < yellow_cones.size(); i++){
        Cone c = yellow_cones[i];
        double dist = hypot(c.x, c.y);
        double angle = atan2(c.y, c.x);
        z(idx, 0) = dist;
        z(idx, 1) = angle;
        idx++;
      }

      // Iterate through all orange cones and store in z
      for(int i = 0; i < orange_cones.size(); i++){
        Cone c = orange_cones[i];
        double dist = hypot(c.x, c.y);
        double angle = atan2(c.y, c.x);
        z(idx, 0) = dist;
        z(idx, 1) = angle;
        idx++;
      }
      
      // Only run ekf_slam if the number of cones is greater than 0
      if (z.rows() != 0){
        slam_output = ekf_slam(this->get_logger(), xEst, pEst, u, z, 0.1);
      }

      xEst = slam_output.x;
      pEst = slam_output.p;
      int num_landmarks = (xEst.rows()-3)/2;
      RCLCPP_INFO(this->get_logger(), "NUM_LANDMARKS: %i\n", num_landmarks);
      
      // Construct SLAMOutput message
      interfaces::msg::SlamOutput slam_output_msg = interfaces::msg::SlamOutput();
      slam_output_msg.car_x = xEst(0, 0);
      slam_output_msg.car_y = xEst(1, 0);
      slam_output_msg.car_heading = xEst(2, 0);

      std::vector<geometry_msgs::msg::Point> landmarks(num_landmarks);
      for(int i = 0; i < num_landmarks; i++){
        geometry_msgs::msg::Point curr_landmark = geometry_msgs::msg::Point();
        curr_landmark.x = xEst(2*i+3, 0);
        curr_landmark.y = xEst(2*i+4, 0);
        landmarks[i] = curr_landmark;
      }
      slam_output_msg.landmarks = landmarks;
      slam_output_pub->publish(slam_output_msg);
    }


    // ------ TOPIC SUBSCRIBERS + PUBLISHERS ------
    rclcpp::Subscription<eufs_msgs::msg::ConeArrayWithCovariance>::SharedPtr cone_sub;
    rclcpp::Subscription<eufs_msgs::msg::CarState>::SharedPtr vehicle_state_sub;
    rclcpp::Publisher<interfaces::msg::SlamOutput>::SharedPtr slam_output_pub;

    // ------ CONE ARRAYS ------
    vector<Cone> blue_cones;
    vector<Cone> yellow_cones;
    vector<Cone> orange_cones;

    VehiclePosition vehicle_pos;
    rclcpp::TimerBase::SharedPtr timer;

    // ------ STATE MATRIX - Initialized to all zeros ------
    Eigen::MatrixXd xEst = Eigen::MatrixXd::Zero(STATE_SIZE, 1);

    // ------ COVARIANCE MATRIX - Initialized to identity ------
    Eigen::MatrixXd pEst = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);

    double dt;
    ekfPackage slam_output;
};

int main(int argc, char * argv[]){
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SLAMSimNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}