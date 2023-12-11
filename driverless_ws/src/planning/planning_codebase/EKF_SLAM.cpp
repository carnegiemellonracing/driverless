#include <memory>

#include "rclcpp/rclcpp.hpp"

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "eufs_msgs/msg/cone_array_with_covariance.hpp"
#include "eufs_msgs/msg/cone_array.hpp"
#include "eufs_msgs/msg/car_state.hpp"
#include "eufs_msgs/msg/slam_frame.hpp"
#include <gsl/gsl_block.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>


// #include "new_slam.cpp"
// Adding new_slam_correct.cpp
#include "new_slam_correct.cpp"

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
  double dx;
  double dy;
  double dyaw;
};

class SLAMValidation : public rclcpp::Node
{
  public:
    SLAMValidation(): Node("slam_validation"){
      // gsl_matrix_set_identity(pEst);

      slamframe_sub = this->create_subscription<eufs_msgs::msg::SlamFrame>(
      "/SLAMFrame", 10, std::bind(&SLAMValidation::slamframe_callback, this, _1));
      timer = this->create_wall_timer(100ms, std::bind(&SLAMValidation::timer_callback, this));
    }
  private:
    void slamframe_callback(const eufs_msgs::msg::SlamFrame::SharedPtr slamframe){
      eufs_msgs::msg::ConeArray cone_data = slamframe->stereo_cones;
      geometry_msgs::msg::Vector3Stamped imu_linear_velocity = slamframe->imu_linear_velocity;
      sensor_msgs::msg::Imu imu_data = slamframe->imu_data;

      RCLCPP_INFO(this->get_logger(), "CONECALLBACK: B: %i| Y: %i| O: %i", cone_data.blue_cones.size(), cone_data.yellow_cones.size(), cone_data.orange_cones.size());
      blue_cones.clear();
      yellow_cones.clear();
      orange_cones.clear();
      for(int i = 0; i < cone_data.blue_cones.size(); i++){
        Cone to_add;
        to_add.x = cone_data.blue_cones[i].x;
        to_add.y = cone_data.blue_cones[i].y;
        blue_cones.push_back(to_add);
      }
      for(int i = 0; i < cone_data.yellow_cones.size(); i++){
        Cone to_add;
        to_add.x = cone_data.yellow_cones[i].x;
        to_add.y = cone_data.yellow_cones[i].y;
        yellow_cones.push_back(to_add);
      }
      for(int i = 0; i < cone_data.orange_cones.size(); i++){
        Cone to_add = {cone_data.orange_cones[i].x, cone_data.orange_cones[i].y};
        orange_cones.push_back(to_add);
      }

      vpos_is_jacob.dx = imu_linear_velocity.vector.x;
      vpos_is_jacob.dy = -imu_linear_velocity.vector.z;
      vpos_is_jacob.dyaw = imu_data.angular_velocity.y;

    }
    void timer_callback(){
      RCLCPP_INFO(this->get_logger(), "hello\n");
      RCLCPP_ERROR(this->get_logger(), "ERRRO!\n");
      run_slam();
    }

    void run_slam(){
      // dt = std::chrono::duration_cast<std::chrono::microseconds>(curr_time - prev_time).count() / 1000000.0;
      // prev_time = curr_time;
      // RCLCPP_INFO(this->get_logger(), "curr_time: %d | prev_time: %d | dt: %d\n", curr_time.count(), prev_time.count(), dt);
      // Time difference from one callback to another is so close to 0.1 seconds, gonna assume 0.1 everywhere
      // Cuz elapsed time stuff not working rn


      // gsl_matrix* u = gsl_matrix_calloc(2, 1);
      // gsl_matrix_set(u, 0, 0, hypot(vpos_is_jacob.dx, vpos_is_jacob.dy));
      // gsl_matrix_set(u, 1, 0, vpos_is_jacob.dyaw);

      // Eigen::MatrixXd u = calcInput();
      Eigen::MatrixXd u(2, 1);
      u << hypot(vpos_is_jacob.dx, vpos_is_jacob.dy),
           vpos_is_jacob.dyaw;

      int n = blue_cones.size() + yellow_cones.size() + orange_cones.size();
      int idx = 0;

      // Declaring z matrix dimensions
      Eigen::MatrixXd z(n, 2);

      // gsl_matrix* z = gsl_matrix_calloc(n, 3);
      RCLCPP_INFO(this->get_logger(), "RUNSLAM: B: %i | Y: %i | O: %i\n", blue_cones.size(), yellow_cones.size(), orange_cones.size());
      for(int i = 0; i < blue_cones.size(); i++){
        Cone c = blue_cones[i];
        double dist = hypot(c.x, c.y);
        double angle = atan2(c.y, c.x);
        double corrected_angle = std::fmod(angle + M_PI, 2 * M_PI) - M_PI;
        // gsl_matrix_set(z, idx, 0, dist);
        // gsl_matrix_set(z, idx, 1, angle);
        z(idx, 0) = dist;
        z(idx, 1) = angle;
        idx++;
      }

      for(int i = 0; i < yellow_cones.size(); i++){
        Cone c = yellow_cones[i];
        double dist = hypot(c.x, c.y);
        double angle = atan2(c.y, c.x);
        double corrected_angle = std::fmod(angle + M_PI, 2 * M_PI) - M_PI;
        // gsl_matrix_set(z, idx, 0, dist);
        // gsl_matrix_set(z, idx, 1, angle);
        z(idx, 0) = dist;
        z(idx, 1) = angle;
        idx++;
      }

      for(int i = 0; i < orange_cones.size(); i++){
        Cone c = orange_cones[i];
        double dist = hypot(c.x, c.y);
        double angle = atan2(c.y, c.x);
        double corrected_angle = std::fmod(angle + M_PI, 2 * M_PI) - M_PI;
        // gsl_matrix_set(z, idx, 0, dist);
        // gsl_matrix_set(z, idx, 1, angle);
        z(idx, 0) = dist;
        z(idx, 1) = angle;
        idx++;
      }
      RCLCPP_INFO(this->get_logger(), "before ekf slam");
      // slam_output = ekf_slam(xEst, pEst, u, z, 0.1, this->get_logger());
      if (z.rows() != 0){
        RCLCPP_INFO(this->get_logger(), "in herebb\n");
        slam_output = ekf_slam(this->get_logger(), xEst, pEst, u, z, 0.1);
      }
      RCLCPP_INFO(this->get_logger(), "got output\n");
      RCLCPP_INFO(this->get_logger(), "NUM_LANDMARKS: %i\n", (xEst.rows()-3)/2);
      // RCLCPP_INFO(this->get_logger(), "Position: %f, %f, %f", xEst(0, 0), xEst(1, 0), xEst(2, 0));
      xEst = slam_output.x;
      pEst = slam_output.p;

    }
    rclcpp::Subscription<eufs_msgs::msg::SlamFrame>::SharedPtr slamframe_sub;
    vector<Cone> blue_cones;
    vector<Cone> yellow_cones;
    vector<Cone> orange_cones;

    VehiclePosition vpos_is_jacob;
    rclcpp::TimerBase::SharedPtr timer;

    // gsl_matrix* xEst = gsl_matrix_calloc(STATE_SIZE, 1);
    // gsl_matrix* pEst = gsl_matrix_calloc(STATE_SIZE, STATE_SIZE);

    // Initialize xEst matrix with zeros
    Eigen::MatrixXd xEst = Eigen::MatrixXd::Zero(STATE_SIZE, 1);
    // Initialize PEst matrix as an identity matrix
    Eigen::MatrixXd pEst = Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);

    double dt;

    ekfPackage slam_output;
};

int main(int argc, char * argv[]){
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SLAMValidation>();
  RCLCPP_INFO(node->get_logger(), "got output\n");
  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}