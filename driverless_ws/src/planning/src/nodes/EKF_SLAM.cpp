#include <memory>
#include <string>
#include <cstring>

#include "rclcpp/rclcpp.hpp"
//Message includes
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/temperature.hpp"
#include "eufs_msgs/msg/cone_array_with_covariance.hpp"
#include "eufs_msgs/msg/car_state.hpp"

//Message Filter
#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"

#include <boost/shared_ptr.hpp>

#define CONE_DATA_TOPIC "/cones"
#define VEHICLE_DATA_TOPIC "/ground_truth/state"

using std::placeholders::_1;
using std::placeholders::_2;

class SLAMValidation : public rclcpp::Node {
  public:
    SLAMValidation(): Node("EKF_SLAM"){
      cone_sub.subscribe(this, CONE_DATA_TOPIC);
      vehicle_state_sub.subscribe(this, VEHICLE_DATA_TOPIC);

      sync_ = std::make_shared<message_filters::TimeSynchronizer<eufs_msgs::msg::ConeArrayWithCovariance, 
                                                            eufs_msgs::msg::CarState>>(cone_sub, vehicle_state_sub, 3);
      sync_->registerCallback(std::bind(&SLAMValidation::topic_callback, this, _1, _2));
    }

    void run_slam(const eufs_msgs::msg::ConeArrayWithCovariance::SharedPtr cone_data, 
                  const eufs_msgs::msg::CarState::SharedPtr vehicle_state_data) const {
      RCLCPP_INFO(this->get_logger(), 
                  "Blue: %i, Yellow: %i, Orange: %i\n", 
                  sizeof(cone_data->blue_cones)/sizeof(cone_data->blue_cones[0]),
                  sizeof(cone_data->yellow_cones)/sizeof(cone_data->yellow_cones[0]),
                  sizeof(cone_data->orange_cones)/sizeof(cone_data->orange_cones[0]));
      RCLCPP_INFO(this->get_logger(), 
                  "X: %f, Y: %f, Z: %f\n", 
                  vehicle_state_data->pose.pose.position.x,
                  vehicle_state_data->pose.pose.position.y,
                  vehicle_state_data->pose.pose.position.z);
      RCLCPP_INFO(this->get_logger(),
                  "-------------------------------\n\n");
    }

  private:
    message_filters::Subscriber<eufs_msgs::msg::ConeArrayWithCovariance> cone_sub;
    message_filters::Subscriber<eufs_msgs::msg::CarState> vehicle_state_sub;
    std::shared_ptr<message_filters::TimeSynchronizer<eufs_msgs::msg::ConeArrayWithCovariance, eufs_msgs::msg::CarState>> sync_;

  void topic_callback(const eufs_msgs::msg::ConeArrayWithCovariance::ConstSharedPtr& tmp_1, 
                        const eufs_msgs::msg::CarState::ConstSharedPtr& tmp_2) const{
    const char *temp_1 = std::to_string(tmp_1->header.stamp.sec).c_str(); //change to actual values
    const char *temp_2 = std::to_string(tmp_2->header.stamp.sec).c_str();
    RCLCPP_INFO(this->get_logger(), "Cone Array Time: '%s' \n Car State time: '%s'", temp_1, temp_2);
  }
};

int main(int argc, char * argv[]){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SLAMValidation>());
  rclcpp::shutdown();
  return 0;
}