#include <chrono>
#include <functional>
#include <memory>
#include <string>


// fix imports
#include "rclcpp/rclcpp.hpp"
#include "common/config/collection_config.hpp"
#include "common/config/common_config.hpp"
#include "common/CAN/can_types.hpp"
#include "interfaces/msg/Heartbeat.hpp"



/*
 * Simple struct-like class to store status of each node.
 *
 */
struct NodeStatus {
  std::string name;
  int last_time = 0;
  int status = ALIVE_STATE;
  bool started = false;
  void* subscription = nullptr;
};



/**
 * @brief 
 * 
 * The GlobalHeartbeat class publishes its current status at 10Hz and processes
 * vital heartbeats from other nodes at 10HZ. It publishes its current state to the 'global_heartbeat'
 * topic of type interfaces::msg::Heartbeat.
 * 
 * @param timer_period publish rate 
 */
class GlobalHeartbeat : public rclcpp::Node {
  public:
    GlobalHeartbeat(double timer_period = TX10HZ_PERIOD_S)
      : Node("global_heartbeat"),
        beating(true)
    {
      global_heartbeat_publisher_ = this->create_publisher<interfaces::msg::Heartbeat>(
          "global_heartbeat", RELIABLE_QOS_PROFILE);
      
      timer_ = this->create_timer(std::chrono::duration<double>(timer_period),
                                  std::bind(&GlobalHeartbeat::hb_handler, this));
      // NOTE: this can be changed but this should be about 3-4 timer periods, currently 4 timer periods
      threshold_ = rclcpp::Duration(4 * timer_period);
      
      // create a ode for each vital node
      // self.nodes = {}
      updateRequiredNodes();
    }

  private:
    void updateRequiredNodes() {
      for (const auto& node : HEARTBEAT_REQUIRED_NODES) {
        if (nodes.find(node) != nodes.end()) {
          continue;
        }

        nodes[node] = std::make_unique<NodeStatus>(node);
        nodes[node]->subscription = create_subscription<interfaces::msg::Heartbeat>(
            node + "_status", RELIABLE_QOS_PROFILE,
            std::bind(&GlobalHeartbeat::nodeCheck, this, std::placeholders::_1));
        nodes[node]->last_time = this->get_clock()->now();
      }
    }


    void nodeCheck(const interfaces::msg::Heartbeat::SharedPtr msg) {
      auto node = nodes[msg->header.frame_id];

      if (!node->started) {
        node->started = true;
        node->last_time = this->get_clock()->now();
      }

      // Create Time object from the header
      auto curr_time = Time(msg->header.stamp);

      // Checking node status
      // Checking last time node published message
      if (msg->status != ERROR_STATE && (curr_time - node->last_time) <= threshold_) {
        node->status = ALIVE_STATE;
      } else {
        node->status = ERROR_STATE;
      }
      node->last_time = curr_time;

      // For debugging
      RCLCPP_INFO(this->get_logger(), "%s: %d time: %f", msg->header.frame_id.c_str(),
                  msg->status, (curr_time - node->last_time).seconds());
    }


    void hbHandler() {
      auto curr_time = this->get_clock()->now();

      for (const auto& nodePair : nodes) {
        const auto& node_name = nodePair.first;
        const auto& node = nodePair.second;
        bool node_ok = ((node->status != ERROR_STATE &&
                        (curr_time - node->last_time) <= threshold_) ||
                        (!node->started));

        RCLCPP_INFO_ONCE(this->get_logger(), "%s received: ERROR_STATE: %d, TIMEOUT: %d",
                        node_name.c_str(), (node->status != ERROR_STATE),
                        (curr_time - node->last_time) <= threshold_);

        this->beating = this->beating && node_ok;
      }

      auto heartbeat = interfaces::msg::Heartbeat();
      if (this->beating) {
        heartbeat.status = ALIVE_STATE;
      } else {
        heartbeat.status = ERROR_STATE;
      }

      heartbeat.header.frame_id = "global_heartbeat";
      heartbeat.header.stamp = curr_time;

      // Uncomment the following line if you want to publish the heartbeat
      // this->global_heartbeat->publish(heartbeat);
    }


    /**
     * Immediately tell the system to error out by setting the alive boolean to False
     */
    void panic() {
      this->beating = false;
    }


    /**
     * Tell the system to reset itself by setting the alive boolean to True
     */
    void clearError() {
      this->beating = true;
    }


    /**
     * Whether or not the system is set to error out
     */
    bool isAlive() {
      return this->beating;
    }


    bool beating;
    rclcpp::Publisher<interfaces::msg::Heartbeat>::SharedPtr global_heartbeat_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Duration threshold_;
    std::map<std::string, rclcpp::Node::SharedPtr> nodes;
};



int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);

  auto heartbeat = std::make_shared<GlobalHeartbeat>();

  rclcpp::spin(heartbeat);

  rclcpp::shutdown();
  return 0;
}

