#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rcllscpp.hpp"
#include "std_msgs/msg/string.hpp"

// #include "interfaces/msg/Heartbeat.hpp"

// #include "common/config/collection_config.hpp"
// #include "common/config/common_config.hpp"
// #include "common/CAN/can_types.hpp"

// Usage of the imported constants/enums in your C++ code
const auto RELIABLE_QOS_PROFILE = common::config::RELIABLE_QOS_PROFILE;
const auto GLOBAL_TIMEOUT_SEC = common::config::GLOBAL_TIMEOUT_SEC;
const auto ALIVE_STATE = common::CAN::ALIVE_STATE;
const auto WARNING_STATE = common::CAN::WARNING_STATE;
const auto ERROR_STATE = common::CAN::ERROR_STATE;
const auto TX10HZ_PERIOD_S = common::CAN::TX10HZ_PERIOD_S;


// don't forget to import types when done

class HeartbeatNode : public rclcpp::Node {
  public:
    HeartbeatNode(const std::string& node_name, double timer_period = TX10HZ_PERIOD_S) 
    : Node(node_name),
      node_name_(node_name)
    {
      time_ = rclcpp::Clock().now();
      timer_ = this->create_wall_timer(std::chrono::duration<double>(timer_period_), 
                                std::bind(&HeartbeatNode::node_status_pub, this));

      threshold_ =rclcpp::Duration(4 * timer_period);

      // TODO: perhaps change node status to correspond to CAN status
      node_status_ = ALIVE_STATE;
      global_status_ = ALIVE_STATE;

      status_publisher_ = create_publisher<interfaces::msg::Heartbeat>(node_name + "_status", rclcpp::QoS(10).reliable());
      global_heartbeat_subscription_ = create_subscription<interfaces::msg::Heartbeat>(
          "global_heartbeat",
          rclcpp::QoS(10).reliable(),
          std::bind(&HeartbeatNode::update_status, this, std::placeholders::_1));

			// TODO: why use global timeout sec and not check at 10hz?
			last_global_heartbeat_ = rclcpp::Clock().now();
			check_global_timeout_ = this->create_wall_timer(std::chrono::seconds(GLOBAL_TIMEOUT_SEC),
																					std::bind(&HeartbeatNode::check_global_heartbeat_timeout, this));
    }


		void node_status_pub() {
			time_ = rclcpp::Clock().now();

			interfaces::msg::Heartbeat heartbeat;

			// NOTE: Updating the node status to be in line with the global status
			// what is node_status??????
			if (global_status_ != ERROR_STATE && node_status_ != ERROR_STATE) {
					node_status = global_status_;
			} else {
					// You can add more logic here if needed
					node_status = ERROR_STATE;
			}

			if (node_status != node_status_) {
					RCLCPP_INFO(get_logger(), "publishing %s status: %s", node_name_.c_str(), node_status.c_str());
			}

			node_status_ = node_status;

			heartbeat.status = node_status_;
			heartbeat.header.stamp = time.to_msg();
			heartbeat.header.frame_id = node_name_;

			status_publisher_->publish(heartbeat);
		}


		void check_global_heartbeat_timeout() {
			rclcpp::Time current_time = rclcpp::Clock().now();

			if (current_time - last_global_heartbeat_ > threshold_) {
					global_status_ = ERROR_STATE;
					RCLCPP_INFO(get_logger(), "GLOBAL HEARTBEAT TIMEOUT!");
			}
		}

		void update_status(const interfaces::msg::Heartbeat::SharedPtr msg) {
			global_status_ = msg->status;
			last_global_heartbeat_ = rclcpp::Clock().now();
			// get_logger()->info("receving hb status: " + std::to_string(msg->status));
		}

		void panic() {
			node_status_ = ERROR_STATE;
		}

		void clear_error() {
			node_status_ = ALIVE_STATE;
		}

		bool alive() {
			return (node_status_ != ERROR_STATE);
		}

}


class Perceptions : public HeartbeatNode {
	public:
		Perceptions() : HeartbeatNode("perceptions") {
			count_ = 0;
			timer_ = this->create_wall_timer(std::chrono::milliseconds(500), 
															std::bind(&Perceptions::increment_count, this));
		}

	private:
		void increment_count() {
			count_++;
			std::cout << count_ << std::endl;
			if (!alive()) { // how to call parent function?
				std::cout << "ERROR" << std::endl;
			}
			if (count_ > 10) {
				panic();
			}
		}

		int count_;
  	rclcpp::TimerBase::SharedPtr timer_;

};




class Planning : public HeartbeatNode {
	public:
		Planning() : HeartbeatNode("planning") {
			count_ = 0;
			timer_ = this->create_wall_timer(std::chrono::seconds(1), 
																std::bind(&Planning::increment_count, this));
		}

	private:
		void increment_count() {
			count_++;
			if (!alive()) {
				std::cout << "ERROR" << std::endl;
			}
			if (count_ > 100) {
				panic();
			}
		}

		int count_;
    rclcpp::TimerBase::SharedPtr timer_;

};


class DIM : public HeartbeatNode {
	public:
		DIM() : HeartbeatNode("dim") {
			count_ = 0;
			timer_ = this->create_wall_timer(std::chrono::seconds(1), [this]() {
				increment_count();
			});

			state_subscriber_ = this->create_subscription<std_msgs::msg::String>(
				"DIM_request", 10,
				std::bind(&DIM::process_state_request, this, std::placeholders::_1)
			);
		}

	private:
		void process_state_request(const std_msgs::msg::String::SharedPtr msg) {
			RCLCPP_INFO(this->get_logger(), "DIM REQUEST: %s", msg->data.c_str());
		}

		void increment_count() {
			count_++;
			if (!alive()) {
				std::cout << "ERROR" << std::endl;
			}
		}

		int count_;
		rclcpp::TimerBase::SharedPtr timer_;
		rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_subscriber_;
	};



int main(int argc, char **argv) {
	rclcpp::init(argc, argv);

	// DIM Node
	// auto dimNode = std::make_shared<DIM>();
	// rclcpp::spin(dimNode);
	// rclcpp::shutdown();

	// Perceptions Node
	// auto perceptionsNode = std::make_shared<Perceptions>();
	// rclcpp::spin(perceptionsNode);
	// rclcpp::shutdown();

	// Planning Node
	// auto planningNode = std::make_shared<Planning>();

	auto node = std::make_shared<HeartbeatNode>("node");

	rclcpp::spin(planningNode);
	rclcpp::shutdown();

	return 0;
}