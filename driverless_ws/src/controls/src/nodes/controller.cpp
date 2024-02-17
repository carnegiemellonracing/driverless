#include <mppi/mppi.hpp>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <types.hpp>
#include <constants.hpp>
#include <interfaces/msg/control_action.hpp>
#include <state/state_estimator.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "controller.hpp"

namespace controls {
    namespace nodes {

            ControllerNode::ControllerNode(
                std::unique_ptr<state::StateEstimator> state_estimator,
                std::unique_ptr<mppi::MppiController> mppi_controller)
                : Node {controller_node_name},

                  m_state_estimator {std::move(state_estimator)},
                  m_mppi_controller {std::move(mppi_controller)},

                  m_action_timer {
                      create_wall_timer(
                          std::chrono::duration<float>(1),
                          [this] { publish_action_callback(); })
                  },

                  m_action_publisher {
                      create_publisher<interfaces::msg::ControlAction>(
                          control_action_topic_name,
                          control_action_qos)
                  },

                  m_spline_subscription {
                      create_subscription<SplineMsg>(
                          spline_topic_name, spline_qos,
                          [this] (const SplineMsg::SharedPtr msg) { spline_callback(*msg); })
                  },

                  m_slam_subscription {
                      create_subscription<SlamMsg>(
                          spline_topic_name, spline_qos,
                          [this] (const SlamMsg::SharedPtr msg) { slam_callback(*msg); })
                  },

#ifdef PUBLISH_STATES
                  m_state_trajectory_publisher {
                      create_publisher<visualization_msgs::msg::MarkerArray>(
                          state_trajectories_topic_name,
                          state_trajectories_qos)
                  },
#endif

                  m_action_read {std::make_unique<Action>()},
                  m_action_write {std::make_unique<Action>()} { }

            void ControllerNode::publish_action_callback() {
                std::lock_guard<std::mutex> action_read_guard {action_read_mut};

                publish_action(*m_action_read);
            }

            void ControllerNode::spline_callback(const SplineMsg& spline_msg) {
                std::cout << "Received spline" << std::endl;

                m_state_estimator->on_spline(spline_msg);
                run_mppi();
            }

            void ControllerNode::slam_callback(const SlamMsg& slam_msg) {
                std::cout << "Received slam" << std::endl;

                m_state_estimator->on_slam(slam_msg);
                run_mppi();
            }

            void ControllerNode::publish_action(const Action& action) const {
                interfaces::msg::ControlAction msg;
                msg.swangle = action[action_swangle_idx];
                msg.torque_f = action[action_torque_f_idx];
                msg.torque_r = action[action_torque_r_idx];

                // std::cout << "Publishing message: { swangle: " << msg.swangle << ", torque_f: "
                //           << msg.torque_f << ", torque_r: " << msg.torque_r << " }" << std::endl;

                m_action_publisher->publish(msg);
            }

#ifdef PUBLISH_STATES
            void ControllerNode::publish_state_trajectories(const std::vector<float>& state_trajectories) {
                visualization_msgs::msg::MarkerArray paths {};

                std::cout << "States:" << std::endl;
                for (int i = 0; i < num_samples; i++) {
                    for (int j = 0; j < num_timesteps; j++) {
                        std::cout << "{ ";
                        for (int k = 0; k < state_dims; k++) {
                            std::cout << state_trajectories[i * num_timesteps * state_dims + j * state_dims + k] << " ";
                        }
                        std::cout << " }";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;


                for (uint32_t i = 0; i < num_samples; i++) {
                    visualization_msgs::msg::Marker lines {};
                    lines.type = visualization_msgs::msg::Marker::LINE_STRIP;
                    lines.header.stamp = get_clock()->now();
                    lines.header.frame_id = "/world_frame";
                    lines.id = i;
                    lines.pose.orientation.w = 1.0f;
                    lines.color.r = 1.0f;
                    lines.color.a = 1.0f;
                    lines.scale.y = lines.scale.z = 1.0f;
                    lines.scale.x = 0.0001f;

                    for (uint32_t j = 0; j < num_timesteps; j++) {
                        geometry_msgs::msg::Point point;

                        const float* state = &state_trajectories[
                            i * num_timesteps * state_dims + j * state_dims
                        ];

                        point.x = state[state_x_idx];
                        point.y = state[state_y_idx];
                        point.z = 0;

                        lines.points.push_back(point);
                    }

                    paths.markers.push_back(lines);
                }

                m_state_trajectory_publisher->publish(paths);
            }
#endif

            void ControllerNode::run_mppi() {
                std::cout << "-------- RUN MPPI -------" << std::endl;

                std::cout << "locking action_write" << std::endl;
                {
                    std::lock_guard<std::mutex> action_write_guard {action_write_mut};

                    std::cout << "generating action" << std::endl;
                    *m_action_write = m_mppi_controller->generate_action();
                }

                std::cout << "swapping action buffers" << std::endl;
                swap_action_buffers();

#ifdef PUBLISH_STATES
                RCLCPP_DEBUG(get_logger(), "publishing state trajectories");
                publish_state_trajectories(m_mppi_controller->last_state_trajectories());
#endif
            }

            void ControllerNode::swap_action_buffers() {
                std::lock(action_read_mut, action_write_mut);  // lock both using a deadlock avoidance scheme
                std::lock_guard<std::mutex> action_read_guard {action_read_mut, std::adopt_lock};
                std::lock_guard<std::mutex> action_write_guard {action_write_mut, std::adopt_lock};

                std::swap(m_action_read, m_action_write);
            }

    }
}


int main(int argc, char *argv[]) {
    using namespace controls;

    std::unique_ptr<state::StateEstimator> state_estimator = state::StateEstimator::create();
    std::unique_ptr<mppi::MppiController> controller = mppi::MppiController::create();

    rclcpp::init(argc, argv);
    std::cout << "rclcpp initialized" << std::endl;

    const auto node = std::make_shared<nodes::ControllerNode>(
        std::move(state_estimator),
        std::move(controller)
    );

    std::cout << "controller node created" << std::endl;

    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}