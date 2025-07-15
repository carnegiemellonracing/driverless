#include "generic_slam_node.cpp"

namespace nodes {
template class GenericSLAMNode<interfaces::msg::ConeArray,
                                geometry_msgs::msg::TwistStamped,
                                geometry_msgs::msg::QuaternionStamped,
                                geometry_msgs::msg::Vector3Stamped>;


template class GenericSLAMNode<interfaces::msg::ConeArray,
                                geometry_msgs::msg::TwistStamped,
                                geometry_msgs::msg::QuaternionStamped,
                                geometry_msgs::msg::PoseStamped>;

}