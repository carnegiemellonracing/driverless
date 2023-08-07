// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from eufs_msgs:msg/EKFErr.idl
// generated code does not contain a copyright notice

#ifndef EUFS_MSGS__MSG__DETAIL__EKF_ERR__BUILDER_HPP_
#define EUFS_MSGS__MSG__DETAIL__EKF_ERR__BUILDER_HPP_

#include "eufs_msgs/msg/detail/ekf_err__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace eufs_msgs
{

namespace msg
{

namespace builder
{

class Init_EKFErr_ekf_yaw_var
{
public:
  explicit Init_EKFErr_ekf_yaw_var(::eufs_msgs::msg::EKFErr & msg)
  : msg_(msg)
  {}
  ::eufs_msgs::msg::EKFErr ekf_yaw_var(::eufs_msgs::msg::EKFErr::_ekf_yaw_var_type arg)
  {
    msg_.ekf_yaw_var = std::move(arg);
    return std::move(msg_);
  }

private:
  ::eufs_msgs::msg::EKFErr msg_;
};

class Init_EKFErr_ekf_y_acc_var
{
public:
  explicit Init_EKFErr_ekf_y_acc_var(::eufs_msgs::msg::EKFErr & msg)
  : msg_(msg)
  {}
  Init_EKFErr_ekf_yaw_var ekf_y_acc_var(::eufs_msgs::msg::EKFErr::_ekf_y_acc_var_type arg)
  {
    msg_.ekf_y_acc_var = std::move(arg);
    return Init_EKFErr_ekf_yaw_var(msg_);
  }

private:
  ::eufs_msgs::msg::EKFErr msg_;
};

class Init_EKFErr_ekf_x_acc_var
{
public:
  explicit Init_EKFErr_ekf_x_acc_var(::eufs_msgs::msg::EKFErr & msg)
  : msg_(msg)
  {}
  Init_EKFErr_ekf_y_acc_var ekf_x_acc_var(::eufs_msgs::msg::EKFErr::_ekf_x_acc_var_type arg)
  {
    msg_.ekf_x_acc_var = std::move(arg);
    return Init_EKFErr_ekf_y_acc_var(msg_);
  }

private:
  ::eufs_msgs::msg::EKFErr msg_;
};

class Init_EKFErr_ekf_y_vel_var
{
public:
  explicit Init_EKFErr_ekf_y_vel_var(::eufs_msgs::msg::EKFErr & msg)
  : msg_(msg)
  {}
  Init_EKFErr_ekf_x_acc_var ekf_y_vel_var(::eufs_msgs::msg::EKFErr::_ekf_y_vel_var_type arg)
  {
    msg_.ekf_y_vel_var = std::move(arg);
    return Init_EKFErr_ekf_x_acc_var(msg_);
  }

private:
  ::eufs_msgs::msg::EKFErr msg_;
};

class Init_EKFErr_ekf_x_vel_var
{
public:
  explicit Init_EKFErr_ekf_x_vel_var(::eufs_msgs::msg::EKFErr & msg)
  : msg_(msg)
  {}
  Init_EKFErr_ekf_y_vel_var ekf_x_vel_var(::eufs_msgs::msg::EKFErr::_ekf_x_vel_var_type arg)
  {
    msg_.ekf_x_vel_var = std::move(arg);
    return Init_EKFErr_ekf_y_vel_var(msg_);
  }

private:
  ::eufs_msgs::msg::EKFErr msg_;
};

class Init_EKFErr_imu_yaw_err
{
public:
  explicit Init_EKFErr_imu_yaw_err(::eufs_msgs::msg::EKFErr & msg)
  : msg_(msg)
  {}
  Init_EKFErr_ekf_x_vel_var imu_yaw_err(::eufs_msgs::msg::EKFErr::_imu_yaw_err_type arg)
  {
    msg_.imu_yaw_err = std::move(arg);
    return Init_EKFErr_ekf_x_vel_var(msg_);
  }

private:
  ::eufs_msgs::msg::EKFErr msg_;
};

class Init_EKFErr_imu_y_acc_err
{
public:
  explicit Init_EKFErr_imu_y_acc_err(::eufs_msgs::msg::EKFErr & msg)
  : msg_(msg)
  {}
  Init_EKFErr_imu_yaw_err imu_y_acc_err(::eufs_msgs::msg::EKFErr::_imu_y_acc_err_type arg)
  {
    msg_.imu_y_acc_err = std::move(arg);
    return Init_EKFErr_imu_yaw_err(msg_);
  }

private:
  ::eufs_msgs::msg::EKFErr msg_;
};

class Init_EKFErr_imu_x_acc_err
{
public:
  explicit Init_EKFErr_imu_x_acc_err(::eufs_msgs::msg::EKFErr & msg)
  : msg_(msg)
  {}
  Init_EKFErr_imu_y_acc_err imu_x_acc_err(::eufs_msgs::msg::EKFErr::_imu_x_acc_err_type arg)
  {
    msg_.imu_x_acc_err = std::move(arg);
    return Init_EKFErr_imu_y_acc_err(msg_);
  }

private:
  ::eufs_msgs::msg::EKFErr msg_;
};

class Init_EKFErr_gps_y_vel_err
{
public:
  explicit Init_EKFErr_gps_y_vel_err(::eufs_msgs::msg::EKFErr & msg)
  : msg_(msg)
  {}
  Init_EKFErr_imu_x_acc_err gps_y_vel_err(::eufs_msgs::msg::EKFErr::_gps_y_vel_err_type arg)
  {
    msg_.gps_y_vel_err = std::move(arg);
    return Init_EKFErr_imu_x_acc_err(msg_);
  }

private:
  ::eufs_msgs::msg::EKFErr msg_;
};

class Init_EKFErr_gps_x_vel_err
{
public:
  explicit Init_EKFErr_gps_x_vel_err(::eufs_msgs::msg::EKFErr & msg)
  : msg_(msg)
  {}
  Init_EKFErr_gps_y_vel_err gps_x_vel_err(::eufs_msgs::msg::EKFErr::_gps_x_vel_err_type arg)
  {
    msg_.gps_x_vel_err = std::move(arg);
    return Init_EKFErr_gps_y_vel_err(msg_);
  }

private:
  ::eufs_msgs::msg::EKFErr msg_;
};

class Init_EKFErr_header
{
public:
  Init_EKFErr_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_EKFErr_gps_x_vel_err header(::eufs_msgs::msg::EKFErr::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_EKFErr_gps_x_vel_err(msg_);
  }

private:
  ::eufs_msgs::msg::EKFErr msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::eufs_msgs::msg::EKFErr>()
{
  return eufs_msgs::msg::builder::Init_EKFErr_header();
}

}  // namespace eufs_msgs

#endif  // EUFS_MSGS__MSG__DETAIL__EKF_ERR__BUILDER_HPP_
