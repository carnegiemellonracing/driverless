set_throttle() {
    ros2 topic pub /throttle std_msgs/Int64 "data: $1"
}

set_steering() {
    ros2 topic pub /steer std_msgs/Int64 "data: $1"
}

alias run_dim_heartbeat='ros2 run cmrdv_common dim_heartbeat'

alias run_dim_request='ros2 run cmrdv_common dim_request'

alias run_steering='ros2 run cmrdv_actuators steering'

alias run_fsm='ros2 run cmrdv_actuators fsm'

alias bringup_pipeline='ros2 launch cmrdv_bringup pipeline.launch.py mode:=trackdrive'



