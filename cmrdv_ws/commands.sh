set_throttle() {
    ros2 topic pub /throttle std_msgs/Int64 "data: $1"
}

set_steering() {
    ros2 topic pub /steer std_msgs/Int64 "data: $1"
}

run_dim_heartbeat() { 
    ros2 run cmrdv_common dim_heartbeat
}

run_dim_request() {
    ros2 run cmrdv_common dim_request
}

run_steering() {
    ros2 run cmrdv_actuators steering
}

run_fsm() {
    ros2 run cmrdv_actuators fsm
}

run_actuators() {
    run_steering;
    run_fsm;
}

bringup_pipeline() {
    ros2 launch cmrdv_bringup pipeline.launch.py mode:=trackdrive
}

