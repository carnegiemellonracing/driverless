tmux new-session -d;  # start new detached tmux session

#rosbag
tmux send 'source $DRIVERLESS/driverless_ws/install/setup.bash' ENTER;
#tmux send 'ros2 bag record -a -o';   

#perceptions
tmux split-window -h -p 100;  
tmux send 'source ~/movella_ws/install/setup.bash && ros2 run bluespace_ai_xsens_mti_driver xsens_mti_node' ENTER;                    
tmux split-window -v -p 70;
tmux send 'source $DRIVERLESS/driverless_ws/install/setup.bash && ros2 run hesai_ros_driver hesai_ros_driver_node' ENTER;

#planning
tmux select-pane -t 0
tmux split-window -h -p 50;                             # split the detached tmux session
tmux send 'source $DRIVERLESS/driverless_ws/install/setup.bash && ros2 run actuators throttle_node' ENTER;    
tmux split-window -v -p 70;
tmux send 'sudo mount /dev/sda /storage1' ENTER;         
tmux send 'chip22a' ENTER;         
tmux send 'cd /storage1' ENTER;         



tmux select-pane -t 2
tmux split-window -h -p 100;
tmux send 'source $DRIVERLESS/driverless_ws/install/setup.bash && ros2 run perceptions endtoend_node' ENTER;                

tmux select-pane -t 4
tmux split-window -h -p 100;
tmux send 'source $DRIVERLESS/driverless_ws/install/setup.bash && ros2 run controls controller' ENTER;

# open (attach) tmux session.
tmux a;                                                 