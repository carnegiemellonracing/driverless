tmux new-session -d;  # start new detached tmux session
tmux send 'source ~/Documents/driverless/driverless_ws/install/setup.bash && ros2 run perceptions yolov5_zed_node';
# tmux send 'ros2 bag record -a -o';                    
tmux split-window -h -p 100;                             # split the detached tmux session
tmux send 'source ~/Documents/driverless/driverless_ws/install/setup.bash && ros2 run perceptions yolov5_zed2_node';                    
tmux split-window -v -p 70;
tmux send 'source ~/Documents/driverless/driverless_ws/install/setup.bash && ros2 run perceptions lidar_node';                    
tmux split-window -h -p 100;                             # split the detached tmux session
tmux send 'source ~/Documents/driverless/driverless_ws/install/setup.bash && ros2 run perceptions cone_node';                    
tmux split-window -h -p 60;                             # split the detached tmux session
tmux send 'source ~/movella_ws/install/setup.bash && ros2 launch bluespace_ai_xsens_mti_driver xsens_mti_node.launch.py';                    
tmux split-window -h -p 70;
tmux send 'source ~/Documents/driverless/driverless_ws/install/setup.bash && ros2 run perceptions midline_node';
tmux select-pane -t 0;
tmux split-window -v -p 70;
tmux send 'source ~/Documents/driverless/driverless_ws/install/setup.bash && ros2 run controls controller';
tmux split-window -v -p 70;
tmux send 'source ~/Documents/driverless/driverless_ws/install/setup.bash && ros2 run actuators throttle_node';
tmux a;                                                 # open (attach) tmux session.

