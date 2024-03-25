tmux new-session -d;  # start new detached tmux session

#rosbag
tmux send 'source $DRIVERLESS/driverless_ws/install/setup.bash && clear' ENTER;
#tmux send 'ros2 bag record -a -o';   

#perceptions
tmux split-window -h -p 100;                            
tmux send 'source $DRIVERLESS/driverless_ws/install/setup.bash && ros2 run perceptions lidar_node' ENTER;                    
tmux split-window -v -p 70;
tmux send 'source $DRIVERLESS/driverless_ws/install/setup.bash && ros2 run perceptions cone_node_lidar' ENTER;                    
tmux split-window -v -p 70;
tmux send 'source $DRIVERLESS/driverless_ws/install/setup.bash && ros2 run perceptions yolov5_zed_node' ENTER;                    
tmux split-window -h -p 100;                            
tmux send 'source $DRIVERLESS/driverless_ws/install/setup.bash && ros2 run perceptions yolov5_zed2_node' ENTER;                    

#planning
tmux split-window -h -p 50;                             # split the detached tmux session
tmux send 'source $DRIVERLESS/driverless_ws/install/setup.bash && ros2 run perceptions midline_node' ENTER;                    
#TODO: raceline
#TODO: slam

#controls
tmux split-window -h -p 50;                             # split the detached tmux session
tmux send 'source $DRIVERLESS/driverless_ws/install/setup.bash && ros2 run controls controller_node' ENTER;    

#actuators
tmux split-window -h -p 50;                             # split the detached tmux session
tmux send 'source $DRIVERLESS/driverless_ws/install/setup.bash && ros2 run actuators throttle_node' ENTER;    

#sbg
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

