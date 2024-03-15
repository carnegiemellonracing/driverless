tmux new-session -d;  # start new detached tmux session
tmux send 'source ~/Documents/driverless/driverless_ws/install/setup.bash && clear' ENTER;
tmux send 'ros2 bag record -a -o';                    
tmux split-window -h -p 100;                             # split the detached tmux session
tmux send 'source ~/Documents/driverless/driverless_ws/install/setup.bash && ros2 launch perceptions sensor_launch.py' ENTER;                    
tmux split-window -v -p 70;
tmux send 'source ~/Documents/driverless/driverless_ws/install/setup.bash' ENTER;                    
tmux split-window -h -p 100;                             # split the detached tmux session
tmux send 'source ~/Documents/driverless/driverless_ws/install/setup.bash' ENTER;                    
tmux split-window -h -p 60;                             # split the detached tmux session
tmux send 'source ~/movella_ws/install/setup.bash && ros2 launch bluespace_ai_xsens_mti_driver xsens_mti_node.launch.py' ENTER;                    
tmux a;                                                 # open (attach) tmux session.
