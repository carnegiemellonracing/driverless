tmux new-session -d;  # start new detached tmux session

#rosbag
tmux send 'source $DRIVERLESS/driverless_ws/install/setup.bash' ENTER;
#tmux send 'ros2 bag record -a -o';   

#perceptions
tmux split-window -h -p 100;                            
tmux send 'source $DRIVERLESS/driverless_ws/install/setup.bash' ENTER;                    
tmux split-window -v -p 70;
tmux send 'source $DRIVERLESS/driverless_ws/install/setup.bash' ENTER;             

#planning
tmux select-pane -t 0
tmux split-window -h -p 50;                             # split the detached tmux session
tmux send 'squirrel' ENTER;     
                    
# open (attach) tmux session.
tmux a;                                                 