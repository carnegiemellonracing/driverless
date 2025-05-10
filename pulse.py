import subprocess
import os

EXPECTED_NODES = ["/point_to_pixel",
         "/controller",
         "/cone_history_test_node",
         "/xsens_mti_node"
        #  "hesai_ros_driver_node"
         ]

def get_nodes():
    try:
        output = subprocess.check_output(["ros2", "node", "list"], env=os.environ.copy())
        return output.decode().strip().split("\n")
    except Exception as e:
        print(e)
        return []

def pulse():    
    active_nodes = get_nodes()
    print(active_nodes)
    for node in EXPECTED_NODES:
        if node not in active_nodes:
            if (node == "/point_to_pixel"):
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.2", "C-c"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.2", "ros2 run point_to_pixel p2p.sh", "C-m"])
        
            elif (node == "/cone_history_test_node"):
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.3", "C-c"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.3", "ros2 run point_to_pixel cone_history_test_node", "C-m"])
            elif (node == "/xsens_mti_node"):
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.0", "C-c"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.0", "ros2 launch xsens_mti_ros2_driver xsens_mti_node.launch.py", "C-m"])
            elif (node == "/controller"):
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.4", "C-z"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.4", "k9", "C-m"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.4", "k9", "C-m"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.4", "k9", "C-m"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.4", "k9", "C-m"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.4", "ros2 run controls controller", "C-m"])
            elif (node == "/hesai_ros_driver_node"):
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.1", "C-z"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.1", "k9", "C-m"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.1", "k9", "C-m"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.1", "k9", "C-m"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.1", "k9", "C-m"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.1", "ros2 run hesai_ros_driver hesai_ros_driver_node", "C-m"])
    return

    print("All nodes found")



if __name__ == "__main__":
    pulse()