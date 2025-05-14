import subprocess
import threading
import time
import os

EXPECTED_NODES = ["/point_to_pixel",
         "/controller",
         "/cone_history_test_node",
         "/xsens_mti_node"
         "hesai_ros_driver_node"
         ]

EXPECTED_TOPICS = [# "/filter/twist",
                   # "/filter/quaternion", 
                   # "/filter/euler", 
                   # "/cpp_cones",
                     "/perc_cones",
                     "/associated_perc_cones",
                     "/control_action"
                  ] 

def get_nodes():
    try:
        output = subprocess.check_output(["ros2", "node", "list"], env=os.environ.copy())
        return output.decode().strip().split("\n")
    except Exception as e:
        return []
    
def check_topic(index, topic_statuses):
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    output = subprocess.Popen(f"source /opt/ros/foxy/setup.bash && source /home/chip/Documents/driverless/driverless_ws/install/setup.bash && exec ros2 topic echo {EXPECTED_TOPICS[index]}", 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                shell=True, executable="/bin/bash", env=env, bufsize=0, text=True)
    
    time.sleep(5)
    output.terminate()
    stdout = output.communicate(timeout=10.0)[0]

    if '---' in stdout:
        topic_statuses[index] = True
    else:
        topic_statuses[index] = False




def pulse():    
    active_nodes = get_nodes()
    print("\n Active nodes:")
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

    topic_statuses = [False for i in range(len(EXPECTED_TOPICS))]
    threads = []
    for i in range(len(EXPECTED_TOPICS)):
        # check_topic(i, topic_statuses)
        t = threading.Thread(target=check_topic, args=(i, topic_statuses))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    
    print('\nPublishing?')
    print(topic_statuses)
    print(EXPECTED_TOPICS)
    print('\n\n')
    for i, (topic, status) in enumerate(zip(EXPECTED_TOPICS, topic_statuses)):
        if not status:
            if (topic == "/control_action"): # Controller Check
                # Restart IMU
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.0", "C-c"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.0", "ros2 launch xsens_mti_ros2_driver xsens_mti_node.launch.py", "C-m"])

                # Restart Cone History
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.3", "C-c"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.3", "ros2 run point_to_pixel cone_history_test_node", "C-m"])

            elif (topic == "/associated_perc_cones"): # Cone History Check
                # Restart IMU
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.0", "C-c"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.0", "ros2 launch xsens_mti_ros2_driver xsens_mti_node.launch.py", "C-m"])

                # Restart Point to Pixel
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.2", "C-c"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.2", "ros2 run point_to_pixel p2p.sh", "C-m"])

            elif (topic == "/perc_cones"): # Point to Pixel Check
                # Restart lidar
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.1", "C-z"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.1", "k9", "C-m"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.1", "k9", "C-m"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.1", "k9", "C-m"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.1", "k9", "C-m"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.1", "ros2 run hesai_ros_driver hesai_ros_driver_node", "C-m"])

                # Restart IMU
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.0", "C-c"])
                subprocess.run(["tmux", "send-keys", "-t", "pipeline:0.0", "ros2 launch xsens_mti_ros2_driver xsens_mti_node.launch.py", "C-m"])
    return

if __name__ == "__main__":
    pulse()