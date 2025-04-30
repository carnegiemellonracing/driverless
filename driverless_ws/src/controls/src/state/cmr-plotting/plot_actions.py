import argparse
import matplotlib.pyplot as plt

# Input is a log file

def parse_lines(log_lines):
        swangle_values = []
        torque_values = []
        timestamp_values = []
        latency_values = []
        current_torque_value = 0
        for line in log_lines:
            line = line.strip()
            if "mppi step complete" in line:
                timestamp_values.append(float(line.split("] [")[1]))
            if line.startswith("torque"):
                torque_string = (line.split(': ')[1]).strip()
                current_torque_value += float(torque_string)
            if line.startswith("torque_rr"):
                torque_values.append(current_torque_value)
                current_torque_value = 0
            if line.startswith("swangle"):
                 swangle_value = float((line.split(': ')[1]).strip())
                 swangle_values.append(swangle_value)
            if line.startswith("Total Latency"):
                 latency_value = float((line.split(': ')[1]).strip())
                 latency_values.append(latency_value)
                 
        print("Swangle values: ", swangle_values)
        print("Torque values: ", torque_values)
        swangle_sum = max(swangle_values) 
        torque_sum = max(torque_values)
        latency_sum = max(latency_values)

        minimum_length = min(len(swangle_values), len(torque_values), len(timestamp_values), len(latency_values))
        swangle_values = swangle_values[:minimum_length]
        torque_values = torque_values[:minimum_length]
        timestamp_values = timestamp_values[:minimum_length]
        latency_values = latency_values[:minimum_length]
        
        time = list(range(len(swangle_values)))
        for i in range(len(time)):
             swangle_values[i]  /= swangle_sum
             torque_values[i] /= torque_sum
             latency_values[i] /= latency_sum


        plt.plot(timestamp_values, swangle_values, c='r', label='swangle')
        plt.plot(timestamp_values, torque_values, c='b', label='torque')
        plt.plot(timestamp_values, latency_values, c='g', label='latency')
        plt.legend()
        plt.show()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build controls")
    parser.add_argument("log_filename")
    args = parser.parse_args()
    with open(args.log_filename, "r") as f:
        log_lines = f.readlines()
        parse_lines(log_lines)


