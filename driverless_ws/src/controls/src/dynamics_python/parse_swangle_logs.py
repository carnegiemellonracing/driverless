import re
import sys
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(log_file_path):
    # Lists to store parsed data
    action_a_values = []
    action_b_values = []
    time_values = []
    
    # Regular expression to match the pattern - updated to handle negative numbers
    pattern = r'\[WARN\].*? Swangle:([-\d.]+)\|([-\d.]+)\|([\d.]+)'
    
    # Read the log file
    with open(log_file_path, 'r') as file:
        for line in file:
            if '[WARN]' in line:
                match = re.search(pattern, line)
                if match:
                    action_a = float(match.group(1))
                    action_b = float(match.group(2))
                    timestamp = float(match.group(3))
                    
                    action_a_values.append(action_a)
                    action_b_values.append(action_b)
                    time_values.append(timestamp)
    
    return action_a_values, action_b_values, time_values

def plot_actions(action_a_values, action_b_values, time_values):
    # Convert to numpy arrays
    action_a = np.array(action_a_values)
    action_b = np.array(action_b_values)
    time = np.array(time_values)
    
    # If we have timestamps, normalize them to start from 0
    if len(time) > 0:
        time = time - time[0]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(time, action_a, 'b-', label='Action A')
    plt.plot(time, action_b, 'r-', label='Action B')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Action Value')
    plt.title('Actions A and B Over Time')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig('swangle_actions_plot.png')
    print(f"Plot saved as 'swangle_actions_plot.png'")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_swangle_logs.py <log_file_path>")
        sys.exit(1)
    
    log_file_path = sys.argv[1]
    
    try:
        action_a_values, action_b_values, time_values = parse_log_file(log_file_path)
        
        if not action_a_values:
            print("No matching data found in the log file.")
            sys.exit(1)
            
        print(f"Found {len(action_a_values)} data points.")
        plot_actions(action_a_values, action_b_values, time_values)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 