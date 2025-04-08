import os
import matplotlib.pyplot as plt

def plot_endpoints(directory="endpoints"):
    """
    Reads all .txt files in the specified directory, parses tuples of the form (x1, x2),
    and creates side-by-side subplots with the plot title "Time vs. {Property}".
    
    Parameters:
    - directory (str): The folder containing the endpoint files.
    """
    # Get all .txt files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    n_files = len(files)
    
    if n_files == 0:
        print(f"No .txt files found in the directory: {directory}")
        return
    
    # Create subplots horizontally
    fig, axes = plt.subplots(1, n_files, figsize=(6 * n_files, 4))
    
    # In case there's only one file, convert axes to a list for uniform processing.
    if n_files == 1:
        axes = [axes]
    
    # Iterate through each file and plot its data
    for ax, file in zip(axes, files):
        # Extract property name from file name (without extension)
        property_name = os.path.splitext(file)[0]
        x1_values = []
        x2_values = []
        file_path = os.path.join(directory, file)
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                
                # Remove enclosing parentheses if they exist
                if line.startswith('(') and line.endswith(')'):
                    line = line[1:-1]
                
                # Split the line based on comma
                parts = line.split(',')
                if len(parts) == 2:
                    try:
                        # Convert both parts to float after stripping whitespace
                        x1 = float(parts[0].strip())
                        x2 = float(parts[1].strip())
                    except ValueError:
                        # Skip lines that can't be parsed into two floats
                        continue
                    x1_values.append(x1)
                    x2_values.append(x2)
        
        # Plot the parsed data into the current subplot
        ax.plot(x1_values, x2_values, marker='o', linestyle='-')
        ax.set_xlabel("Time (unit)")
        ax.set_ylabel(property_name)
        ax.set_title(f"Time vs. {property_name}")
    
    plt.tight_layout()
    plt.show()

# To run the function, simply call:
plot_endpoints()
