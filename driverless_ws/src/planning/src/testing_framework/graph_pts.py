import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to read points from CSV file
def read_points_from_file(file_path):
    points = np.loadtxt(file_path, delimiter=',')
    return points

# Read points from the CSV file
file_path = 'clicked_points.csv'
clicked_points = read_points_from_file(file_path)

# Plot the points using Matplotlib
plt.scatter(clicked_points[:, 1], clicked_points[:, 0], color='red', marker='o')
plt.title('Clicked Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()
