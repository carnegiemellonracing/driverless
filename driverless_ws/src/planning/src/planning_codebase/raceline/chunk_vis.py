"""Visualizes chunks from chunk_vis.txt file
After generating the chunks from the chunking algorithm, print out the
coordinates of the blue and yellow cones into chunk_vis.txt.

Please follow the following format:
1.) each cone is in the listed in the form: x,y
2.) End each chunk with a # symbol (including the last)
ex.
1,2
3,4
5,6
#
4,5
6,7
#
"""

import numpy as np
import matplotlib.pyplot as plt
import time

with open("chunk_vis.txt") as f:
    lines = f.readlines() # list containing lines of file

chunks_x = []
chunks_y = []
cur_chunk_x = np.array([])
cur_chunk_y = np.array([])

for line in lines:
    line = line.strip()
    if len(lines) == 0 or line == "#":
        continue
    elif line != "#" and len(line) > 0:
        comma_idx = line.index(',')
        cur_chunk_x = np.append(cur_chunk_x, [float(line[0:comma_idx])])
        cur_chunk_y = np.append(cur_chunk_y, [float(line[comma_idx+1:])])
    elif line == "#":
        chunks_x.append(cur_chunk_x)
        chunks_y.append(cur_chunk_y)

        cur_chunk_x = np.array([])
        cur_chunk_y = np.array([])

i = 0
plt.style.use('dark_background')

for (c_x, c_y) in zip(cur_chunk_x, cur_chunk_y):
    if i == 0:
        plt.scatter(c_x, c_y, color='white')
        i = 1
    elif i == 1:
        plt.scatter(c_x, c_y, color='green')
        i = 0

plt.show()



