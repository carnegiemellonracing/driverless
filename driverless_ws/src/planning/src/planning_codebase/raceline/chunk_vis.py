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

with open("chunk_vis_blue.txt") as f:
    lines = f.readlines() # list containing lines of file

blue_chunks_x = []
blue_chunks_y = []
cur_chunk_x = np.array([])
cur_chunk_y = np.array([])

for line in lines:
    line = line.strip()
    if line != "#" and len(line) > 0:
        comma_idx = line.index(',')
        cur_chunk_x = np.append(cur_chunk_x, [float(line[0:comma_idx])])
        cur_chunk_y = np.append(cur_chunk_y, [float(line[comma_idx+1:])])
    elif line == "#":
        blue_chunks_x.append(cur_chunk_x)
        blue_chunks_y.append(cur_chunk_y)

        cur_chunk_x = np.array([])
        cur_chunk_y = np.array([])

print("DONE READING BLUE CHUNKS")
plt.style.use('dark_background')
for (j,(l_x, l_y)) in enumerate(zip(blue_chunks_x, blue_chunks_y)):
    for i in range(len(l_x)):
        if j % 2 == 0:
            print(l_x[i], l_y[i], "BLUE")
            plt.scatter(l_x[i], l_y[i], color='blue')
        else:
            print(l_x[i], l_y[i], "LBLUE")
            plt.scatter(l_x[i], l_y[i], color='lightskyblue')
# plt.show()

with open("chunk_vis_yellow.txt") as f:
    lines = f.readlines() # list containing lines of file

yellow_chunks_x = []
yellow_chunks_y = []
cur_chunk_x = np.array([])
cur_chunk_y = np.array([])

for line in lines:
    line = line.strip()
    if line != "#" and len(line) > 0:
        comma_idx = line.index(',')
        cur_chunk_x = np.append(cur_chunk_x, [float(line[0:comma_idx])])
        cur_chunk_y = np.append(cur_chunk_y, [float(line[comma_idx+1:])])
    elif line == "#":
        yellow_chunks_x.append(cur_chunk_x)
        yellow_chunks_y.append(cur_chunk_y)

        cur_chunk_x = np.array([])
        cur_chunk_y = np.array([])

for (j, (l_x, l_y)) in enumerate(zip(yellow_chunks_x, yellow_chunks_y)):
    for i in range(len(l_x)):
        if j % 2 == 0:
            plt.scatter(l_x[i], l_y[i], color='yellow')
        else:
            plt.scatter(l_x[i], l_y[i], color='red')

plt.show()



