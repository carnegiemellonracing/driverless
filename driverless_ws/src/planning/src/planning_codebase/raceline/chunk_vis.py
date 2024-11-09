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

with open("blue.txt") as b:
    linesBlue = b.readlines() # list containing lines of file
    
with open("yellow.txt") as y:
    linesYellow = y.readlines() # list containing lines of file

b_chunks_x, y_chunks_x = [], []
b_chunks_y, y_chunks_y = [], []
b_cur_chunk_x, y_cur_chunk_x = np.array([]), np.array([])
b_cur_chunk_y, y_cur_chunk_y= np.array([]), np.array([])

for line in linesBlue:
    line = line.strip()
    if len(linesBlue) == 0 or line == "#":
        continue
    elif line != "#" and len(line) > 0:
        comma_idx = line.index(',')
        b_cur_chunk_x = np.append(b_cur_chunk_x, [float(line[0:comma_idx])])
        b_cur_chunk_y = np.append(b_cur_chunk_y, [float(line[comma_idx+1:])])
    elif line == "#":
        b_chunks_x.append(b_cur_chunk_x)
        b_chunks_y.append(b_cur_chunk_y)

        y_cur_chunk_x = np.array([])
        y_cur_chunk_y = np.array([])
        
for line in linesYellow:
    line = line.strip()
    if len(linesYellow) == 0 or line == "#":
        continue
    elif line != "#" and len(line) > 0:
        comma_idx = line.index(',')
        y_cur_chunk_x = np.append(y_cur_chunk_x, [float(line[0:comma_idx])])
        y_cur_chunk_y = np.append(y_cur_chunk_y, [float(line[comma_idx+1:])])
    elif line == "#":
        y_chunks_x.append(y_cur_chunk_x)
        y_chunks_y.append(y_cur_chunk_y)

        y_cur_chunk_x = np.array([])
        y_cur_chunk_y = np.array([])


plt.style.use('dark_background')

i = 0
for (c_x, c_y) in zip(b_cur_chunk_x, b_cur_chunk_y):
        if i % 2 == 0:
            plt.scatter(c_x, c_y, color='blue')
        else:
            plt.scatter(c_x, c_y, color='green')
        i += 1
for (c_x, c_y) in zip(y_cur_chunk_x, y_cur_chunk_y):
        if i % 2 == 0:
            plt.scatter(c_x, c_y, color='yellow')
        else:
            plt.scatter(c_x, c_y, color='red')
        i += 1


def ator(a):
    return a * (np.pi / 180.0)

def ftom(f):
    return 0.3048 * f

yellow_cones = np.array([
    [0, 0, 0],
    [0, 2, 0],
    [0, 4, 0],
    [0, 6, 0],
    [0, 8, 0],
    [0, 10, 0],
    [0, 12, 0],
    [0, 14, 0],
    [0, 16, 0],
    [0, 18, 0],
    [0, 20, 0],
    [0, 22, 0],
    [0 - 6 + 6 * np.cos(ator(30)), 22 + 6 * np.sin(ator(30)), 0],
    [0 - 6 + 6 * np.cos(ator(60)), 22 + 6 * np.sin(ator(60)), 0],
    [-6, 28, 0],
    [-8, 28, 0],
    [-10, 28, 0],
    [-12, 28, 0],
    [-14, 28, 0],
    [-14 + 6 * np.cos(ator(120)), 28 - 6 + 6 * np.sin(ator(120)), 0],
    [-14 + 6 * np.cos(ator(150)), 28 - 6 + 6 * np.sin(ator(150)), 0],
    [-20, 22, 0],
    [-20 + ftom(4), 20, 0],
    [-20 + ftom(6), 18, 0],
    [-20 + ftom(2), 16, 0],
    [-20 - ftom(2), 14, 0],
    [-20 - ftom(6), 12, 0],
    [-20 - ftom(4), 10, 0],
    [-20, 8, 0],
    [-20, 6, 0],
    [-20, 4, 0],
    [-16+2 - 6*np.cos(ator(30)),4-6*np.sin(ator(30)),0],
    [-16+2 - 6*np.cos(ator(60)),4-6*np.sin(ator(90)),0],
    [-14,-2,0],
    [-12,-2,0],
    [-10,-2,0],
    [-8,-2,0],
    [-6,-2,0],
    [-4,-2,0]
])

blue_cones = np.array([
    [-4, 0, 0],
    [-4, 2, 0],
    [-4, 4, 0],
    [-4, 6, 0],
    [-4, 8, 0],
    [-4, 10, 0],
    [-4, 12, 0],
    [-4, 14, 0],
    [-4, 16, 0],
    [-4, 18, 0],
    [-4, 20, 0],
    [-4, 22, 0],
    [-4-2 + 2 * np.cos(ator(30)), 22 + 2 * np.sin(ator(30)), 0],
    [-4-2 + 2 * np.cos(ator(60)), 22 + 2 * np.sin(ator(60)), 0],
    [-6, 24, 0],
    [-8, 24, 0],
    [-10, 24, 0],
    [-12, 24, 0],
    [-14, 24, 0],
    [-14 + 2 * np.cos(ator(120)), 24 - 2 + 2 * np.sin(ator(120)), 0],
    [-14 + 2 * np.cos(ator(150)), 24 - 2 + 2 * np.sin(ator(150)), 0],
    [-16, 22, 0],
    [-16 + ftom(4), 20, 0],
    [-16 + ftom(6), 18, 0],
    [-16 + ftom(2), 16, 0],
    [-16 - ftom(2), 14, 0],
    [-16 - ftom(6), 12, 0],
    [-16 - ftom(4), 10, 0],
    [-16, 8, 0],
    [-16, 6, 0],
    [-16, 4, 0],
    [-16+2 - 2*np.cos(ator(30)),4-2*np.sin(ator(30)),0],
    [-16+2 - 2*np.cos(ator(60)),4-2*np.sin(ator(90)),0],
    [-14,2,0],
    [-12,2,0],
    [-10,2,0],
    [-8,2,0],
    [-6,2,0],
    [-4,2,0]
])

plt.scatter(yellow_cones[:, 0], yellow_cones[:, 1], c="orange"),
plt.scatter(blue_cones[:, 0], blue_cones[:, 1], c="red")

ax = plt.gca()
ax.set_aspect("equal", adjustable="box")

plt.show()



