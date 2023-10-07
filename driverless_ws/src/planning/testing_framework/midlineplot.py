import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate 

def plot_lin_input(blue_cone_pos, yellow_cone_pos, midline):
    bx = []
    by = []
    
    yx = []
    yy = []
    
    for i in range(len(blue_cone_pos)):
        bx.append(blue_cone_pos[i][0])
        by.append(blue_cone_pos[i][1])
    
    for j in range(len(yellow_cone_pos)):
        yx.append(yellow_cone_pos[j][0])
        yy.append(yellow_cone_pos[j][1])

    plt.plot(bx, by, "b*", yx, yy, "y*")
    plt.show()

def plot_cub_input(blue_con_pos, yellow_cone_pos, midline):
    return 0;

def test():
    blue_cone_pos = [(0,0), (1, 1), (2, 2), (3,3)]
    yellow_cone_pos = [(0, 1), (1, 2), (2, 3), (3, 4)]
    plot_lin_input(blue_cone_pos, yellow_cone_pos, [0])
    
test()