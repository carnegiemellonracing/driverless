import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate 
from mpl_toolkits import mplot3d

def plot_lin_input(blue_cone_pos, yellow_cone_pos, midline):
    bx = []
    by = []
    bz = []
    
    yx = []
    yy = []
    yz = []
    
    ax = plt.axes(projection = '3d')
    
    for i in range(len(blue_cone_pos)):
        bx.append(blue_cone_pos[i][0])
        by.append(blue_cone_pos[i][1])
        bz.append(blue_cone_pos[i][2])
    
    for j in range(len(yellow_cone_pos)):
        yx.append(yellow_cone_pos[j][0])
        yy.append(yellow_cone_pos[j][1])
        yz.append(blue_cone_pos[j][2])

    ax.plot3D(bx, by, bz, 'b*', yx, yy, yz, 'y*')
    plt.show()

def plot_cub_input(blue_con_pos, yellow_cone_pos, midline):
    return 0;

def test():
    blue_cone_pos = [(0,0, 0), (1, 1, 1), (2, 2, 2), (3,3, 3)]
    yellow_cone_pos = [(0, 1, 1), (1, 2, 2), (2, 3, 3), (3, 4, 4)]
    plot_lin_input(blue_cone_pos, yellow_cone_pos, [0])
    
test()

# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
# import numpy as np
# from scipy import interpolate 


# def plot_lin_input(blue_cone_pos, yellow_cone_pos, midline):
#     bx = []
#     by = []
#     bz = []
    
#     yx = []
#     yy = []
#     bz = []
    
#     for i in range(len(blue_cone_pos)):
#         bx.append(blue_cone_pos[i.x])
#         by.append(blue_cone_pos[i.y])
#         bz.append(blue_cone_pos(i.z))
    
#     for j in range(len(yellow_cone_pos)):
#         yx.append(yellow_cone_pos[j.x])
#         yy.append(yellow_cone_pos[j.y])
#         bz.append(blue_cone_pos[i.z])
        
#     plt.plot(bx, by, "b*", yx, yy, "y*")
    
#     fig = plt.figure()
#     ax = plt.aces(projection='3d')
#     plt.show()

# def plot_cube_input(blue_con_pos, yellow_cone_pos, midline):
#     return 0;