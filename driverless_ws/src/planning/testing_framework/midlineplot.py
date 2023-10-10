import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def plot_lin_input(blue_cone_pos, yellow_cone_pos, midline):
    bx = []
    by = []
    
    yx = []
    yy = []
    
    mx = []
    my = []
    
    for i in range(len(blue_cone_pos)):
        bx.append(blue_cone_pos[i][0])
        by.append(blue_cone_pos[i][1])
    
    for j in range(len(yellow_cone_pos)):
        yx.append(yellow_cone_pos[j][0])
        yy.append(yellow_cone_pos[j][1])
        
    for k in range(len(midline)):
        mx.append(midline[k][0])
        my.append(midline[k][1])
        
    mx_interp = np.linspace(np.min(mx), np.max(mx), 100)
    my_linear = interp1d(mx, my)


    plt.plot(bx, by, 'b*', yx, yy, 'y*', mx, my, 'r*', mx_interp, my_linear(mx_interp), "black")
    plt.show()

def plot_cub_input(blue_con_pos, yellow_cone_pos, midline):
    return 0;

def test():
    blue_cone_pos = [(0,0), (1, 1), (2, 2), (3,3)]
    yellow_cone_pos = [(0, 1), (1, 2), (2, 3), (3, 4)]
    midline = [(1, 0), (3, 1), (4, 2), (0, 3)]
    plot_lin_input(blue_cone_pos, yellow_cone_pos, midline)
    
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